"""Reference-free native Flashlight word decoding in the isolated CPython 3.12 env."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from btt25.host_accounting import HostAccounting, WATCHDOG_SECONDS


TOKENS = (
    'BLANK', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M',
    'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
    'V', 'W', 'Y', 'Z', 'ZH', 'SIL',
)
PHONE_IDS = {phone: index for index, phone in enumerate(TOKENS)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_tokens(path=None):
    if path is None:
        return TOKENS
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    tokens = value.get('tokens') if isinstance(value, dict) else value
    if tuple(tokens or ()) != TOKENS:
        raise ValueError('The token manifest must match all 41 official tokens in order')
    if isinstance(value, dict) and (value.get('blank_id', 0) != 0 or value.get('boundary_id', 40) != 40):
        raise ValueError('The token manifest must use blank 0 and SIL 40')
    return TOKENS


def valid_word(word):
    # Match the evaluator's lowercase letters/apostrophes/hyphens without deleting characters.
    return bool(re.fullmatch(r"[a-z'-]+", word) and re.search('[a-z]', word))


def pronunciation_ids(phones):
    if not phones or any(phone not in PHONE_IDS or PHONE_IDS[phone] in (0, 40) for phone in phones):
        raise ValueError('A pronunciation must contain only real official phonemes')
    return tuple(PHONE_IDS[phone] for phone in phones)


def build_lexicon(cmudict_path, output_path, tokens_manifest=None):
    load_tokens(tokens_manifest)
    pronunciations = set()
    rejected = Counter()
    with Path(cmudict_path).open(encoding='utf-8') as source:
        for line in source:
            line = line.strip()
            if not line or line.startswith((';;;', '#')):
                continue
            pieces = line.split()
            if len(pieces) < 2:
                rejected['missing_pronunciation'] += 1
                continue
            word = re.sub(r'\(\d+\)$', '', pieces[0]).lower()
            if not valid_word(word):
                rejected['invalid_word'] += 1
                continue
            phones = tuple(re.sub('[012]$', '', phone.upper()) for phone in pieces[1:])
            try:
                pronunciation_ids(phones)
            except ValueError:
                rejected['invalid_phone'] += 1
                continue
            pronunciations.add((word, phones))
    if not pronunciations:
        raise ValueError('No valid CMUdict pronunciations remain')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(f"{word} {' '.join(phones)}\n" for word, phones in sorted(pronunciations)), encoding='utf-8')
    summary = {
        'version': 1, 'source': str(Path(cmudict_path).resolve()), 'source_sha256': sha256(cmudict_path),
        'lexicon_sha256': sha256(output_path), 'tokens_hash': object_hash(TOKENS),
        'words': len({word for word, _ in pronunciations}), 'pronunciations': len(pronunciations),
        'rejected_lines': sum(rejected.values()), 'rejection_counts': dict(rejected),
        'normalization': 'lowercase words; preserve apostrophes/hyphens; strip numbered variants and phone stress 0/1/2',
        'references_loaded': False,
    }
    write_json(output_path.with_suffix(output_path.suffix + '.manifest.json'), summary)
    return summary


def load_lexicon(path):
    entries = {}
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith((';;;', '#')):
                continue
            pieces = line.split()
            if len(pieces) < 2 or not valid_word(pieces[0]):
                raise ValueError(f'Invalid lexicon word at line {number}')
            phones = pronunciation_ids(pieces[1:])
            entries.setdefault(pieces[0], set()).add(phones)
    if not entries:
        raise ValueError('Lexicon contains no valid words')
    return {word: sorted(phones) for word, phones in sorted(entries.items())}


def load_unigrams(path):
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Unigrams must be a word-to-count mapping')
    if 'counts' in value:
        if value.get('source_split', 'train') != 'train':
            raise ValueError('Unigrams must come from the train split')
        value = value['counts']
    if not isinstance(value, dict) or not value:
        raise ValueError('Unigram counts must be nonempty')
    for word, count in value.items():
        if not valid_word(word) or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f'Invalid unigram word or count: {word}')
    if sum(value.values()) <= 0:
        raise ValueError('Unigram counts must have positive total mass')
    return value


def log_softmax(logits):
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != 41:
        raise ValueError('Logits must have nonempty shape (time, 41)')
    if logits.dtype != np.float32:
        raise ValueError('Logits must use float32')
    if not np.isfinite(logits).all():
        raise ValueError('Logits must be finite')
    shifted = logits - logits.max(axis=1, keepdims=True)
    return np.ascontiguousarray(shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True)), dtype=np.float32)


def collapse_path(path):
    labels, previous = [], None
    for token in path:
        token = int(token)
        if token != previous and token != 0:
            labels.append(token)
        previous = token
    return labels


def kenlm_configuration(model, weight):
    """Weights apply to natural log probability; the optional backend starts at 1."""
    weight = float(1.0 if model is not None else 0.0) if weight is None else float(weight)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('lm_weight must be finite and nonnegative')
    if model is None:
        if weight:
            raise ValueError('A positive lm_weight requires a KenLM model')
        return None, weight
    model = Path(model).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f'KenLM model is missing or not a file: {model}')
    return model, weight


class NativeDecoder:
    def __init__(self, lexicon, beam=50, nbest=10, unigrams=None, unigram_weight=0.0, beam_threshold=25.0,
                 kenlm_model=None, lm_weight=None):
        kenlm_model, self.lm_weight = kenlm_configuration(kenlm_model, lm_weight)
        from flashlight.lib.text.decoder import (
            CriterionType, LexiconDecoder, LexiconDecoderOptions, SmearingMode, Trie, ZeroLM,
        )
        from flashlight.lib.text.dictionary import Dictionary

        if not 1 <= nbest <= beam or not np.isfinite(beam_threshold) or beam_threshold <= 0:
            raise ValueError('Require 1 <= nbest <= beam and a positive beam threshold')
        if not np.isfinite(unigram_weight) or unigram_weight < 0 or (unigram_weight and not unigrams):
            raise ValueError('A positive unigram weight requires training unigram counts')
        self.nbest, self.unigrams, self.unigram_weight = nbest, unigrams, unigram_weight
        self.word_dictionary = Dictionary()
        self.word_dictionary.add_entry('<unk>')
        for word, variants in sorted(lexicon.items()):
            if not valid_word(word):
                raise ValueError(f'Invalid lexicon word: {word}')
            self.word_dictionary.add_entry(word)
            for phones in variants:
                if not phones or any(phone not in range(1, 40) for phone in phones):
                    raise ValueError(f'Invalid pronunciation IDs: {word}')

        self.uses_kenlm = kenlm_model is not None
        if self.uses_kenlm:
            try:
                from flashlight.lib.text.decoder.kenlm import KenLM
            except (ImportError, OSError) as error:
                raise RuntimeError('Requested native KenLM backend is unavailable') from error
            try:
                # KenLM snapshots the entire word-to-LM index map in its constructor.
                self.lm = KenLM(str(kenlm_model), self.word_dictionary)
            except Exception as error:
                raise RuntimeError(f'Unable to load requested KenLM model: {kenlm_model}') from error
            self.kenlm_status = {'available': True, 'used': True, 'model_path': str(kenlm_model),
                                 'model_sha256': sha256(kenlm_model), 'native_log_base': 10,
                                 'score_log_base': 'e', 'lm_weight': self.lm_weight,
                                 'native_lm_weight': float(self.lm_weight * np.log(10.0)),
                                 'sentence_start': 'BOS', 'sentence_end': 'EOS',
                                 'lexicon_oov': 'KenLM unknown-word probability; output words restricted to the lexicon'}
        else:
            self.lm = ZeroLM()
            try:
                from flashlight.lib.text.decoder.kenlm import KenLM  # noqa: F401
                self.kenlm_status = {'available': True, 'used': False}
            except (ImportError, OSError) as error:
                self.kenlm_status = {'available': False, 'used': False, 'blocker': 'KenLM native DLL dependency unavailable', 'detail': str(error).strip()}

        trie = Trie(41, 40)
        start = self.lm.start(False) if self.uses_kenlm else None
        for word, variants in sorted(lexicon.items()):
            word_id = self.word_dictionary.get_index(word)
            # Native KenLM returns log10. Smearing stays on that scale; the
            # decoder multiplies both trie deltas and completed-word scores.
            score = float(self.lm.score(start, word_id)[1]) if self.uses_kenlm else 0.0
            if not np.isfinite(score):
                raise ValueError(f'KenLM returned a nonfinite trie score for word: {word}')
            for phones in variants:
                trie.insert(list(phones), word_id, score)
        trie.smear(SmearingMode.MAX)
        self.trie = trie
        native_weight = float(self.lm_weight * np.log(10.0)) if self.uses_kenlm else 0.0
        options = LexiconDecoderOptions(beam, 41, beam_threshold, native_weight, 0.0, float('-inf'), 0.0, True, CriterionType.CTC)
        self.decoder = LexiconDecoder(options, trie, self.lm, 40, 0, 0, [], False)
        self.mode = ('lexicon_kenlm_unigram_rerank' if unigram_weight else 'lexicon_kenlm') if self.uses_kenlm else ('lexicon_unigram_rerank' if unigram_weight else 'lexicon_only')
        self.prior_denominator = sum(unigrams.values()) + len(lexicon) if unigrams else None

    def decode(self, logits):
        emissions = log_softmax(logits)
        results = self.decoder.decode(emissions.ctypes.data, *emissions.shape)
        unique = {}
        for result in results:
            word_ids = [int(word) for word in result.words if word >= 0]
            if 0 in word_ids or not np.isfinite(result.score):
                continue
            words = [self.word_dictionary.get_entry(word) for word in word_ids]
            if '<unk>' in words:
                continue
            prior = sum(np.log((self.unigrams.get(word, 0) + 1) / self.prior_denominator) for word in words) if self.unigrams else None
            contribution = float(self.unigram_weight * prior) if prior is not None else 0.0
            # Flashlight adds one artificial root and one final boundary frame.
            frame_tokens = list(result.tokens)
            if len(frame_tokens) != len(emissions) + 2:
                raise ValueError('Unexpected native frame-token boundary contract')
            candidate = {
                'text': ' '.join(words), 'words': words, 'word_ids': word_ids,
                'phoneme_ids': collapse_path(frame_tokens[1:-1]),
                'component_scores': {'decoder_total': float(result.score), 'acoustic': float(result.emittingModelScore),
                                     'native_lm': float(result.lmScore), 'unigram_logprob': float(prior) if prior is not None else None,
                                     'unigram_weight': float(self.unigram_weight), 'unigram_contribution': contribution,
                                     'rank_score': float(result.score) + contribution},
            }
            if self.uses_kenlm:
                logprob = float(result.lmScore) * float(np.log(10.0))
                candidate['component_scores'].update(kenlm_log10=float(result.lmScore), kenlm_logprob=logprob,
                                                     lm_weight=self.lm_weight, kenlm_contribution=self.lm_weight * logprob)
            key = tuple(words)
            if key not in unique or candidate['component_scores']['rank_score'] > unique[key]['component_scores']['rank_score']:
                unique[key] = candidate
        candidates = sorted(unique.values(), key=lambda row: (-row['component_scores']['rank_score'], row['text']))[:self.nbest]
        for rank, candidate in enumerate(candidates, 1):
            candidate['rank'] = rank
        return candidates


class ProcessMonitor:
    """Measure this CPU worker only; the parent may monitor the full job tree."""

    def __init__(self, output, max_seconds):
        self.output, self.max_seconds = Path(output), max_seconds
        self.host = HostAccounting()
        self.max_seconds = min(self.max_seconds, WATCHDOG_SECONDS)
        self.soft_ram, self.hard_ram = self.host.soft, self.host.hard
        self.ram_source = self.host.limit_source
        self.minimum_required_available = 2_000_000_000
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.reason = None
        self.observations = []

    def sample(self):
        with self.lock:
            sample = self.host.sample()
            self.observations.append(sample)
            self.samples.write(json.dumps(sample) + '\n')
            self.samples.flush()
            if self.reason is None:
                self.reason = ('worker wall-clock deadline' if sample['elapsed_seconds'] > self.max_seconds
                               else self.host.stop_reason(sample, self.minimum_required_available))

    def loop(self):
        while not self.stop.wait(1.0):
            self.sample()

    def check(self):
        stop_file = os.environ.get('BTT_STOP_FILE')
        if stop_file and Path(stop_file).exists():
            self.reason = 'whole-job supervisor requested a safe stop'
        if self.reason:
            raise TimeoutError(self.reason)

    def __enter__(self):
        self.samples = self.output.with_suffix('.samples.jsonl').open('w', encoding='utf-8')
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, typ, error, traceback):
        self.stop.set()
        self.thread.join(timeout=3)
        self.sample()
        self.samples.close()
        def peak(key):
            return max((s[key] for s in self.observations if s.get(key) is not None), default=None)

        final = self.observations[-1]
        write_json(self.output, {**self.host.describe(), 'device': 'CPU',
                                'elapsed_seconds': time.monotonic() - self.started,
                                'sample_count': len(self.observations), 'sampling_seconds': 1,
                                'peak_rss_bytes': peak('tree_rss_bytes'),
                                'peak_private_bytes': peak('tree_private_bytes'),
                                'peak_pss_bytes': peak('tree_pss_bytes'), 'peak_uss_bytes': peak('tree_uss_bytes'),
                                'peak_accounted_host_bytes': peak('accounted_host_bytes'),
                                'peak_cgroup_bytes': peak('cgroup_peak_bytes'),
                                'cgroup_events': final['cgroup_events'],
                                'cpu': {k: v for k, v in final.items() if k.startswith('cpu_')},
                                'sampler_seconds': sum(s['sampler_seconds'] for s in self.observations),
                                'minimum_required_system_available_bytes': self.minimum_required_available,
                                'ceiling_method': 'Cooperative checks between trials; the scheduler can terminate a native call',
                                'minimum_system_available_bytes': min(s['system_available_bytes'] for s in self.observations),
                                'stop_reason': self.reason,
                                'outcome': 'budget_stopped' if isinstance(error, (TimeoutError, KeyboardInterrupt)) or (self.reason and error is None)
                                else 'failed' if error else 'complete'})



def indexed(rows, name):
    result = {}
    for row in rows:
        trial = row.get('trial_id')
        if not isinstance(trial, str) or not trial or trial in result:
            raise ValueError(f'Invalid or duplicate trial ID in {name}')
        result[trial] = row
    return result


def run_decode(logit_run, lexicon, output_dir, manifest, unigrams=None, tokens_manifest=None,
               beam=50, nbest=10, unigram_weight=0.0, beam_threshold=25.0, max_seconds=28200,
               kenlm_model=None, lm_weight=None):
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('max_seconds must be positive')
    kenlm_model, lm_weight = kenlm_configuration(kenlm_model, lm_weight)
    load_tokens(tokens_manifest)
    source, output = Path(logit_run).resolve(), Path(output_dir).resolve()
    if source == output:
        raise ValueError('Decoder output must differ from its source logit run')
    source_cache = json.loads((source / 'cache_manifest.json').read_text(encoding='utf-8'))
    source_status = json.loads((source / 'run_manifest.json').read_text(encoding='utf-8'))
    if source_status.get('state') not in ('complete', 'diagnostic_partial'):
        raise ValueError('Source inference must finish before word decoding')
    records = json.loads(Path(manifest).read_text(encoding='utf-8'))
    if not isinstance(records, list) or source_cache.get('manifest_hash') != object_hash(records):
        raise ValueError('Data manifest hash differs from the logit cache')
    if source_cache.get('dtype') != 'float32' or source_cache.get('references_loaded') is not False:
        raise ValueError('Expected a reference-free float32 logit cache')
    record_map = indexed(records, 'data manifest')
    source_rows = read_jsonl(source / 'predictions.jsonl')
    source_map = indexed(source_rows, 'source predictions')
    if not source_rows or len(source_rows) != source_cache.get('expected_trials'):
        raise ValueError('Source predictions do not cover their declared trial count')
    for trial, row in source_map.items():
        record = record_map.get(trial)
        if record is None or record.get('split') != source_cache.get('partition') or row.get('status') != 'ok':
            raise ValueError(f'Unknown or invalid source trial: {trial}')
        if row.get('official_order_index') != record.get('official_order_index') or row.get('split') != record.get('split'):
            raise ValueError(f'Source trial order/split mismatch: {trial}')
    partition_ids = {trial for trial, record in record_map.items() if record.get('split') == source_cache.get('partition')}
    if source_status['state'] == 'complete' and set(source_map) != partition_ids:
        raise ValueError('Complete source run does not have full partition coverage')
    source_rows.sort(key=lambda row: row['official_order_index'])
    cache = {
        'version': 1, 'worker_sha256': sha256(__file__), 'source_logit_run': str(source), 'source_predictions_path': str(source / 'predictions.jsonl'),
        'source_predictions_sha256': sha256(source / 'predictions.jsonl'), 'source_cache_manifest': source_cache,
        'data_manifest_path': str(Path(manifest).resolve()), 'data_manifest_sha256': sha256(manifest),
        'lexicon_path': str(Path(lexicon).resolve()), 'lexicon_sha256': sha256(lexicon),
        'tokens_hash': object_hash(TOKENS), 'unigrams_path': str(Path(unigrams).resolve()) if unigrams else None,
        'unigrams_sha256': sha256(unigrams) if unigrams else None,
        'decoder': {'backend': 'flashlight-text', 'version': importlib.metadata.version('flashlight-text'),
                    'variant': os.environ.get('BTT_FLASHLIGHT_VARIANT', 'v0.0.7'),
                    'numpy_version': np.__version__, 'beam': beam, 'beam_size_token': 41, 'beam_threshold': beam_threshold,
                    'nbest': nbest, 'blank_id': 0, 'sil_id': 40, 'log_add': True, 'unigram_weight': unigram_weight,
                    'lm': 'ZeroLM', 'unknown_words': 'forbidden'},
        'references_loaded': False,
    }
    if kenlm_model is not None:
        cache.update(kenlm_model_path=str(kenlm_model), kenlm_model_sha256=sha256(kenlm_model))
        cache['decoder'].update(lm='KenLM', lm_weight=lm_weight, score_log_base='e', native_lm_log_base=10,
                                native_lm_weight=float(lm_weight * np.log(10.0)), trie_score_log_base=10,
                                sentence_start='BOS', sentence_end='EOS',
                                lexicon_oov='KenLM unknown-word probability; output words restricted to the lexicon')
    cache['manifest_hash'] = object_hash(cache)
    output.mkdir(parents=True, exist_ok=True)
    cache_path, prediction_path, candidates_path = output / 'cache_manifest.json', output / 'predictions.jsonl', output / 'candidates.jsonl'
    if cache_path.exists():
        if json.loads(cache_path.read_text(encoding='utf-8')) != cache:
            raise ValueError('Decoder cache mismatch; use a new output directory')
    elif any(path.exists() for path in (prediction_path, candidates_path)):
        raise ValueError('Decoder outputs exist without a cache manifest')
    existing = indexed(read_jsonl(prediction_path), 'decoded predictions') if prediction_path.exists() else {}
    existing_candidates = indexed(read_jsonl(candidates_path), 'decoded candidates') if candidates_path.exists() else {}
    if set(existing) != set(existing_candidates) or not set(existing).issubset(source_map):
        raise ValueError('Incremental prediction/candidate trial IDs disagree')
    for trial, row in existing.items():
        packet = existing_candidates[trial]
        if (row.get('logit_sha256') != source_map[trial].get('logit_sha256')
                or row.get('decoder_cache_hash') != cache['manifest_hash']
                or packet.get('decoder_cache_hash') != cache['manifest_hash']
                or packet.get('logit_sha256') != source_map[trial].get('logit_sha256')):
            raise ValueError('Incremental prediction cache mismatch')
    write_json(cache_path, cache)
    write_json(output / 'config.resolved.yaml', {'cache':cache, 'max_seconds':max_seconds})
    status = {'state': 'decoding', 'started_at': datetime.now(timezone.utc).isoformat(), 'expected_trials': len(source_rows),
              'saved_trials': len(existing), 'references_loaded': False, 'decoder_cache_hash': cache['manifest_hash'],
              'source_predictions_path': cache['source_predictions_path'], 'source_state': source_status['state']}
    write_json(output / 'run_manifest.json', status)
    durations = []
    try:
        with ProcessMonitor(output / 'resources.json', max_seconds) as monitor:
            decoder = NativeDecoder(load_lexicon(lexicon), beam, nbest, load_unigrams(unigrams), unigram_weight, beam_threshold,
                                    kenlm_model=kenlm_model, lm_weight=lm_weight)
            if kenlm_model is not None and decoder.kenlm_status['model_sha256'] != cache['kenlm_model_sha256']:
                raise ValueError('KenLM model changed while loading; use a stable model file')
            status.update(decoder_mode=decoder.mode, kenlm=decoder.kenlm_status)
            write_json(output / 'run_manifest.json', status)
            with prediction_path.open('a', encoding='utf-8') as predictions, candidates_path.open('a', encoding='utf-8') as candidate_stream:
                for row in source_rows:
                    monitor.check()
                    trial = row['trial_id']
                    logit_path = (source / row['logit_path']).resolve()
                    if not logit_path.is_relative_to(source) or not logit_path.is_file():
                        raise ValueError(f'Invalid logit path for trial {trial}')
                    if sha256(logit_path) != row.get('logit_sha256'):
                        raise ValueError(f'Logit hash mismatch for trial {trial}')
                    if trial in existing:
                        continue
                    started = time.monotonic()
                    logits = np.load(logit_path, allow_pickle=False)
                    if logits.shape != (row['output_length'], 41):
                        raise ValueError(f'Logit length mismatch for trial {trial}')
                    choices = decoder.decode(logits)
                    elapsed = time.monotonic() - started
                    durations.append(elapsed)
                    selected = choices[0] if choices else None
                    prediction = {key: row[key] for key in ('trial_id', 'official_order_index', 'session_id', 'split', 'input_length', 'output_length', 'logit_path', 'logit_sha256') if key in row}
                    prediction.update(input_id=trial, output_id=trial, text=selected['text'] if selected else '',
                                      words=selected['words'] if selected else [], word_ids=selected['word_ids'] if selected else [],
                                      phoneme_ids=row.get('phoneme_ids', collapse_path(logits.argmax(axis=1))),
                                      decoder_phoneme_ids=selected['phoneme_ids'] if selected else [],
                                      component_scores=selected['component_scores'] if selected else None,
                                      text_available=True, status='ok' if selected else 'decoder_error', phoneme_status='ok',
                                      decoder_mode=decoder.mode, decoder_cache_hash=cache['manifest_hash'], elapsed_seconds=elapsed)
                    if selected is None:
                        prediction['error'] = 'No lexical hypothesis survived; no word fallback was used'
                    packet = {'trial_id': trial, 'input_id': trial, 'output_id': trial, 'logit_path': str(logit_path),
                              'logit_sha256': row['logit_sha256'], 'decoder_cache_hash': cache['manifest_hash'], 'candidates': choices}
                    candidate_stream.write(json.dumps(packet, allow_nan=False) + '\n')
                    candidate_stream.flush()
                    predictions.write(json.dumps(prediction, allow_nan=False) + '\n')
                    predictions.flush()
                    existing[trial] = prediction
                    status['saved_trials'] = len(existing)
                    write_json(output / 'run_manifest.json', status)
                    if len(existing) % 100 == 0:
                        print(json.dumps({'stage': 'word_decoding', 'saved': len(existing), 'expected': len(source_rows)}), flush=True)
            monitor.check()
            status.update(state='complete' if source_status['state'] == 'complete' else 'diagnostic_partial',
                          elapsed_seconds=time.monotonic() - monitor.started,
                          p50_trial_seconds=float(np.median(durations)) if durations else None,
                          p95_trial_seconds=float(np.quantile(durations, .95)) if durations else None,
                          decoder_error_trials=sum(row['status'] != 'ok' for row in existing.values()))
    except BaseException as error:
        status.update(state='budget_stopped' if isinstance(error, (TimeoutError, KeyboardInterrupt)) else 'failed',
                      saved_trials=len(existing), error=f'{type(error).__name__}: {error}')
        write_json(output / 'run_manifest.json', status)
        write_json(output / 'metrics.json', {'scope':'reference-free word coverage','wer':None,'per':None,**status})
        raise
    write_json(output / 'run_manifest.json', status)
    write_json(output / 'metrics.json', {'scope':'reference-free word coverage','wer':None,'per':None,**status})
    (output/'report.md').write_text(
        f"# Compact word decoding\n\nSaved {status['saved_trials']} of {status['expected_trials']} predictions "
        f"using `{status['decoder_mode']}`. Reference-based scores are computed in a separate evaluation.\n",
        encoding='utf-8',
    )
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logit-run', type=Path)
    parser.add_argument('--lexicon', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--tokens-manifest', type=Path)
    parser.add_argument('--unigrams', type=Path)
    parser.add_argument('--unigram-weight', type=float, default=0.0)
    parser.add_argument('--kenlm-model', type=Path, help='Optional native KenLM ARPA or binary model; load failures stop decoding')
    parser.add_argument('--lm-weight', type=float, default=None,
                        help='Weight on KenLM natural log probability (default: 1 with a model, 0 without); native log10 converted once')
    parser.add_argument('--beam', type=int, default=50)
    parser.add_argument('--nbest', type=int, default=10)
    parser.add_argument('--beam-threshold', type=float, default=25.0)
    parser.add_argument('--max-seconds', type=float, default=28200)
    parser.add_argument('--cmudict', type=Path)
    parser.add_argument('--build-lexicon-only', action='store_true')
    args = parser.parse_args(argv)
    if args.cmudict:
        summary = build_lexicon(args.cmudict, args.lexicon, args.tokens_manifest)
        print(json.dumps(summary), flush=True)
    if args.build_lexicon_only:
        if not args.cmudict:
            parser.error('--build-lexicon-only requires --cmudict')
        return
    if not all((args.logit_run, args.output_dir, args.manifest)):
        parser.error('decoding requires --logit-run, --output-dir, and --manifest')
    status = run_decode(args.logit_run, args.lexicon, args.output_dir, args.manifest, args.unigrams,
                        args.tokens_manifest, args.beam, args.nbest, args.unigram_weight, args.beam_threshold, args.max_seconds,
                        kenlm_model=args.kenlm_model, lm_weight=args.lm_weight)
    print(json.dumps(status, allow_nan=False), flush=True)
    return status


if __name__ == '__main__':
    main()
