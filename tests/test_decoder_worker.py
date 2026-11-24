"""Synthetic decoder contracts; no neural model or participant data is loaded."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import decode_worker as worker


def native_available():
    try:
        return importlib.util.find_spec('flashlight.lib.text.decoder') is not None
    except ModuleNotFoundError:
        return False


def native_kenlm_available():
    try:
        from flashlight.lib.text.decoder.kenlm import KenLM  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def peaked_logits(path):
    values = np.full((len(path), 41), -40.0, dtype=np.float32)
    values[np.arange(len(path)), path] = 0.0
    return values


@contextlib.contextmanager
def fake_native(kenlm_error=None):
    """Exercise the worker's binding contract without native libraries or LM assets."""
    modules = {name: ModuleType(name) for name in (
        'flashlight', 'flashlight.lib', 'flashlight.lib.text',
        'flashlight.lib.text.decoder', 'flashlight.lib.text.dictionary',
        'flashlight.lib.text.decoder.kenlm',
    )}
    for module in modules.values():
        module.__path__ = []

    class Dictionary:
        def __init__(self):
            self.entries = []

        def add_entry(self, word):
            self.entries.append(word)

        def get_index(self, word):
            return self.entries.index(word)

        def get_entry(self, index):
            return self.entries[index]

    trie = mock.Mock()
    native_decoder = mock.Mock()
    native_decoder.decode.side_effect = lambda pointer, frames, tokens: [SimpleNamespace(
        words=[1], tokens=[40, 1] + [40] * frames, score=-1.0 - 6.0 * np.log(10),
        emittingModelScore=-1.0, lmScore=-3.0,
    )]
    lm = mock.Mock()
    lm.start.return_value = object()
    lm.score.side_effect = lambda state, word_id: (object(), -float(word_id))
    binding = modules['flashlight.lib.text.decoder']
    binding.CriterionType = SimpleNamespace(CTC='CTC')
    binding.SmearingMode = SimpleNamespace(MAX='MAX')
    binding.Trie = mock.Mock(return_value=trie)
    binding.ZeroLM = mock.Mock(return_value=mock.Mock())
    binding.LexiconDecoderOptions = mock.Mock()
    binding.LexiconDecoder = mock.Mock(return_value=native_decoder)
    modules['flashlight.lib.text.dictionary'].Dictionary = Dictionary
    constructor = mock.Mock(return_value=lm, side_effect=kenlm_error)
    modules['flashlight.lib.text.decoder.kenlm'].KenLM = constructor
    with mock.patch.dict(sys.modules, modules):
        yield SimpleNamespace(binding=binding, lm=lm, kenlm=constructor, trie=trie)


class KenLMContractTests(unittest.TestCase):
    def test_trie_uses_bos_native_log10_and_complete_word_dictionary(self):
        with tempfile.TemporaryDirectory() as directory, fake_native() as native:
            path = Path(directory) / 'fixture.binary'
            path.write_bytes(b'Only the mocked loader reads this fixture.')
            words_at_construction = []
            native.kenlm.side_effect = lambda path, dictionary: (
                words_at_construction.extend(dictionary.entries) or native.lm
            )
            decoder = worker.NativeDecoder(
                {'beta': [(2,)], 'alpha': [(1,), (1, 1)]},
                kenlm_model=path, lm_weight=2.0,
            )
            self.assertEqual(words_at_construction, ['<unk>', 'alpha', 'beta'])
            native.lm.start.assert_called_once_with(False)
            self.assertEqual(native.lm.score.call_args_list, [
                mock.call(native.lm.start.return_value, 1),
                mock.call(native.lm.start.return_value, 2),
            ])
            self.assertEqual(native.trie.insert.call_args_list, [
                mock.call([1], 1, -1.0), mock.call([1, 1], 1, -1.0), mock.call([2], 2, -2.0),
            ])
            native.trie.smear.assert_called_once_with('MAX')
            self.assertAlmostEqual(native.binding.LexiconDecoderOptions.call_args.args[3], 2.0 * np.log(10))
            self.assertEqual(decoder.mode, 'lexicon_kenlm')
            self.assertTrue(decoder.kenlm_status['used'])

    def test_native_lm_scores_convert_to_natural_log_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory, fake_native():
            path = Path(directory) / 'fixture.binary'
            path.write_bytes(b'fixture')
            decoder = worker.NativeDecoder({'alpha': [(1,)]}, kenlm_model=path, lm_weight=2.0)
            score = decoder.decode(peaked_logits([1, 40]))[0]['component_scores']
            self.assertEqual(score['native_lm'], -3.0)
            self.assertEqual(score['kenlm_log10'], -3.0)
            self.assertAlmostEqual(score['kenlm_logprob'], -3.0 * np.log(10))
            self.assertEqual(score['lm_weight'], 2.0)
            self.assertAlmostEqual(score['kenlm_contribution'], -6.0 * np.log(10))
            self.assertAlmostEqual(score['decoder_total'], score['acoustic'] + score['kenlm_contribution'])

    def test_requested_model_and_weight_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.binary'
            with self.assertRaisesRegex(FileNotFoundError, 'KenLM model'):
                worker.NativeDecoder({'alpha': [(1,)]}, kenlm_model=path)
            for weight in (-1.0, float('nan'), float('inf')):
                with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, 'lm_weight'):
                    worker.NativeDecoder({'alpha': [(1,)]}, lm_weight=weight)
            with self.assertRaisesRegex(ValueError, 'requires.*KenLM'):
                worker.NativeDecoder({'alpha': [(1,)]}, lm_weight=1.0)
            path.write_bytes(b'unloadable')
            with fake_native(RuntimeError('invalid binary')) as native:
                with self.assertRaisesRegex(RuntimeError, 'Unable to load.*KenLM'):
                    worker.NativeDecoder({'alpha': [(1,)]}, kenlm_model=path)
                native.binding.ZeroLM.assert_not_called()

    def test_default_zero_lm_and_optional_default_kenlm_weight(self):
        with tempfile.TemporaryDirectory() as directory, fake_native() as native:
            zero = worker.NativeDecoder({'alpha': [(1,)]})
            self.assertEqual(zero.mode, 'lexicon_only')
            native.kenlm.assert_not_called()
            self.assertEqual(native.binding.LexiconDecoderOptions.call_args.args[3], 0.0)
            self.assertNotIn('kenlm_log10', zero.decode(peaked_logits([1, 40]))[0]['component_scores'])
            path = Path(directory) / 'fixture.binary'
            path.write_bytes(b'fixture')
            worker.NativeDecoder({'alpha': [(1,)]}, kenlm_model=path)
            self.assertAlmostEqual(native.binding.LexiconDecoderOptions.call_args.args[3], np.log(10))

    def test_unavailable_backend_and_nonfinite_trie_scores_never_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.binary'
            path.write_bytes(b'fixture')
            with fake_native() as native:
                with mock.patch.dict(sys.modules, {'flashlight.lib.text.decoder.kenlm': None}):
                    with self.assertRaisesRegex(RuntimeError, 'KenLM backend is unavailable'):
                        worker.NativeDecoder({'alpha': [(1,)]}, kenlm_model=path)
                native.binding.ZeroLM.assert_not_called()
            with fake_native() as native:
                native.lm.score.side_effect = lambda state, word_id: (object(), float('nan'))
                with self.assertRaisesRegex(ValueError, 'nonfinite trie score'):
                    worker.NativeDecoder({'alpha': [(1,)]}, kenlm_model=path)
                native.binding.ZeroLM.assert_not_called()

    def test_cli_forwards_explicit_model_and_natural_log_weight(self):
        with mock.patch.object(worker, 'run_decode', return_value={'state': 'complete'}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                worker.main(['--logit-run', 'source', '--lexicon', 'lexicon.txt', '--output-dir', 'output',
                             '--manifest', 'manifest.json', '--kenlm-model', 'train.binary', '--lm-weight', '0.5'])
            self.assertEqual(run.call_args.kwargs['kenlm_model'], Path('train.binary'))
            self.assertEqual(run.call_args.kwargs['lm_weight'], 0.5)


class LexiconTests(unittest.TestCase):
    def test_cmudict_variants_stress_and_apostrophes_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cmu = root / 'cmudict.txt'
            lexicon = root / 'lexicon.txt'
            cmu.write_text(
                ";;; comment\nCAN'T K AE1 N T\nREAD R IY1 D\nREAD(2) R EH1 D\n"
                "READ(3) R EH0 D\nBAD B NOTAPHONE\nNOISE! N OY1 Z\n"
                "EMPTY\nPADDING BLANK\nBOUNDARY SIL\n",
                encoding='utf-8',
            )
            summary = worker.build_lexicon(cmu, lexicon)
            loaded = worker.load_lexicon(lexicon)
            self.assertEqual(set(loaded), {"can't", 'read'})
            self.assertEqual(len(loaded['read']), 2)
            self.assertEqual(loaded["can't"], [(20, 2, 23, 31)])
            self.assertEqual(summary['pronunciations'], 3)
            self.assertGreaterEqual(summary['rejected_lines'], 4)
            self.assertNotIn('BLANK', lexicon.read_text())

    def test_token_order_and_unknown_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = root / 'tokens.json'
            wrong = list(worker.TOKENS)
            wrong[1], wrong[2] = wrong[2], wrong[1]
            tokens.write_text(json.dumps({'tokens': wrong, 'blank_id': 0, 'boundary_id': 40}))
            with self.assertRaisesRegex(ValueError, 'token'):
                worker.load_tokens(tokens)
            lexicon = root / 'lexicon.txt'
            lexicon.write_text('<unk> AA\n')
            with self.assertRaisesRegex(ValueError, 'word'):
                worker.load_lexicon(lexicon)

    def test_stable_natural_log_softmax_preserves_all_41_columns(self):
        values = peaked_logits([1, 0, 1, 40])
        actual = worker.log_softmax(values)
        shifted = worker.log_softmax(values + np.float32(10000))
        np.testing.assert_allclose(actual, shifted, atol=1e-6)
        np.testing.assert_allclose(np.exp(actual).sum(axis=1), np.ones(4), atol=1e-6)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(actual.flags.c_contiguous)
        with self.assertRaisesRegex(ValueError, '41'):
            worker.log_softmax(values[:, :40])
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            worker.log_softmax(values)

    def test_unigrams_require_nonnegative_training_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'counts.json'
            path.write_text(json.dumps({'counts': {'read': 2, "can't": 1}, 'source_split': 'train'}))
            counts = worker.load_unigrams(path)
            self.assertEqual(counts['read'], 2)
            path.write_text(json.dumps({'counts': {'read': 2}, 'source_split': 'val'}))
            with self.assertRaisesRegex(ValueError, 'train'):
                worker.load_unigrams(path)
            path.write_text(json.dumps({'read': -1}))
            with self.assertRaisesRegex(ValueError, 'count'):
                worker.load_unigrams(path)


class ProcessMonitorTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.dict(os.environ, {'SLURM_JOB_ID': '', 'SLURM_MEM_PER_NODE': ''}))

    def test_slurm_allocation_is_used_instead_of_laptop_defaults(self):
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': '1', 'SLURM_MEM_PER_NODE': '20480'}):
            monitor = worker.ProcessMonitor('unused-resources.json', max_seconds=60)
        self.assertEqual(monitor.hard_ram, 20 * 1024**3)
        self.assertEqual(monitor.soft_ram, int(20 * 1024**3 * 0.90))
        self.assertEqual(monitor.ram_source, 'Slurm node allocation')

    def test_supervisor_stop_request_is_cooperative_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'resources.json'
            stop = Path(directory) / 'STOP_REQUESTED'
            stop.write_text('Save at the next valid trial boundary.\n')
            with mock.patch.dict(os.environ, {'BTT_STOP_FILE': str(stop)}):
                with self.assertRaisesRegex(TimeoutError, 'supervisor requested a safe stop'):
                    with worker.ProcessMonitor(output, max_seconds=60) as monitor:
                        monitor.check()
            summary = json.loads(output.read_text())
            self.assertEqual(summary['outcome'], 'failed')
            self.assertIn('supervisor requested a safe stop', summary['stop_reason'])
            self.assertEqual(summary['host_target_bytes'], 12_000_000_000)
            self.assertEqual(summary['host_ceiling_bytes'], 16_000_000_000)
            self.assertEqual(summary['minimum_required_system_available_bytes'], 2_000_000_000)
            self.assertIn('between trials', summary['ceiling_method'])

    def test_ram_target_hard_ceiling_and_os_reserve_have_distinct_reasons(self):
        cases = ((13_000_000_000, 3_000_000_000, '12 GB worker RAM target'),
                 (17_000_000_000, 3_000_000_000, '16 GB worker RAM ceiling'),
                 (100_000_000, 1_000_000_000, 'OS available memory below 2 GB'))
        for rss, available, reason in cases:
            with self.subTest(reason=reason):
                monitor = worker.ProcessMonitor('unused-resources.json', max_seconds=60)
                monitor.samples = io.StringIO()
                monitor.process = mock.Mock()
                monitor.process.memory_info.return_value = SimpleNamespace(rss=rss, private=rss)
                with mock.patch.dict(os.environ, {'BTT_STOP_FILE': ''}):
                    with mock.patch.object(worker.psutil, 'virtual_memory', return_value=SimpleNamespace(available=available)):
                        with self.assertRaisesRegex(TimeoutError, reason):
                            monitor.check()


@unittest.skipUnless(native_available(), 'Run native contracts under .decoder-venv CPython 3.12')
class NativeDecoderTests(unittest.TestCase):
    def test_blank_separated_repeat_and_native_component_scores(self):
        decoder = worker.NativeDecoder({'double': [(1, 1)]}, beam=50, nbest=10)
        candidates = decoder.decode(peaked_logits([1, 0, 1, 40]))
        self.assertEqual(candidates[0]['text'], 'double')
        self.assertEqual(candidates[0]['phoneme_ids'], [1, 1, 40])
        self.assertEqual(candidates[0]['component_scores']['native_lm'], 0.0)
        self.assertTrue(np.isfinite(candidates[0]['component_scores']['acoustic']))
        self.assertNotIn('<unk>', candidates[0]['words'])

    def test_nbest_deduplicates_word_sequences_and_uses_optional_prior(self):
        decoder = worker.NativeDecoder(
            {'alpha': [(1,), (1, 1)], 'beta': [(1,)]},
            beam=50, nbest=10, unigrams={'alpha': 1, 'beta': 20}, unigram_weight=1.0,
        )
        candidates = decoder.decode(peaked_logits([1, 40]))
        texts = [row['text'] for row in candidates]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertLessEqual(len(texts), 10)
        self.assertEqual(texts[0], 'beta')

    def fixture(self, root):
        source = root / 'source'
        source.mkdir()
        (source / 'logits').mkdir()
        path = source / 'logits/000000.npy'
        np.save(path, peaked_logits([1, 0, 1, 40]), allow_pickle=False)
        records = [{'trial_id': 'fixture/val/trial_1', 'split': 'val', 'session_id': 'fixture',
                    'official_order_index': 0, 'input_length': 10, 'source_file': 'never-open.hdf5'}]
        manifest = root / 'data_manifest.json'
        manifest.write_text(json.dumps(records))
        cache = {'checkpoint_hash': 'checkpoint-fixture', 'contract_hash': 'contract-fixture',
                 'split_hash': 'split-fixture', 'dtype': 'float32', 'partition': 'val',
                 'manifest_hash': worker.object_hash(records), 'expected_trials': 1, 'references_loaded': False}
        (source / 'cache_manifest.json').write_text(json.dumps(cache))
        (source / 'run_manifest.json').write_text(json.dumps({'state': 'complete', 'expected_trials': 1}))
        row = {**records[0], 'logit_path': 'logits/000000.npy', 'logit_sha256': worker.sha256(path),
               'output_length': 4, 'phoneme_ids': [1, 1, 40], 'status': 'ok', 'excluded_field': 'fixture-only'}
        (source / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
        lexicon = root / 'lexicon.txt'
        lexicon.write_text('double AA AA\n')
        return source, lexicon, manifest

    def test_incremental_output_resume_cache_identity_and_no_model_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, lexicon, manifest = self.fixture(root)
            output = root / 'decoded'
            status = worker.run_decode(source, lexicon, output, manifest)
            self.assertEqual(status['state'], 'complete')
            self.assertFalse(status['references_loaded'])
            rows = worker.read_jsonl(output / 'predictions.jsonl')
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['text'], 'double')
            self.assertEqual(rows[0]['trial_id'], 'fixture/val/trial_1')
            self.assertEqual(rows[0]['phoneme_ids'], [1, 1, 40])
            self.assertEqual(rows[0]['phoneme_status'], 'ok')
            self.assertEqual(rows[0]['decoder_mode'], 'lexicon_only')
            cache = json.loads((output / 'cache_manifest.json').read_text())
            self.assertEqual(cache['decoder']['lm'], 'ZeroLM')
            self.assertNotIn('lm_weight', cache['decoder'])
            self.assertNotIn('kenlm_model_sha256', cache)
            self.assertNotIn('excluded_field', rows[0])
            self.assertEqual(len(worker.read_jsonl(output / 'candidates.jsonl')), 1)
            resources = json.loads((output / 'resources.json').read_text())
            self.assertGreater(resources['peak_rss_bytes'], 0)
            self.assertEqual(resources['device'], 'N/A')
            worker.run_decode(source, lexicon, output, manifest)
            self.assertEqual(len(worker.read_jsonl(output / 'predictions.jsonl')), 1)
            lexicon.write_text('different AA AA\n')
            with self.assertRaisesRegex(ValueError, 'cache'):
                worker.run_decode(source, lexicon, output, manifest)

    def test_kenlm_cache_hash_weight_resume_and_unloadable_status(self):
        with tempfile.TemporaryDirectory() as directory, fake_native():
            root = Path(directory)
            source, lexicon, manifest = self.fixture(root)
            model = root / 'train.binary'
            model.write_bytes(b'fixture model version one')
            output = root / 'kenlm'
            status = worker.run_decode(source, lexicon, output, manifest, kenlm_model=model, lm_weight=2.0)
            self.assertEqual(status['state'], 'complete')
            self.assertEqual(status['decoder_mode'], 'lexicon_kenlm')
            self.assertTrue(status['kenlm']['used'])
            cache = json.loads((output / 'cache_manifest.json').read_text())
            self.assertEqual(cache['kenlm_model_sha256'], worker.sha256(model))
            self.assertEqual(cache['decoder']['lm'], 'KenLM')
            self.assertEqual(cache['decoder']['lm_weight'], 2.0)
            self.assertAlmostEqual(cache['decoder']['native_lm_weight'], 2.0 * np.log(10))
            self.assertFalse(cache['references_loaded'])
            worker.run_decode(source, lexicon, output, manifest, kenlm_model=model, lm_weight=2.0)
            self.assertEqual(len(worker.read_jsonl(output / 'predictions.jsonl')), 1)
            with self.assertRaisesRegex(ValueError, 'cache mismatch'):
                worker.run_decode(source, lexicon, output, manifest, kenlm_model=model, lm_weight=1.0)
            model.write_bytes(b'fixture model version two')
            with self.assertRaisesRegex(ValueError, 'cache mismatch'):
                worker.run_decode(source, lexicon, output, manifest, kenlm_model=model, lm_weight=2.0)
            failed = root / 'failed'
            with fake_native(RuntimeError('bad model')):
                with self.assertRaisesRegex(RuntimeError, 'Unable to load requested KenLM'):
                    worker.run_decode(source, lexicon, failed, manifest, kenlm_model=model)
            failed_status = json.loads((failed / 'run_manifest.json').read_text())
            self.assertEqual(failed_status['state'], 'failed')
            self.assertIn('Unable to load requested KenLM', failed_status['error'])
            self.assertFalse((failed / 'predictions.jsonl').exists())

    def test_word_decoder_failure_preserves_successful_greedy_phonemes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, lexicon, manifest = self.fixture(root)
            output = root / 'decoded'
            with mock.patch.object(worker.NativeDecoder, 'decode', return_value=[]):
                status = worker.run_decode(source, lexicon, output, manifest)
            prediction = worker.read_jsonl(output / 'predictions.jsonl')[0]
            self.assertEqual(prediction['status'], 'decoder_error')
            self.assertEqual(prediction['phoneme_status'], 'ok')
            self.assertEqual(prediction['phoneme_ids'], [1, 1, 40])
            self.assertEqual(prediction['decoder_phoneme_ids'], [])
            self.assertEqual(prediction['text'], '')
            self.assertEqual(status['decoder_error_trials'], 1)

    def test_changed_logits_fail_before_decode_and_unknown_trial_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, lexicon, manifest = self.fixture(root)
            np.save(source / 'logits/000000.npy', peaked_logits([1, 40]), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, 'hash'):
                worker.run_decode(source, lexicon, root / 'decoded', manifest)
            row = worker.read_jsonl(source / 'predictions.jsonl')[0]
            row['trial_id'] = 'unknown'
            (source / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'trial'):
                worker.run_decode(source, lexicon, root / 'other', manifest)

    def test_complete_source_requires_full_partition_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, lexicon, manifest = self.fixture(root)
            records = json.loads(manifest.read_text())
            records.append({**records[0], 'trial_id': 'fixture/val/missing', 'official_order_index': 1})
            manifest.write_text(json.dumps(records))
            cache = json.loads((source / 'cache_manifest.json').read_text())
            cache['manifest_hash'] = worker.object_hash(records)
            (source / 'cache_manifest.json').write_text(json.dumps(cache))
            with self.assertRaisesRegex(ValueError, 'coverage'):
                worker.run_decode(source, lexicon, root / 'decoded', manifest)

    def test_resume_rejects_candidate_cache_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, lexicon, manifest = self.fixture(root)
            output = root / 'decoded'
            worker.run_decode(source, lexicon, output, manifest)
            packet = worker.read_jsonl(output / 'candidates.jsonl')[0]
            packet['decoder_cache_hash'] = 'wrong'
            (output / 'candidates.jsonl').write_text(json.dumps(packet) + '\n')
            with self.assertRaisesRegex(ValueError, 'cache'):
                worker.run_decode(source, lexicon, output, manifest)

    def test_cli_builds_a_validated_lexicon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cmu = root / 'cmu.txt'
            cmu.write_text("CAN'T K AE1 N T\n")
            with contextlib.redirect_stdout(io.StringIO()):
                worker.main(['--cmudict', str(cmu), '--lexicon', str(root / 'lexicon.txt'), '--build-lexicon-only'])
            self.assertIn("can't", worker.load_lexicon(root / 'lexicon.txt'))


@unittest.skipUnless(native_kenlm_available(), 'Requires the optional native Flashlight KenLM extension')
class RealKenLMTests(unittest.TestCase):
    def test_tiny_arpa_native_word_and_eos_scores_use_common_natural_units(self):
        with tempfile.TemporaryDirectory() as directory:
            arpa = Path(directory) / 'tiny.arpa'
            arpa.write_text(
                '\\data\\\nngram 1=5\nngram 2=6\n\n\\1-grams:\n'
                '-1.0\t<unk>\t0.0\n-0.3010\t<s>\t0.0\n-0.3010\t</s>\n'
                '-0.3010\talpha\t0.0\n-1.0\tbeta\t0.0\n\n\\2-grams:\n'
                '-0.1\t<s> alpha\n-1.0\t<s> beta\n-0.2\talpha </s>\n'
                '-0.3\tbeta </s>\n-0.5\talpha beta\n-0.6\tbeta alpha\n\n\\end\\\n',
                encoding='utf-8',
            )
            decoder = worker.NativeDecoder({'alpha': [(1,)], 'beta': [(1,)]}, kenlm_model=arpa, lm_weight=0.5)
            choices = decoder.decode(peaked_logits([1, 40]))
            self.assertEqual(choices[0]['text'], 'alpha')
            by_text = {choice['text']: choice for choice in choices}
            self.assertIn('beta', by_text)
            for word, expected in (('alpha', -0.3), ('beta', -1.3)):
                score = by_text[word]['component_scores']
                self.assertAlmostEqual(score['kenlm_log10'], expected, places=5)
                self.assertAlmostEqual(score['kenlm_logprob'], expected * np.log(10), places=5)
                self.assertAlmostEqual(score['kenlm_contribution'], 0.5 * expected * np.log(10), places=5)


if __name__ == '__main__':
    unittest.main()
