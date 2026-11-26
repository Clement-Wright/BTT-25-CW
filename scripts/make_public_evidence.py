r"""Export checked aggregate study evidence and static charts without private rows.

Example:
    python scripts/make_public_evidence.py --study-dir runs/locked-a100 \
        --output-dir reports --laptop-resource baseline_inference=runs/baseline-val/resources.json

Only complete report_studies.py outputs are accepted. Optional submissions are
counted locally; their text, trial IDs, input paths, and arbitrary metadata are
never copied into the public artifacts. No model or reference loader is used.
Matplotlib is required to render PNG and SVG figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile


SPLITS = ('dev_report', 'official_val')
COUNT_FIELDS = ('substitutions', 'deletions', 'insertions', 'errors', 'reference_length')
INTERVALS = ('bootstrap', 'session_cluster_bootstrap', 'session_block_cluster_bootstrap')
LAPTOP_LABELS = {
    'baseline_inference': 'Laptop inference', 'lexicon_decode': 'Laptop word search',
    'head_pilot': 'Laptop head pilot', 'session_bias_pilot': 'Laptop session pilot',
    'head_training': 'Laptop head training', 'session_bias_training': 'Laptop session training',
}
FIGURES = ('error_rates', 'paired_changes', 'rescoring_cost', 'resource_footprint')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def number(value, label, *, nullable=False, integer=False, negative=False):
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if integer and not isinstance(value, int):
        raise ValueError(f'{label} must be an integer')
    if not negative and value < 0:
        raise ValueError(f'{label} must be nonnegative')
    return value


def equal_number(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f'{label} differs from aggregate counts')
    elif isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f'{label} differs from aggregate counts')


def condition_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'(?:A[012]|A3_seed[0-9]+|B[12])', name):
        raise ValueError('Unexpected condition identifier')
    return name


def condition_label(name):
    return {'A0': 'A0 · Compact', 'A1': 'A1 · Every trial', 'A2': 'A2 · Confidence gate',
            'B1': 'B1 · Head', 'B2': 'B2 · Session + head'}.get(name, f'A3 · Random seed {name.removeprefix("A3_seed")}')


def edit_counts(value):
    result = {key: number(value[key], key, integer=True) for key in COUNT_FIELDS}
    if result['errors'] != sum(result[key] for key in ('substitutions', 'deletions', 'insertions')):
        raise ValueError('Edit components do not sum to errors')
    return result


def oov_counts(value):
    # Deliberately omit oov_counts: its keys are individual reference words.
    counts = ('reference_token_count', 'reference_type_count', 'in_lexicon_token_count',
              'in_lexicon_type_count', 'oov_token_count', 'oov_type_count', 'sentence_count',
              'sentences_with_oov', 'empty_reference_count')
    rates = ('token_coverage', 'type_coverage', 'oov_token_rate', 'oov_type_rate', 'sentence_oov_rate')
    return {**{key: number(value[key], key, integer=True) for key in counts if key in value},
            **{key: number(value[key], key, nullable=True) for key in rates if key in value}}


def aggregate(metrics, overall=None):
    result = {key: number(metrics[key], key, integer=True) for key in
              ('evaluated_count', 'failed_count', 'missing_count', 'successful_count', 'skipped_count') if key in metrics}
    if 'evaluated_count' not in result:
        result['evaluated_count'] = number(metrics['trial_count'], 'trial_count', integer=True)
    for kind, rate in (('word', 'wer'), ('phoneme', 'per')):
        counts = edit_counts(metrics[kind])
        expected = counts['errors'] / counts['reference_length'] if counts['reference_length'] else None
        equal_number(metrics[rate], expected, rate)
        result[kind], result[rate] = counts, expected
    if overall is not None:
        if overall['trial_count'] != result['evaluated_count']:
            raise ValueError('Diagnostics trial count differs from metrics')
        for kind in ('word', 'phoneme'):
            if edit_counts(overall[kind]) != result[kind]:
                raise ValueError('Diagnostics edit counts differ from metrics')
        for key in ('lexicon_oov', 'lm_oov'):
            if key in overall:
                result[key] = oov_counts(overall[key])
    else:
        for key in ('lexicon_oov', 'lm_oov'):
            if key in metrics:
                result[key] = oov_counts(metrics[key])
    return result


def resource_counts(value):
    keys = ('elapsed_seconds', 'sample_count', 'sampling_seconds', 'peak_tree_rss_bytes',
            'peak_tree_private_bytes', 'peak_rss_bytes', 'peak_private_bytes',
            'minimum_system_available_bytes', 'peak_device_used_bytes',
            'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes',
            'host_ceiling_bytes', 'host_target_bytes', 'minimum_required_system_available_bytes')
    result = {key: number(value[key], key) for key in keys if value.get(key) is not None}
    result['memory_scope'] = 'sequential_stage_peaks' if 'stages' in value else ('process_tree' if 'peak_tree_rss_bytes' in value else 'worker_process')
    if 'outcome' in value:
        if value['outcome'] not in {'complete', 'failed', 'budget_stopped'}:
            raise ValueError('Unexpected resource outcome')
        result['outcome'] = value['outcome']
    if 'stages' in value:
        result['stages'] = {}
        for stage in ('train', 'infer', 'decode', 'evaluate'):
            for index, filename in enumerate(('resources.json', 'resources-resume.json'), 1):
                key = f'{stage}/{filename}'
                if key in value['stages']:
                    result['stages'][f'{stage}_attempt_{index}'] = resource_counts(value['stages'][key])
    return result


def paired_metric(value, a, b, rate):
    if value.get('measured') is not True:
        return {'measured': False}
    kind = 'word' if rate == 'wer' else 'phoneme'
    denominator = a[kind]['reference_length']
    if denominator != b[kind]['reference_length']:
        raise ValueError('Paired denominator differs across conditions')
    for key, expected in (('errors_a', a[kind]['errors']), ('errors_b', b[kind]['errors']),
                          ('reference_length', denominator), ('rate_a', a[rate]), ('rate_b', b[rate])):
        equal_number(value[key], expected, f'paired {key}')
    delta = b[rate] - a[rate] if denominator else None
    equal_number(value['delta_b_minus_a'], delta, 'paired delta')
    result = {'measured': True, 'errors_a': a[kind]['errors'], 'errors_b': b[kind]['errors'],
              'reference_length': denominator, 'rate_a': a[rate], 'rate_b': b[rate], 'delta_b_minus_a': delta}
    for name in INTERVALS:
        interval = value[name]
        bounds = {key: number(interval[key], key, nullable=True, negative=True) for key in ('lower', 'upper')}
        if (bounds['lower'] is None) != (bounds['upper'] is None) or (bounds['lower'] is not None and bounds['lower'] > bounds['upper']):
            raise ValueError('Invalid paired interval bounds')
        result[name] = dict(bounds, **{key: number(interval[key], key, integer=True) for key in ('units', 'valid_replicates')})
    for key in ('rescue_count', 'harm_count', 'unchanged_count'):
        result[key] = number(value[key], key, integer=True)
    return result


def submission_counts(path, expected):
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        raise ValueError('Submissions require a positive expected_test_count')
    count, empty = 0, 0
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['id', 'text']:
            raise ValueError('Submission must have exactly id,text columns')
        for index, row in enumerate(reader):
            if row.get('id') != str(index) or set(row) != {'id', 'text'} or not isinstance(row['text'], str):
                raise ValueError('Submission IDs must be sequential and each row must contain text')
            count += 1
            empty += not row['text'].strip()
    if count != expected:
        raise ValueError('Submission count differs from expected_test_count')
    return {'row_count': count, 'expected_count': expected, 'empty_text_count': empty,
            'sequential_ids_verified': True, 'sha256': file_hash(path), 'test_accuracy': 'unavailable'}


def prediction_examples(path, declared_path, freeze_hash, split_hash):
    value, declared = read_json(path), read_json(declared_path)
    ids = value['ids']
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 12 or any(not isinstance(trial, str) for trial in ids)
            or len(set(ids)) != len(ids) or value['ids_hash'] != object_hash(ids)
            or declared.get('ids') != ids or declared.get('ids_hash') != object_hash(ids)
            or declared.get('split_hash') != split_hash):
        raise ValueError('Model examples must match the small predeclared ID selection')
    if (value['score_split'] != 'dev_report' or value['split_manifest_hash'] != split_hash
            or value['freeze_manifest_hash'] != freeze_hash):
        raise ValueError('Model examples split/freeze mismatch')
    conditions = ['A0', 'A1', 'A2', 'B1', 'B2']
    if value['condition_order'] != conditions or [row['trial_id'] for row in value['rows']] != ids:
        raise ValueError('Model example condition order or predeclared IDs changed')
    examples = []
    for index, row in enumerate(value['rows'], 1):
        hypotheses = {}
        for name in conditions:
            prediction = row['predictions'][name]
            text, status = prediction['text'], prediction['status']
            if not isinstance(text, str) or not re.fullmatch(r"[a-z' -]*", text) or text != text.strip():
                raise ValueError('Expected normalized model word output without paths or control characters')
            if status not in {'ok', 'missing', 'error', 'decoder_error', 'lm_error', 'failed'}:
                raise ValueError('Unexpected model prediction status')
            hypotheses[name] = {'text': text, 'status': status}
        examples.append({'example_number': index, 'example_id_sha256': hashlib.sha256(row['trial_id'].encode()).hexdigest(),
                         'predictions': hypotheses})
    return {'state': 'included', 'score_split': 'dev_report', 'ids_hash': value['ids_hash'],
            'source_sha256': file_hash(path), 'declared_ids_source_sha256': file_hash(declared_path),
            'example_count': len(examples), 'condition_count': len(conditions), 'examples': examples}


def build_evidence(study_dir, laptop_resources=None, submissions=None, expected_test_count=None,
                   laptop_baseline=None, encoder_parity=None, model_examples=None):
    study = Path(study_dir)
    complete, frozen = read_json(study / 'complete.json'), read_json(study / 'joint_freeze.json')
    if complete.get('state') != 'complete':
        raise ValueError('Study evidence must be complete before public export')
    freeze_hash = frozen.get('manifest_hash')
    if (not isinstance(freeze_hash, str) or not re.fullmatch('[0-9a-f]{64}', freeze_hash)
            or object_hash({key: value for key, value in frozen.items() if key != 'manifest_hash'}) != freeze_hash):
        raise ValueError('Joint freeze content hash mismatch')
    if frozen.get('reporting_references_opened') is not False:
        raise ValueError('Joint freeze must precede reporting reference access')
    split_hash = frozen.get('split_hash')
    if not isinstance(split_hash, str) or not re.fullmatch('[0-9a-f]{64}', split_hash):
        raise ValueError('Missing frozen split hash')
    confidence = complete['confidence']
    if (complete['freeze_hash'] != freeze_hash or confidence['freeze_manifest_hash'] != freeze_hash
            or confidence['split_hash'] != split_hash or confidence['score_split'] != 'dev_report'):
        raise ValueError('Complete report and confidence summary disagree with the joint freeze')
    results, study_resources = {}, {}
    names = sorted(complete['results']['dev_report'])
    if not {'A0', 'A1', 'A2', 'B1', 'B2'}.issubset(names):
        raise ValueError('Complete evidence must retain all study conditions')
    for name in names:
        condition_name(name)
    corpus_names = sorted({corpus for split in SPLITS for record in complete['results'][split].values() for corpus in record['random_corpora']})
    corpus_alias = {}
    for category in ('random', 'unusual'):
        members = [name for name in corpus_names if ('random' in name.casefold() if category == 'random' else 'unusual' in name.casefold() and 'random' not in name.casefold())]
        corpus_alias.update({name: f'{category}_{index}' for index, name in enumerate(members, 1)})
    if set(corpus_alias) != set(corpus_names):
        raise ValueError('Unexpected unusual/random corpus category')
    for split in SPLITS:
        if sorted(complete['results'][split]) != names:
            raise ValueError('Scoring selections must retain identical conditions')
        results[split] = {}
        for name in names:
            record = complete['results'][split][name]
            metrics = record['metrics']
            if (metrics.get('score_split') != split or metrics.get('split_manifest_hash') != split_hash
                    or metrics.get('freeze_manifest_hash') != freeze_hash):
                raise ValueError('Metric split/freeze identity mismatch')
            result = aggregate(metrics, record['overall'])
            result['unusual_random_corpora'] = {corpus_alias[key]: aggregate(value) for key, value in sorted(record['random_corpora'].items())}
            results[split][name] = result
            resources = resource_counts(record['performance'])
            if name in study_resources and resources != study_resources[name]:
                raise ValueError('Resource counters changed between scoring selections')
            study_resources[name] = resources
        expected_count = results[split]['A0']['evaluated_count']
        if any(row['evaluated_count'] != expected_count for row in results[split].values()):
            raise ValueError('Condition trial coverage differs')
    with (study / 'experiments.csv').open(newline='', encoding='utf-8') as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_keys = [(row['score_split'], row['condition']) for row in csv_rows]
    if len(csv_keys) != len(set(csv_keys)) or set(csv_keys) != {(split, name) for split in SPLITS for name in names} or complete['rows'] != len(csv_rows):
        raise ValueError('CSV condition coverage differs from complete.json')
    for row in csv_rows:
        result = results[row['score_split']][row['condition']]
        if row['status'] != 'complete' or row['freeze_hash'] != freeze_hash:
            raise ValueError('CSV state/freeze mismatch')
        expected = {key: result[key] for key in ('evaluated_count', 'per', 'wer')}
        expected.update(word_edits=result['word']['errors'], word_denominator=result['word']['reference_length'],
                        phoneme_edits=result['phoneme']['errors'], phoneme_denominator=result['phoneme']['reference_length'])
        policy = confidence['conditions'].get(row['condition'], {})
        expected.update(additional_lm_calls_on_full_validation=policy.get('lm_calls'),
                        p95_incremental_lm_seconds=policy.get('called_latency_p95_seconds'))
        for key, value in expected.items():
            if key not in row:
                raise ValueError(f'CSV is missing the registered {key} column')
            equal_number(float(row[key]) if row[key] else None, value, f'CSV {key}')
    comparisons = {}
    for key, value in sorted(complete['results']['comparisons'].items()):
        parts = key.split('-minus-')
        if len(parts) != 2:
            raise ValueError('Unexpected comparison identifier')
        b, a = map(condition_name, parts)
        if a not in names or b not in names or value['score_split'] != 'dev_report' or value['freeze_manifest_hash'] != freeze_hash or value['split_manifest_hash'] != split_hash:
            raise ValueError('Paired comparison split/freeze mismatch')
        pair = {key: number(value[key], key, integer=True) for key in ('trial_count', 'seed', 'replicates', 'missing_session_count', 'missing_block_count')}
        if pair['trial_count'] != results['dev_report'][a]['evaluated_count']:
            raise ValueError('Paired trial coverage differs')
        pair['confidence_level'] = number(value['confidence_level'], 'confidence_level')
        if pair['confidence_level'] != .95:
            raise ValueError('Expected the registered 95% paired intervals')
        pair.update(condition_a=a, condition_b=b)
        for rate in ('wer', 'per'):
            pair[rate] = paired_metric(value[rate], results['dev_report'][a], results['dev_report'][b], rate)
        comparisons[key] = pair
    required_pairs = {'A1-minus-A0', 'A2-minus-A0', 'A2-minus-A1', 'B1-minus-A0', 'B2-minus-A0', 'B2-minus-B1'}
    required_pairs.update(f'A2-minus-{name}' for name in names if name.startswith('A3_seed'))
    if set(comparisons) != required_pairs:
        raise ValueError('Paired comparisons do not retain all registered conditions')
    policies = confidence['conditions']
    if set(policies) != {name for name in names if name.startswith('A')}:
        raise ValueError('Confidence policy coverage differs from scored conditions')
    every_calls = number(policies['A1']['lm_calls'], 'lm_calls', integer=True)
    calls = {}
    for name, policy in sorted(policies.items()):
        counts = {key: number(policy[key], key, integer=True) for key in ('lm_calls', 'lm_candidates', 'lm_failures', 'scored_trial_lm_calls')}
        if counts['lm_calls'] > results['official_val'][name]['evaluated_count'] or counts['scored_trial_lm_calls'] > results['dev_report'][name]['evaluated_count']:
            raise ValueError('LM calls exceed their declared selection')
        if counts['lm_calls'] > every_calls or counts['lm_failures'] > counts['lm_calls'] or counts['scored_trial_lm_calls'] > counts['lm_calls']:
            raise ValueError('LM counters contradict the fixed policy coverage')
        equal_number(policy['call_fraction'], counts['lm_calls'] / results['official_val'][name]['evaluated_count'], 'call fraction')
        times = {key: number(policy[key], key, nullable=True) for key in
                 ('lm_total_seconds', 'called_latency_p50_seconds', 'called_latency_p95_seconds',
                  'trial_latency_p50_seconds', 'trial_latency_p95_seconds') if key in policy}
        saved = every_calls - counts['lm_calls']
        calls[name] = dict(counts, **times, lm_calls_saved_vs_every_trial=saved,
                           lm_call_savings_fraction=saved / every_calls if every_calls else None)
    laptop, laptop_summary = {}, None
    if laptop_baseline is not None:
        baseline = read_json(laptop_baseline)
        metrics = baseline['evaluation']
        if metrics['score_split'] != 'dev_tune' or metrics['decoder'] != 'lexicon_only' or metrics['split_manifest_hash'] != split_hash:
            raise ValueError('Laptop fallback must retain its lexicon-only dev_tune identity')
        laptop_summary = dict(aggregate(metrics), score_split='dev_tune', decoder='lexicon_only', source_sha256=file_hash(laptop_baseline))
        for source_name, name in (('phoneme_inference', 'baseline_inference'), ('lexicon_only_word_search', 'lexicon_decode'),
                                  ('session_calibration_pilot', 'session_bias_pilot')):
            if source_name in baseline['runs']:
                value = baseline['runs'][source_name]
                laptop[name] = resource_counts(value)
                for key in ('trials_saved', 'steps', 'sustained_seconds', 'step_p95_seconds', 'validation_8_median_seconds', 'checkpoint_median_seconds'):
                    if value.get(key) is not None:
                        laptop[name][key] = number(value[key], key)
        laptop_summary['first_time_setup'] = {key: number(value, key) for key, value in baseline.get('first_time_setup', {}).items()
                                              if key in {'checkpoint_zip_bytes', 'checkpoint_zip_seconds', 'scorer_cpu_fixture_load_seconds', 'scorer_cpu_fixture_peak_reported_rss_bytes'}}
    for name, path in sorted((laptop_resources or {}).items()):
        if name not in LAPTOP_LABELS:
            raise ValueError('Unknown laptop resource label')
        laptop[name] = dict(resource_counts(read_json(path)), source_sha256=file_hash(path))
    tests = {'state': 'not_supplied', 'conditions': {}}
    if submissions:
        if not set(submissions).issubset(names):
            raise ValueError('Submission condition is absent from the study evidence')
        tests = {'state': 'coverage_verified', 'conditions': {condition_name(name): submission_counts(path, expected_test_count)
                                                          for name, path in sorted(submissions.items())}}
    selected = complete['compact_selection']
    if selected['mode'] not in {'kenlm', 'kenlm_3gram', 'lexicon_only'}:
        raise ValueError('Unexpected compact decoder mode')
    result = {'schema_version': 1, 'state': 'complete',
            'provenance': {'freeze_manifest_hash': freeze_hash, 'split_manifest_hash': split_hash,
                           'source_sha256': {name.replace('.', '_'): file_hash(study / name) for name in ('complete.json', 'experiments.csv', 'joint_freeze.json')},
                           'exporter_sha256': file_hash(__file__)},
            'scope': {'participant_count': 1, 'interval_scope': 'resampling of this participant and dataset',
                      'official_val': 'includes dev_tune and dev_report; original checkpoint selected using validation'},
            'compact_decoder': {'mode': selected['mode'], 'lm_weight': number(selected['selected']['weight'], 'compact lm_weight')},
            'confidence': {'call_scope': 'full_validation', 'selected_lambda': number(confidence['selected_lambda'], 'selected_lambda'),
                           'gate_threshold': number(confidence['gate_threshold'], 'gate_threshold', nullable=True),
                           'scorer_setup_seconds': number(confidence['scorer_setup_seconds'], 'scorer_setup_seconds'),
                           'shared_compact_decode': {key: number(confidence['shared_compact_decode'][key], key, nullable=True) for key in
                                                     ('recorded_source_run_elapsed_seconds', 'sum_trial_decode_seconds')},
                           'conditions': calls},
            'results': results, 'comparisons': comparisons, 'resources': {'study': study_resources, 'laptop': laptop},
            'test_submissions': tests,
            'model_predictions': prediction_examples(model_examples, study / 'example_ids.json', freeze_hash, split_hash)
            if model_examples is not None else {'state': 'not_supplied', 'example_count': 0, 'condition_count': 0, 'examples': []}}
    if laptop_summary is not None:
        result['laptop_baseline'] = laptop_summary
    if encoder_parity is not None:
        parity = read_json(encoder_parity)
        if parity.get('references_loaded') is not False or not isinstance(parity['trial_order_and_frame_lengths_match'], bool):
            raise ValueError('Encoder parity must use no references and declare frame/order agreement')
        count = number(parity['trials'], 'parity trials', integer=True)
        agreed = number(parity['exact_greedy_phoneme_agreement'], 'parity agreement', integer=True)
        if agreed > count or count != results['official_val']['A0']['evaluated_count']:
            raise ValueError('Encoder parity coverage differs from full validation')
        result['encoder_parity'] = {'trials': count, 'exact_greedy_phoneme_agreement': agreed,
                                    'trial_order_and_frame_lengths_match': parity['trial_order_and_frame_lengths_match'],
                                    'references_loaded': False, 'resources': resource_counts(parity),
                                    'source_sha256': file_hash(encoder_parity),
                                    'scope': 'decoded phone IDs and output lengths across laptop and A100; no bitwise logit claim'}
    return result


def percent(value):
    return '—' if value is None else f'{value * 100:.2f}%'


def decimal(value):
    return '—' if value is None else f'{value:.2f}'


def markdown_report(result):
    held_back = result['results']['dev_report']
    lines = ['# Measured results', '',
             f"WER on the held-back `dev_report` selection was {percent(held_back['A0']['wer'])} for the compact baseline, "
             f"{percent(held_back['A1']['wer'])} when rescoring every trial, and {percent(held_back['A2']['wer'])} with the confidence gate. "
             'The tables retain every registered condition.', '',
             f"Joint freeze: `{result['provenance']['freeze_manifest_hash']}`.", '',
             'Rates divide summed edit counts by summed reference lengths; failed and missing outputs remain in the denominator. '
             'The `official_val` aggregate includes tuning trials. The original checkpoint was selected using validation, '
             'and these results cover one participant.', '',
             f"Cluster compact decoder: `{result['compact_decoder']['mode']}`; natural-log LM weight {decimal(result['compact_decoder']['lm_weight'])}.", '',
             '![Word error rates](figures/error_rates.png)', '',
             '| Selection | Condition | Trials | Word edits / words | WER | Phone edits / phones | PER | Failed |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for split, conditions in result['results'].items():
        for name, row in conditions.items():
            lines.append(f"| {split} | {condition_label(name)} | {row['evaluated_count']} | {row['word']['errors']} / {row['word']['reference_length']} | {percent(row['wer'])} | {row['phoneme']['errors']} / {row['phoneme']['reference_length']} | {percent(row['per'])} | {row.get('failed_count', 0)} |")
    if 'laptop_baseline' in result:
        row = result['laptop_baseline']
        lines += ['', '## Laptop lexicon-only fallback', '',
                  f"On {row['evaluated_count']} `dev_tune` trials, the Windows laptop fallback made "
                  f"{row['word']['errors']} / {row['word']['reference_length']} word edits ({percent(row['wer'])} WER) and "
                  f"{row['phoneme']['errors']} / {row['phoneme']['reference_length']} phone edits ({percent(row['per'])} PER). "
                  'It used no KenLM. This evaluation uses a different decoder and selection from the cluster study.']
    if 'encoder_parity' in result:
        parity = result['encoder_parity']
        lines += ['', f"Greedy phone sequences agree between the laptop and A100 on {parity['exact_greedy_phoneme_agreement']} / {parity['trials']} validation trials. "
                  f"Trial order and frame lengths match: {str(parity['trial_order_and_frame_lengths_match']).lower()}. "
                  'The check uses no references; agreement in decoded phone IDs and lengths does not establish bitwise logit equality.']
    lines += ['', '## Paired changes', '', 'On `dev_report`, negative B-minus-A changes mean fewer errors. '
              'Changes are in percentage points, with session cluster 95% intervals in the table. '
              'The figure shows the same session intervals; JSON also retains paired trial and session/block intervals. '
              'These intervals describe resampling within this participant\'s dataset.', '', '![Paired changes](figures/paired_changes.png)', '',
              '| B minus A | WER change | Session 95% interval | PER change | Fewer / more / unchanged word edits |',
              '|---|---:|---:|---:|---:|']
    for name, pair in result['comparisons'].items():
        wer, per = pair['wer'], pair['per']
        interval = wer.get('session_cluster_bootstrap', {})
        bounds = '—' if interval.get('lower') is None else f"[{interval['lower'] * 100:+.2f}, {interval['upper'] * 100:+.2f}]"
        delta = '—' if wer.get('delta_b_minus_a') is None else f"{wer['delta_b_minus_a'] * 100:+.2f}"
        phone = '—' if per.get('delta_b_minus_a') is None else f"{per['delta_b_minus_a'] * 100:+.2f}"
        lines.append(f"| {name} | {delta} | {bounds} | {phone} | {wer.get('rescue_count', '—')} / {wer.get('harm_count', '—')} / {wer.get('unchanged_count', '—')} |")
    lines += ['', '## Additional LM calls', '', 'Call and candidate counts cover full validation, including tuning trials. '
              'The saved fraction compares each policy with rescoring every trial. Latencies time live additional LM calls; '
              'shared compact decoding and scorer setup are measured separately. '
              'The figure plots `dev_report` WER against `official_val` LM calls; call latency appears in the table.', '',
              '![Rescoring cost](figures/rescoring_cost.png)', '',
              '| Condition | Calls | Candidates | Failures | Calls saved | Saved fraction | Called p95 seconds |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in result['confidence']['conditions'].items():
        lines.append(f"| {condition_label(name)} | {row['lm_calls']} | {row['lm_candidates']} | {row['lm_failures']} | {row['lm_calls_saved_vs_every_trial']} | {percent(row['lm_call_savings_fraction'])} | {decimal(row.get('called_latency_p95_seconds'))} |")
    lines += ['', f"Scorer setup: {decimal(result['confidence']['scorer_setup_seconds'])} seconds. "
              f"Sum of cached compact trial decoding times: {decimal(result['confidence']['shared_compact_decode']['sum_trial_decode_seconds'])} seconds.", '',
              '## Unusual and random corpora', '', 'The public table reports corpus aggregates under group aliases. '
              'Source labels, individual OOV words, and examples are excluded.', '',
              '| Selection | Condition | Group | Trials | Word edits / words | WER |', '|---|---|---|---:|---:|---:|']
    for split, conditions in result['results'].items():
        for name, row in conditions.items():
            for corpus, group in row['unusual_random_corpora'].items():
                lines.append(f"| {split} | {name} | {corpus} | {group['evaluated_count']} | {group['word']['errors']} / {group['word']['reference_length']} | {percent(group['wer'])} |")
    lines += ['', '## Reference vocabulary coverage', '', '| Selection | Vocabulary | OOV tokens | OOV types | OOV token rate |', '|---|---|---:|---:|---:|']
    for split, conditions in result['results'].items():
        for key, label in (('lexicon_oov', 'Pronunciation lexicon'), ('lm_oov', 'Training LM')):
            value = conditions['A0'].get(key)
            if value:
                lines.append(f"| {split} | {label} | {value.get('oov_token_count', '—')} | {value.get('oov_type_count', '—')} | {percent(value.get('oov_token_rate'))} |")
    lines += ['', '## Measured resources', '', 'Calibration rows include sequential training, inference, decoding and evaluation; '
              'JSON retains the measurements for each stage. Memory is reported in decimal GB. Process/tree RSS, '
              'total device usage, and framework allocations measure different quantities. '
              'Failed or budget-stopped outcomes remain in the table when supplied.', '', '![Resource footprint](figures/resource_footprint.png)', '',
              '| Context | Condition / stage | Seconds | Host RSS GB | Device used GB | CUDA allocated GB | Outcome |', '|---|---|---:|---:|---:|---:|---|']
    for context, resources in result['resources'].items():
        for name, row in resources.items():
            host = row.get('peak_tree_rss_bytes', row.get('peak_rss_bytes'))
            gb = lambda value: decimal(value / 1e9) if value is not None else '—'
            label = LAPTOP_LABELS[name] if context == 'laptop' else condition_label(name)
            lines.append(f"| {context} | {label} | {decimal(row.get('elapsed_seconds'))} | {gb(host)} | {gb(row.get('peak_device_used_bytes'))} | {gb(row.get('peak_cuda_allocated_bytes'))} | {row.get('outcome', 'stage summaries')} |")
    lines += ['', '## Optional test exports', '']
    if result['test_submissions']['state'] == 'not_supplied':
        lines.append('This export has no test submission evidence. Test accuracy is unavailable.')
    else:
        lines += ['These export checks verify file coverage and ordering. Test labels are unavailable, so they provide no test accuracy result.', '',
                  '| Condition | Rows | Expected | Empty text | Sequential IDs |', '|---|---:|---:|---:|---|']
        for name, row in result['test_submissions']['conditions'].items():
            lines.append(f"| {name} | {row['row_count']} | {row['expected_count']} | {row['empty_text_count']} | verified |")
    if result['model_predictions']['state'] == 'included':
        lines += ['', f"[Saved model predictions](model_predictions.json) contain {result['model_predictions']['example_count']} predeclared examples "
                  'for the five primary conditions, with model text and status. References and phone labels are excluded.']
    lines += ['', '[results.json](results.json) contains the aggregate counts, every interval, resource counters, and input hashes. '
              'SVG versions of the charts are in `figures`.', '']
    return '\n'.join(lines)


def render_figures(result, output):
    if __name__ == '__main__':
        from render_public_figures import render_v1
    else:
        from scripts.render_public_figures import render_v1
    render_v1(result, output)


def make_public_evidence(study_dir, output_dir, **kwargs):
    study, output = Path(study_dir).resolve(), Path(output_dir).resolve()
    if output == study or output.is_relative_to(study) or study.is_relative_to(output):
        raise ValueError('Public output must be separate from the private study mirror')
    result = build_evidence(study, **kwargs)
    # Validation and all rendering finish before any existing public artifact is replaced.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='btt-public-', dir=output.parent) as temporary:
        staged = Path(temporary)
        render_figures(result, staged / 'figures')
        public = dict(result, model_predictions={key: value for key, value in result['model_predictions'].items() if key != 'examples'})
        (staged / 'results.json').write_text(json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
        (staged / 'model_predictions.json').write_text(json.dumps(result['model_predictions'], indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
        (staged / 'results.md').write_text(markdown_report(result), encoding='utf-8')
        (output / 'figures').mkdir(parents=True, exist_ok=True)
        for name in FIGURES:
            for extension in ('png', 'svg'):
                (staged / 'figures' / f'{name}.{extension}').replace(output / 'figures' / f'{name}.{extension}')
        for name in ('results.json', 'results.md', 'model_predictions.json'):
            (staged / name).replace(output / name)
    return result


def assignments(values, allowed):
    result = {}
    for value in values:
        name, separator, path = value.partition('=')
        if not separator or not path or name in result or not allowed(name):
            raise ValueError('Expected unique LABEL=PATH inputs with supported labels')
        result[name] = Path(path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--study-dir', type=Path, required=True, help='Local mirror containing complete.json, experiments.csv, joint_freeze.json')
    parser.add_argument('--output-dir', type=Path, default=Path('reports'))
    parser.add_argument('--laptop-resource', action='append', default=[], metavar='LABEL=JSON',
                        help='Repeat for ' + ', '.join(LAPTOP_LABELS))
    parser.add_argument('--submission', action='append', default=[], metavar='CONDITION=CSV', help='Optional test export; only counts and SHA256 are published')
    parser.add_argument('--expected-test-count', type=int, help='Required when submissions are supplied; verify against the captured test manifest')
    parser.add_argument('--laptop-baseline', type=Path, help='Optional sanitized laptop_baseline.json; fallback remains a separate dev_tune evaluation')
    parser.add_argument('--encoder-parity', type=Path, help='Optional sanitized a100_encoder_parity.json')
    parser.add_argument('--model-examples', type=Path, help='Optional fixed_examples.json; matched to study-dir/example_ids.json and stripped to model text/status')
    args = parser.parse_args(argv)
    try:
        laptop = assignments(args.laptop_resource, lambda name: name in LAPTOP_LABELS)
        submissions = assignments(args.submission, lambda name: bool(re.fullmatch(r'(?:A[012]|A3_seed[0-9]+|B[12])', name)))
        result = make_public_evidence(args.study_dir, args.output_dir, laptop_resources=laptop,
                                      submissions=submissions, expected_test_count=args.expected_test_count,
                                      laptop_baseline=args.laptop_baseline, encoder_parity=args.encoder_parity, model_examples=args.model_examples)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({'state': result['state'], 'conditions': len(result['results']['dev_report']),
                      'figures': len(FIGURES), 'test_submission_state': result['test_submissions']['state']}))


if __name__ == '__main__':
    main()
