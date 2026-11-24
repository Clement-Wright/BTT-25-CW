"""Public exports use synthetic aggregate evidence, never participant records."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/make_public_evidence.py'
PRIVATE = 'PRIVATE_REFERENCE_AND_OOV_SENTINEL'


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fixture(root):
    study = root / 'private_mirror'
    study.mkdir()
    freeze = {'split_hash': 'a' * 64, 'reporting_references_opened': False,
              'selection_rules': PRIVATE, 'created_at': '2026-11-23T00:00:00Z'}
    freeze['manifest_hash'] = object_hash(freeze)
    names = ['A0', 'A1', 'A2', 'A3_seed7', 'B1', 'B2']
    errors = dict(zip(names, (8, 7, 9, 8, 8, 6)))
    resources = {'elapsed_seconds': 4.0, 'peak_tree_rss_bytes': 2_000_000_000,
                 'peak_device_used_bytes': 1_000_000_000, 'peak_cuda_allocated_bytes': 500_000_000,
                 'minimum_system_available_bytes': 5_000_000_000, 'outcome': 'complete',
                 'stop_reason': PRIVATE, 'hardware': {'os': PRIVATE}, 'command': PRIVATE}
    results, rows = {}, []
    for split, trials in (('dev_report', 2), ('official_val', 4)):
        results[split] = {}
        for name in names:
            word = {'errors': errors[name], 'reference_length': 20,
                    'substitutions': errors[name], 'deletions': 0, 'insertions': 0}
            phone = {'errors': 4, 'reference_length': 40, 'substitutions': 4, 'deletions': 0, 'insertions': 0}
            metrics = {'score_split': split, 'split_manifest_hash': freeze['split_hash'],
                       'freeze_manifest_hash': freeze['manifest_hash'], 'evaluated_count': trials,
                       'failed_count': 0, 'missing_count': 0, 'successful_count': trials, 'skipped_count': 0,
                       'word': word, 'phoneme': phone, 'wer': errors[name] / 20, 'per': .1,
                       'reference_text': PRIVATE, 'metadata': {'private_path': PRIVATE}}
            oov = {'oov_token_count': 1, 'oov_type_count': 1, 'reference_token_count': 20,
                   'reference_type_count': 10, 'oov_token_rate': .05, 'oov_type_rate': .1,
                   'oov_counts': {PRIVATE: 1}}
            overall = {'trial_count': trials, 'failed_count': 0, 'missing_count': 0, 'phoneme_failed_count': 0,
                       'word': word, 'phoneme': phone, 'wer': metrics['wer'], 'per': .1,
                       'lexicon_oov': oov, 'lm_oov': oov, 'reference_text': PRIVATE}
            results[split][name] = {'metrics': metrics, 'overall': overall,
                                    'random_corpora': {'Random': overall}, 'performance': resources}
            rows.append({'score_split': split, 'condition': name, 'evaluated_count': trials,
                         'freeze_hash': freeze['manifest_hash'], 'per': .1, 'wer': errors[name] / 20,
                         'word_edits': errors[name], 'word_denominator': 20,
                         'phoneme_edits': 4, 'phoneme_denominator': 40, 'status': 'complete',
                         'additional_lm_calls_on_full_validation': {'A0': 0, 'A1': 4, 'A2': 2, 'A3_seed7': 2}.get(name, ''),
                         'p95_incremental_lm_seconds': .12 if name in {'A1', 'A2', 'A3_seed7'} else '',
                         'predictions_sha256': PRIVATE, 'checkpoint_hash': PRIVATE})
    results['comparisons'] = {}
    pairs = [('A0', 'A1'), ('A0', 'A2'), ('A1', 'A2'), ('A0', 'B1'), ('A0', 'B2'), ('B1', 'B2'), ('A3_seed7', 'A2')]
    for a, b in pairs:
        def metric(kind):
            denominator = 20 if kind == 'wer' else 40
            ea, eb = (errors[a], errors[b]) if kind == 'wer' else (4, 4)
            delta = (eb - ea) / denominator
            interval = {'lower': delta - .02, 'upper': delta + .02, 'units': 2, 'valid_replicates': 20}
            return {'measured': True, 'errors_a': ea, 'errors_b': eb, 'reference_length': denominator,
                    'rate_a': ea / denominator, 'rate_b': eb / denominator, 'delta_b_minus_a': delta,
                    'bootstrap': interval, 'session_cluster_bootstrap': interval,
                    'session_block_cluster_bootstrap': interval,
                    'rescue_count': 1, 'harm_count': 0, 'unchanged_count': 1, 'rescue_rate': .5, 'harm_rate': 0.0}
        results['comparisons'][f'{b}-minus-{a}'] = {
            'score_split': 'dev_report', 'split_manifest_hash': freeze['split_hash'],
            'freeze_manifest_hash': freeze['manifest_hash'], 'trial_count': 2,
            'seed': 7, 'replicates': 20, 'confidence_level': .95,
            'missing_session_count': 0, 'missing_block_count': 0,
            'runs': {'a': {'name': PRIVATE}, 'b': {'metadata': PRIVATE}},
            'wer': metric('wer'), 'per': metric('per')}
    confidence = {'freeze_manifest_hash': freeze['manifest_hash'], 'split_hash': freeze['split_hash'],
                  'score_split': 'dev_report', 'selected_lambda': .5, 'gate_threshold': .25,
                  'scorer_setup_seconds': 1.0, 'source_hashes': {'path': PRIVATE},
                  'shared_compact_decode': {'recorded_source_run_elapsed_seconds': 4., 'sum_trial_decode_seconds': 3.},
                  'conditions': {}}
    for name, calls in (('A0', 0), ('A1', 4), ('A2', 2), ('A3_seed7', 2)):
        confidence['conditions'][name] = {'lm_calls': calls, 'lm_candidates': calls * 3, 'lm_failures': 0,
                                        'scored_trial_lm_calls': calls // 2, 'call_fraction': calls / 4,
                                        'lm_total_seconds': calls * .1, 'called_latency_p95_seconds': .12 if calls else None,
                                        'trial_latency_p95_seconds': .11 if calls else .001,
                                        'call_bins': {'1-5': calls, '6-10': 0, '11+': 0},
                                        'raw_reference': PRIVATE}
    complete = {'state': 'complete', 'freeze_hash': freeze['manifest_hash'], 'rows': len(rows),
                'compact_selection': {'mode': 'kenlm', 'selected': {'weight': 1., 'candidate_run': PRIVATE}},
                'confidence': confidence, 'results': results, 'raw_reference': PRIVATE}
    for name, value in (('complete.json', complete), ('joint_freeze.json', freeze)):
        (study / name).write_text(json.dumps(value), encoding='utf-8')
    with (study / 'experiments.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    laptop = root / 'private_laptop_resources.json'
    laptop.write_text(json.dumps(resources), encoding='utf-8')
    return study, laptop, complete


class PublicEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), 'The public aggregate exporter has not been implemented')
        spec = importlib.util.spec_from_file_location('public_evidence', SCRIPT)
        self.public = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.public)

    def test_allowlist_retains_harmful_results_intervals_and_call_savings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study, laptop, _ = fixture(root)
            result = self.public.build_evidence(study, laptop_resources={'baseline_inference': laptop})
            self.assertEqual(result['state'], 'complete')
            self.assertEqual(len(result['results']['dev_report']), 6)
            self.assertGreater(result['comparisons']['A2-minus-A0']['wer']['delta_b_minus_a'], 0)
            self.assertEqual(result['confidence']['conditions']['A2']['lm_calls_saved_vs_every_trial'], 2)
            self.assertEqual(result['confidence']['conditions']['A2']['lm_call_savings_fraction'], .5)
            self.assertEqual(result['confidence']['call_scope'], 'full_validation')
            self.assertEqual(result['resources']['laptop']['baseline_inference']['peak_tree_rss_bytes'], 2_000_000_000)
            self.assertEqual(result['test_submissions']['state'], 'not_supplied')
            text = json.dumps(result) + self.public.markdown_report(result)
            self.assertNotIn(PRIVATE, text)
            self.assertNotIn(str(root), text)
            self.assertNotIn('oov_counts', text)
            self.assertNotIn('reference_text', text)

    def test_incomplete_or_mismatched_input_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study, _, complete = fixture(root)
            output = root / 'public'
            complete['state'] = 'running'
            (study / 'complete.json').write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, 'complete'):
                self.public.make_public_evidence(study, output)
            self.assertFalse(output.exists())
            complete['state'] = 'complete'
            complete['results']['dev_report']['A0']['metrics']['word']['errors'] += 1
            (study / 'complete.json').write_text(json.dumps(complete))
            with self.assertRaises(ValueError):
                self.public.make_public_evidence(study, output)
            self.assertFalse(output.exists())

    def test_freeze_content_and_csv_edits_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            study, _, _ = fixture(Path(directory))
            freeze = json.loads((study / 'joint_freeze.json').read_text())
            freeze['selection_rules'] = 'changed'
            (study / 'joint_freeze.json').write_text(json.dumps(freeze))
            with self.assertRaisesRegex(ValueError, 'freeze'):
                self.public.build_evidence(study)
        with tempfile.TemporaryDirectory() as directory:
            study, _, _ = fixture(Path(directory))
            path = study / 'experiments.csv'
            text = path.read_text().replace('0.4,8,20', '0.4,9,20', 1)
            self.assertNotEqual(text, path.read_text())
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, 'CSV'):
                self.public.build_evidence(study)

    def test_submission_counts_do_not_copy_text_and_require_exact_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study, _, _ = fixture(root)
            submission = root / 'submission.csv'
            submission.write_text(f'id,text\n0,{PRIVATE}\n1,\n', encoding='utf-8')
            result = self.public.build_evidence(study, submissions={'A0': submission}, expected_test_count=2)
            self.assertEqual(result['test_submissions']['conditions']['A0']['row_count'], 2)
            self.assertEqual(result['test_submissions']['conditions']['A0']['empty_text_count'], 1)
            self.assertNotIn(PRIVATE, json.dumps(result))
            submission.write_text(f'id,text\n0,{PRIVATE}\n2,wrong id\n')
            with self.assertRaisesRegex(ValueError, 'sequential'):
                self.public.build_evidence(study, submissions={'A0': submission}, expected_test_count=2)

    def test_csv_call_counters_must_match_the_confidence_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            study, _, complete = fixture(Path(directory))
            complete['confidence']['conditions']['A2']['lm_calls'] = 1
            complete['confidence']['conditions']['A2']['call_fraction'] = .25
            (study / 'complete.json').write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, 'CSV'):
                self.public.build_evidence(study)

    def test_laptop_fallback_and_encoder_parity_are_distinct_from_kenlm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study, _, complete = fixture(root)
            metrics = dict(complete['results']['dev_report']['A0']['metrics'], score_split='dev_tune', decoder='lexicon_only')
            baseline = root / 'laptop.json'
            baseline.write_text(json.dumps({'evaluation': metrics, 'platform': {'gpu': 'NVIDIA GeForce RTX 3080 Laptop GPU'},
                                            'runs': {'phoneme_inference': complete['results']['dev_report']['A0']['performance']},
                                            'history': PRIVATE}))
            parity = root / 'parity.json'
            parity.write_text(json.dumps({'trials': 4, 'exact_greedy_phoneme_agreement': 4,
                                          'trial_order_and_frame_lengths_match': True, 'references_loaded': False,
                                          'elapsed_seconds': 2., 'peak_tree_rss_bytes': 1_000_000_000,
                                          'gpu': 'NVIDIA A100-SXM4-40GB', 'command': PRIVATE}))
            result = self.public.build_evidence(study, laptop_baseline=baseline, encoder_parity=parity)
            self.assertEqual(result['laptop_baseline']['decoder'], 'lexicon_only')
            self.assertEqual(result['laptop_baseline']['score_split'], 'dev_tune')
            self.assertEqual(result['compact_decoder']['mode'], 'kenlm')
            self.assertEqual(result['encoder_parity']['exact_greedy_phoneme_agreement'], 4)
            self.assertIn('Laptop lexicon-only fallback', self.public.markdown_report(result))
            self.assertNotIn(PRIVATE, json.dumps(result))

    def test_examples_include_model_text_only_for_predeclared_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study, _, complete = fixture(root)
            ids = ['private/session/trial_1', 'private/session/trial_2']
            declared = {'ids': ids, 'ids_hash': object_hash(ids), 'split_hash': 'a' * 64}
            (study / 'example_ids.json').write_text(json.dumps(declared))
            examples = {'score_split': 'dev_report', 'split_manifest_hash': 'a' * 64,
                        'freeze_manifest_hash': complete['freeze_hash'], 'ids': ids, 'ids_hash': object_hash(ids),
                        'condition_order': ['A0', 'A1', 'A2', 'B1', 'B2'], 'markdown': PRIVATE,
                        'rows': [{'trial_id': trial, 'reference_text': PRIVATE, 'reference_phoneme_ids': [777],
                                  'predictions': {name: {'text': 'model output only', 'status': 'ok', 'phoneme_ids': [888],
                                                         'word_edit_counts': {'errors': 777}, 'raw_labels': PRIVATE}
                                                  for name in ('A0', 'A1', 'A2', 'B1', 'B2')}} for trial in ids]}
            source = study / 'fixed_examples.json'
            source.write_text(json.dumps(examples))
            result = self.public.build_evidence(study, model_examples=source)
            text = json.dumps(result['model_predictions'])
            self.assertIn('model output only', text)
            for private in (PRIVATE, 'private/session', 'reference_text', 'reference_phoneme_ids', 'phoneme_ids', 'word_edit_counts'):
                self.assertNotIn(private, text)
            self.assertEqual(len(result['model_predictions']['examples']), 2)
            for example in result['model_predictions']['examples']:
                for prediction in example['predictions'].values():
                    self.assertEqual(set(prediction), {'text', 'status'})
            declared['ids'] = list(reversed(ids))
            (study / 'example_ids.json').write_text(json.dumps(declared))
            with self.assertRaisesRegex(ValueError, 'predeclared'):
                self.public.build_evidence(study, model_examples=source)

    def test_real_matplotlib_export_writes_readable_static_figures(self):
        if importlib.util.find_spec('matplotlib') is None:
            self.skipTest('Optional matplotlib is required for release charts')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study, laptop, _ = fixture(root)
            output = root / 'public'
            self.public.make_public_evidence(study, output, laptop_resources={'baseline_inference': laptop})
            result = json.loads((output / 'results.json').read_text())
            self.assertEqual(result['state'], 'complete')
            markdown = (output / 'results.md').read_text()
            self.assertNotIn(PRIVATE, markdown)
            self.assertIn('Unusual and random', markdown)
            for stem in ('error_rates', 'paired_changes', 'rescoring_cost', 'resource_footprint'):
                png, svg = output / 'figures' / f'{stem}.png', output / 'figures' / f'{stem}.svg'
                self.assertTrue(png.read_bytes().startswith(b'\x89PNG'))
                self.assertGreater(png.stat().st_size, 10_000)
                self.assertIn('<svg', svg.read_text())
                self.assertNotIn(PRIVATE, svg.read_text())


if __name__ == '__main__':
    unittest.main()
