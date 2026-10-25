"""Дополнительные регрессии: строгий JSON и согласованность gate/provenance."""

import unittest
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluation import cli
from evaluation.contract import strict_json_loads, validate_golden
from evaluation.solution import quality_gate, score, validate_model_spec
from test_eval import outputs, report, rows


class StrictScoring(unittest.TestCase):
    def test_duplicate_json_keys_and_nonfinite_constants_are_invalid(self):
        for raw in ['{"answer":"A","answer":"B"}', '{"answer":NaN}',
                    '{"answer":Infinity}', '{"answer":1e999}', '{"answer":"B"} trailing', '[{"answer":"B"}]']:
            with self.subTest(raw=raw):
                result = score(rows(1), [dict(id='q0', raw=raw, error=None)])
                self.assertEqual(result['overall']['valid'], 0)
                self.assertEqual(len(result['per_item']), 1)
                json.dumps(result, allow_nan=False)

    def test_nonstring_raw_is_contract_error(self):
        with self.assertRaises(ValueError):
            score(rows(1), [dict(id='q0', raw=float('nan'), error=None)])

    def test_error_with_correct_raw_remains_in_denominator(self):
        result = score(rows(1), [dict(id='q0', raw='{"answer":"B"}', error='worker failed')])
        self.assertEqual(result['overall'], dict(n=1, correct=0, valid=0, errors=1,
                                                accuracy=0.0, format_rate=0.0))

    def test_duplicate_rows_and_slices_are_contract_errors(self):
        bad_rows = rows(2)
        bad_rows[1]['id'] = bad_rows[0]['id']
        with self.assertRaises(ValueError):
            score(bad_rows, outputs(rows(2)))
        bad_rows = rows(1)
        bad_rows[0]['slices'] *= 2
        with self.assertRaises(ValueError):
            score(bad_rows, outputs(rows(1)))

    def test_json_files_reject_duplicate_keys(self):
        with self.assertRaises(ValueError):
            strict_json_loads('{"revision":"old", "revision":"new"}')


class DefensiveGate(unittest.TestCase):
    def test_empty_critical_list_rejected(self):
        with self.assertRaises(ValueError):
            quality_gate(report(), report(), [])

    def test_bool_counts_and_infinite_rates_rejected(self):
        for key, value in [('correct', True), ('accuracy', float('inf')), ('format_rate', -0.1)]:
            candidate = report()
            candidate['overall'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                quality_gate(report(), candidate, ['s0'])

    def test_inconsistent_slice_or_per_item_rejected(self):
        for change in ('slice', 'membership', 'valid', 'parsed', 'missing-item', 'behavior'):
            candidate = report()
            if change == 'slice':
                candidate['slices']['s0'].update(correct=16, accuracy=16 / 17)
            elif change == 'membership':
                candidate['per_item'][0]['slices'] = ['s1']
            elif change == 'valid':
                candidate['per_item'][0]['valid'] = False
            elif change == 'parsed':
                candidate['per_item'][0]['parsed'] = {'answer': 'A'}
            elif change == 'missing-item':
                candidate['per_item'].pop()
            else:
                candidate['behavioral']['overall']['correct'] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                quality_gate(report(), candidate, ['s0'])

    def test_baseline_generation_error_rejected_too(self):
        baseline = report()
        baseline['behavioral'] = score(rows(3), [dict(id=r['id'], raw='', error='timeout') for r in rows(3)])
        with self.assertRaises(ValueError):
            quality_gate(baseline, report(), ['s0'])

    def test_real_model_specs_require_pinned_revision_and_uri(self):
        bad = [dict(kind='base', name='base', revision='main'),
               dict(kind='adapter', name='adapter', version='1', run_id='run', uri='models:/other/1')]
        for model in bad:
            with self.subTest(model=model), self.assertRaises(ValueError):
                validate_model_spec(model)

    def test_critical_regression_rejected_even_with_overall_improvement(self):
        items = rows()
        baseline_outputs = [dict(id=r['id'], raw='{"answer":"B"}' if 's0' in r['slices'] else '{"answer":"A"}', error=None)
                            for r in items]
        candidate_outputs = outputs(items)
        candidate_outputs[0]['raw'] = '{"answer":"A"}'
        baseline, candidate = report(), report()
        baseline.update(score(items, baseline_outputs))
        candidate.update(score(items, candidate_outputs))
        self.assertGreater(candidate['overall']['accuracy'], baseline['overall']['accuracy'])
        with self.assertRaisesRegex(ValueError, 'критического среза'):
            quality_gate(baseline, candidate, ['s0'])

    def test_registry_model_source_is_bound_to_training_run(self):
        spec = dict(kind='adapter', name='my-model', version='2', run_id='run', uri='models:/my-model/2')
        client = Mock()
        client.get_run.return_value = SimpleNamespace(info=SimpleNamespace(artifact_uri='mlflow-artifacts:/1/run/artifacts'))
        client.get_model_version.return_value = SimpleNamespace(run_id='run', status='READY',
                                                                source='mlflow-artifacts:/1/run/artifacts/model')
        self.assertEqual(cli.registry_version(client, spec).run_id, 'run')
        client.get_model_version.return_value.source = 'mlflow-artifacts:/1/other-run/artifacts/model'
        with self.assertRaises(ValueError):
            cli.registry_version(client, spec)

    def test_failure_recorded_before_release(self):
        with patch.object(cli, 'current_reports', side_effect=ValueError('regression')), \
                patch.object(cli, 'record_gate') as record:
            with self.assertRaises(ValueError):
                cli.gate()
            self.assertFalse(record.call_args.args[0]['passed'])
            self.assertEqual(record.call_args.args[0]['reason'], 'regression')
            self.assertIs(record.call_args.args[0]['reports_verified'], False)
            self.assertNotIn('candidate_model', record.call_args.args[0])
            self.assertNotIn('evaluation_run_id', record.call_args.args[0])


class GateEvidence(unittest.TestCase):
    def test_quality_rejection_keeps_verified_models_protocol_and_raw_hashes(self):
        baseline, candidate = report(), report(49)
        for role, result in [('baseline', baseline), ('candidate', candidate)]:
            result.update(evaluation_run_id='same-eval', outputs_sha256=role + '-raw-hash')
        with patch.object(cli, 'current_reports', return_value=(baseline, candidate)), \
                patch.object(cli, 'read_json', return_value={'critical_slices': ['s0']}), \
                patch.object(cli, 'record_gate') as record:
            with self.assertRaises(ValueError):
                cli.gate()
            saved = record.call_args.args[0]
            self.assertIs(saved['reports_verified'], True)
            self.assertFalse(saved['passed'])
            self.assertEqual(saved['evaluation_run_id'], 'same-eval')
            self.assertEqual(saved['baseline_model'], baseline['model'])
            self.assertEqual(saved['candidate_model'], candidate['model'])
            self.assertEqual(saved['baseline_outputs_sha256'], 'baseline-raw-hash')
            self.assertEqual(saved['candidate_outputs_sha256'], 'candidate-raw-hash')
            for key in ['protocol_sha256', 'golden_sha256', 'baseline_spec_sha256']:
                self.assertEqual(saved[key], candidate[key])

    def test_mlflow_gate_tags_link_verified_evaluation_and_candidate(self):
        baseline, candidate = report(), report(49)
        candidate['model'] = dict(kind='adapter', name='model', version='2', run_id='training-run', uri='models:/model/2')
        for role, result in [('baseline', baseline), ('candidate', candidate)]:
            result.update(evaluation_run_id='evaluation-run', outputs_sha256=role + '-raw-hash')
        result = dict(passed=False, reason='regression', **cli.report_evidence(baseline, candidate))
        fake_mlflow = Mock()
        fake_mlflow.start_run.return_value.__enter__ = Mock(return_value=SimpleNamespace(info=SimpleNamespace(run_id='gate-run')))
        fake_mlflow.start_run.return_value.__exit__ = Mock(return_value=False)
        with patch.dict(sys.modules, {'mlflow': fake_mlflow}), \
                patch.object(cli, 'tracking'), patch.object(cli, 'write_json'):
            cli.record_gate(result, 'metrics/quality-gate.json', 'quality-gate')
        tags = fake_mlflow.set_tags.call_args.args[0]
        self.assertEqual(tags['reports_verified'], 'true')
        self.assertEqual(tags['evaluation_run_id'], 'evaluation-run')
        self.assertEqual(tags['candidate_run_id'], 'training-run')
        self.assertEqual(tags['candidate_model_uri'], 'models:/model/2')
        self.assertEqual(tags['protocol_sha256'], candidate['protocol_sha256'])

    def test_repeat_evaluation_does_not_log_stale_negative_reports(self):
        import test_pipeline
        from evaluation.contract import write_json
        fixture = test_pipeline.PipelineContract(methodName='test_refused_candidate_keeps_alias')
        fixture.setUp()
        try:
            write_json('metrics/negative-eval.json', {'stale': True})
            write_json('metrics/negative-outputs.json', [{'stale': True}])
            cli.evaluate()
            artifacts = [call.args[0] for call in sys.modules['mlflow'].log_artifact.call_args_list]
            self.assertIn('metrics/candidate-eval.json', artifacts)
            self.assertIn('metrics/baseline-outputs.json', artifacts)
            self.assertNotIn('metrics/negative-eval.json', artifacts)
            self.assertNotIn('metrics/negative-outputs.json', artifacts)
        finally:
            fixture.tearDown()


class BehavioralLeakage(unittest.TestCase):
    def test_behavioral_train_overlap_rejected(self):
        behavior = [dict(id=f'b{i}', group_id=f'b{i}', input=f'Behavior {i}', choices=['B'],
                         expected='B', slices=['behavior'], source='fixture', type=kind)
                    for i, kind in enumerate(['format', 'paraphrase', 'refusal'])]
        training = [dict(id='t', group_id='b0', input='Another question')]
        with self.assertRaises(ValueError):
            validate_golden(rows(), training, behavior)

    def test_broken_explicit_paraphrase_pair_rejected(self):
        behavior = [dict(id=f'b{i}', group_id=f'b{i}', input=f'Behavior {i}', choices=['B'],
                         expected='B', slices=['behavior'], source='fixture', type=kind)
                    for i, kind in enumerate(['format', 'paraphrase', 'refusal'])]
        behavior[1]['paired_with'] = 'missing'
        with self.assertRaises(ValueError):
            validate_golden(rows(), [dict(id='t', group_id='t', input='Train')], behavior)


if __name__ == '__main__':
    unittest.main()
