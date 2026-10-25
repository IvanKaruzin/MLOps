"""Several pinned winners must be evaluated and validated under one protocol."""

import unittest
from pathlib import Path
from unittest.mock import patch

import test_pipeline
from evaluation import cli
from evaluation.contract import read_json, write_json


class ComparisonPipeline(unittest.TestCase):
    def setUp(self):
        self.fixture = test_pipeline.PipelineContract(methodName='test_refused_candidate_keeps_alias')
        self.fixture.setUp()
        self.references = {
            'previous': dict(kind='adapter', name='old-winner', version='2', run_id='old-run', uri='models:/old-winner/2'),
            'sampling_control': dict(kind='adapter', name='control-winner', version='1', run_id='control-run', uri='models:/control-winner/1'),
            'unaligned': dict(kind='adapter', name='unaligned-winner', version='4', run_id='unaligned-run', uri='models:/unaligned-winner/4'),
        }
        config = read_json('evaluation/params.json')
        config['comparison_models'] = 'evaluation/comparison-models.json'
        write_json('evaluation/params.json', config)
        write_json(config['comparison_models'], self.references)
        self.registry_patch = patch.object(cli, 'registry_version')
        self.registry = self.registry_patch.start()

    def tearDown(self):
        self.registry_patch.stop()
        self.fixture.tearDown()

    def test_all_five_roles_share_protocol_ids_and_are_fully_verified(self):
        cli.evaluate()
        reports = cli.validated_reports()
        self.assertEqual(list(reports), ['baseline', 'previous', 'sampling_control', 'unaligned', 'candidate'])
        self.assertEqual(len({r['protocol_sha256'] for r in reports.values()}), 1)
        self.assertEqual(len({r['evaluation_run_id'] for r in reports.values()}), 1)
        self.assertEqual(len({tuple(x['id'] for x in r['per_item']) for r in reports.values()}), 1)
        self.assertEqual(reports['previous']['model'], self.references['previous'])
        self.assertEqual(reports['sampling_control']['model'], self.references['sampling_control'])
        self.assertEqual(len(read_json('metrics/comparison-diffs.json')), 8)
        direct = read_json('metrics/comparison-diffs.json')['unaligned_vs_candidate']
        self.assertEqual(direct['anchor_model'], self.references['unaligned'])
        self.assertEqual(direct['compared_model'], self.fixture.spec)
        for role in reports:
            self.assertTrue(Path(f'metrics/{role}-outputs.json').is_file())
        # The pinned old roles use registry source checks, without requiring the
        # live candidate alias to point to each historical version.
        checked = [call.args[1] for call in self.registry.call_args_list]
        self.assertIn(self.references['previous'], checked)
        self.assertIn(self.references['sampling_control'], checked)
        self.assertIn(self.references['unaligned'], checked)

    def test_changing_reference_spec_content_invalidates_protocol(self):
        cli.evaluate()
        before = cli.inputs()[-1]
        self.references['previous'].update(version='3', uri='models:/old-winner/3')
        write_json('evaluation/comparison-models.json', self.references)
        self.assertNotEqual(cli.inputs()[-1], before)
        with self.assertRaises(ValueError):
            cli.current_reports()

    def test_reference_raw_and_diff_tampering_block_gate(self):
        cli.evaluate()
        for role in ('previous', 'unaligned'):
            with self.subTest(role=role):
                path = f'metrics/{role}-outputs.json'
                original = read_json(path)
                changed = list(original)
                changed[0] = dict(changed[0], raw='changed')
                write_json(path, changed)
                with self.assertRaisesRegex(ValueError, 'сырые ответы'):
                    cli.current_reports()
                write_json(path, original)
        diffs = read_json('metrics/comparison-diffs.json')
        diffs['unaligned_vs_candidate']['improved'] = ['fake-id']
        write_json('metrics/comparison-diffs.json', diffs)
        with self.assertRaisesRegex(ValueError, 'diff'):
            cli.current_reports()

    def test_wrong_reference_role_or_evaluation_run_is_rejected(self):
        cli.evaluate()
        path = 'metrics/unaligned-eval.json'
        original = read_json(path)
        changed = dict(original, role='candidate')
        write_json(path, changed)
        with self.assertRaisesRegex(ValueError, 'роль'):
            cli.current_reports()
        write_json(path, dict(original, evaluation_run_id='different-evaluation'))
        with self.assertRaisesRegex(ValueError, 'одному конкретному'):
            cli.current_reports()

    def test_missing_unaligned_report_is_not_silently_dropped(self):
        cli.evaluate()
        Path('metrics/unaligned-eval.json').unlink()
        with self.assertRaises(FileNotFoundError):
            cli.current_reports()

    def test_reference_schema_rejects_new_alias_roles_or_unpinned_specs(self):
        for refs in [dict(champion=self.references['previous']),
                     dict(previous=dict(kind='base', name='base', revision='a' * 40)),
                     dict(previous=dict(self.references['previous'], uri='models:/old-winner@champion'))]:
            write_json('evaluation/comparison-models.json', refs)
            with self.assertRaises(ValueError):
                cli.inputs()


if __name__ == '__main__':
    unittest.main()
