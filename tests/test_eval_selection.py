"""Explicit finished-model assessment must not masquerade as a completed sweep."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluation import cli
from evaluation.contract import fingerprint, read_json, write_json


class ExplicitSelection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cwd = Path.cwd()
        os.chdir(self.temp.name)
        self.stack = ExitStack()
        self.params = dict(model=dict(name='base', revision='a' * 40, dtype='bfloat16'),
                           train=dict(max_steps=2, batch_size=4, gradient_checkpointing=True),
                           sampling=dict(method='full_shuffle', seed=17),
                           lora=dict(alpha_over_r=2, dropout=0.05),
                           registry=dict(name='model', max_val_ratio=1.0),
                           sweep=[dict(name='all-r8-low', r=8, lr=0.0001, target_modules=['q_proj'])])
        keys = ('code_sha256', 'config_sha256', 'data_train_sha256', 'data_val_sha256',
                'source_train_sha256', 'source_val_sha256', 'tokenization_params_sha256', 'auxiliary_sha256')
        self.current = {key: str(index) * 64 for index, key in enumerate(keys)}
        self.source = dict(passed=True, tokenization_params_sha256=self.current['tokenization_params_sha256'],
                           tokenizer={key: self.params['model'][key] for key in ('name', 'revision')}, splits={})
        for split in ('train', 'val'):
            self.source['splits'][split] = dict(source_sha256=self.current[f'source_{split}_sha256'],
                tensor_sha256=self.current[f'data_{split}_sha256'], source_records=2, tensor_records=2,
                all_input_ids_attention_masks_labels_equal=True)
        source_sha = hashlib.sha256(json.dumps(self.source, sort_keys=True, allow_nan=False).encode()).hexdigest()
        configuration = self.params['sweep'][0]
        logged = dict(r='8', lr='0.0001', target_modules='q_proj', model='base', dtype='bfloat16',
                      **{'sampling.method': 'full_shuffle', 'sampling.seed': '17',
                         'alpha_over_r': '2', 'dropout': '0.05'},
                      **{f'train.{key}': str(value) for key, value in self.params['train'].items()})
        self.run = SimpleNamespace(info=SimpleNamespace(run_id='child', status='FINISHED', experiment_id='exp',
                                                       artifact_uri='mlflow-artifacts:/exp/child/artifacts'),
            data=SimpleNamespace(tags={**self.current, 'kind': 'candidate', 'candidate': configuration['name'],
                                       'mlflow.parentRunId': 'parent', 'base_revision': 'a' * 40,
                                       'source_link_sha256': source_sha, 'sampling_sha256': '8' * 64,
                                       'training_view_sha256': '9' * 64},
                                 params=logged, metrics=dict(steps=2, final_val_loss=0.2, base_val_loss=1.0)))
        self.parent = SimpleNamespace(info=SimpleNamespace(run_id='parent', status='KILLED', experiment_id='exp'),
                                      data=SimpleNamespace(tags=dict(self.current)))
        self.checkpoint = dict(adapter_sha256='f' * 64, checkpoints=[dict(step=0, val_loss=1.0),
                               dict(step=1, val_loss=0.3), dict(step=2, val_loss=0.2)])
        self.histories = dict(train_loss=[SimpleNamespace(step=i, value=0.4) for i in (1, 2)],
                              val_loss=[SimpleNamespace(step=point['step'], value=point['val_loss'])
                                        for point in self.checkpoint['checkpoints']])
        self.client = Mock()
        self.client.get_run.side_effect = lambda identity: {'child': self.run, 'parent': self.parent}[identity]
        self.client.get_metric_history.side_effect = lambda identity, key: self.histories[key]
        self.client.download_artifacts.side_effect = self.download
        self.stack.enter_context(patch('src.common.provenance', return_value={**self.current,
                                                                              'git_sha': 'old', 'git_dirty': 'True'}))
        self.sampling = self.stack.enter_context(patch('src.promote.validate_sampling', return_value={'proof': 'sampling'}))
        self.view = self.stack.enter_context(patch('src.promote.validate_training_view'))
        self.checkpoints = self.stack.enter_context(patch('src.promote.validate_checkpoints'))
        self.stack.enter_context(patch.object(cli, 'tracking', return_value=(self.params, self.client)))
        self.version = SimpleNamespace(name='model', version='6', run_id='child', status='READY',
                                       source='runs:/child/model', tags={})
        self.client.get_model_version.return_value = self.version
        self.client.get_model_version_by_alias.return_value = self.version
        self.spec = dict(kind='adapter', name='model', version='6', run_id='child', uri='models:/model/6')
        write_json('metrics/sweep.json', dict(parent_id='parent', complete=False, rows=[]))

    def tearDown(self):
        self.stack.close()
        os.chdir(self.cwd)
        self.temp.cleanup()

    def download(self, identity, name, directory):
        values = {'config.json': self.params, 'source-link-check.json': self.source,
                  'checkpoint.json': self.checkpoint}
        path = Path(directory) / name
        write_json(path, values[name])
        return str(path)

    def selected(self):
        return cli.explicit_training_run(self.client, self.params, 'child', 'User requested assessment only')

    def pin(self):
        _, evidence = self.selected()
        evidence['model'] = self.spec
        self.version.tags = {'selection.mode': cli.EXPLICIT_SELECTION, 'selection.sha256': fingerprint(evidence)}
        write_json(cli.SELECTION, evidence)
        write_json(cli.CANDIDATE, self.spec)
        return evidence

    def test_finished_child_of_cancelled_parent_keeps_incomplete_sweep_explicit(self):
        run, evidence = self.selected()
        self.assertIs(run, self.run)
        self.assertEqual(evidence['parent_status'], 'KILLED')
        self.assertTrue(evidence['source_sweep_cancelled'])
        self.assertFalse(evidence['fair_sweep_verified'])
        self.sampling.assert_called_once_with(self.client, self.params, [self.run])
        self.view.assert_called_once_with(self.client, self.params, [self.run], {'proof': 'sampling'})
        self.checkpoints.assert_called_once_with(self.client, self.params, [self.run], {'proof': 'sampling'})
        self.client.set_registered_model_alias.assert_not_called()

    def test_status_versions_configuration_and_budget_fail_before_registration(self):
        mutations = [lambda: setattr(self.run.info, 'status', 'RUNNING'),
                     lambda: setattr(self.parent.info, 'status', 'RUNNING'),
                     lambda: setattr(self.parent.info, 'status', 'FAILED'),
                     lambda: setattr(self.parent.info, 'experiment_id', 'other'),
                     lambda: self.run.data.tags.update(auxiliary_sha256='changed'),
                     lambda: self.parent.data.tags.update(code_sha256='changed'),
                     lambda: self.run.data.tags.update(base_revision='b' * 40),
                     lambda: self.run.data.params.update(lr='0.0003'),
                     lambda: self.run.data.metrics.update(steps=1)]
        original_run, original_parent = copy.deepcopy(self.run), copy.deepcopy(self.parent)
        with patch('mlflow.register_model') as register:
            for mutate in mutations:
                self.run, self.parent = copy.deepcopy(original_run), copy.deepcopy(original_parent)
                mutate()
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    cli.prepare_candidate('child', 'explicit')
            register.assert_not_called()
        self.client.set_registered_model_alias.assert_not_called()

    def test_server_config_and_source_proof_are_not_trusted_from_tags(self):
        original = self.client.download_artifacts.side_effect
        for artifact in ('config.json', 'source-link-check.json'):
            def changed(identity, name, directory):
                path = original(identity, name, directory)
                if name == artifact:
                    value = read_json(path)
                    if name == 'config.json': value['train']['batch_size'] = 8
                    else: value['splits']['train']['all_input_ids_attention_masks_labels_equal'] = False
                    write_json(path, value)
                return path
            with self.subTest(artifact=artifact):
                self.client.download_artifacts.side_effect = changed
                with self.assertRaises(ValueError): self.selected()
        self.client.download_artifacts.side_effect = original

    def test_full_histories_and_all_existing_integrity_validators_are_required(self):
        self.histories['train_loss'].pop()
        with self.assertRaises(ValueError): self.selected()
        self.histories['train_loss'].append(SimpleNamespace(step=2, value=0.4))
        self.histories['val_loss'][1].value = 0.9
        with self.assertRaises(ValueError): self.selected()
        self.histories['val_loss'][1].value = 0.3
        for validator in (self.sampling, self.view, self.checkpoints):
            with self.subTest(validator=validator):
                validator.side_effect = ValueError('server proof changed')
                with self.assertRaises(ValueError): self.selected()
                validator.side_effect = None

    def test_explicit_registration_binds_evidence_and_changes_only_candidate_alias(self):
        before = Path('metrics/sweep.json').read_bytes()
        with patch('mlflow.register_model', return_value=self.version) as register:
            cli.prepare_candidate('child', '  User requested assessment only  ')
        register.assert_called_once_with('runs:/child/model', 'model')
        self.client.set_registered_model_alias.assert_called_once_with('model', 'candidate', '6')
        evidence = read_json(cli.SELECTION)
        self.assertEqual(evidence['model'], self.spec)
        self.assertEqual(evidence['reason'], 'User requested assessment only')
        self.client.set_model_version_tag.assert_any_call('model', '6', 'selection.sha256', fingerprint(evidence))
        self.assertEqual(Path('metrics/sweep.json').read_bytes(), before)
        self.assertEqual(self.parent.info.status, 'KILLED')

    def test_exact_pin_revalidates_weights_and_rejects_missing_or_changed_evidence(self):
        evidence = self.pin()
        self.assertEqual(cli.candidate_model(), self.spec)
        self.checkpoints.side_effect = ValueError('packaged weights changed')
        with self.assertRaises(ValueError): cli.candidate_model()
        self.checkpoints.side_effect = None
        changed = dict(evidence, reason='silently changed')
        write_json(cli.SELECTION, changed)
        with self.assertRaises(ValueError): cli.candidate_model()
        cli.SELECTION.unlink()
        with self.assertRaises(FileNotFoundError): cli.candidate_model()

    def test_alias_and_registry_binding_remain_strict(self):
        self.pin()
        self.client.get_model_version_by_alias.return_value = SimpleNamespace(version='5', run_id='old')
        with self.assertRaises(ValueError): cli.candidate_model()
        self.client.get_model_version_by_alias.return_value = self.version
        self.version.source = 'runs:/other/model'
        with self.assertRaises(ValueError): cli.candidate_model()

    def test_default_selection_still_uses_strict_six_run_select(self):
        with patch('src.promote.select', side_effect=ValueError('incomplete six-run sweep')) as select:
            with self.assertRaises(ValueError): cli.prepare_candidate()
        select.assert_called_once()
        self.client.set_registered_model_alias.assert_not_called()

    def test_even_passing_answer_gate_cannot_release_an_incomplete_sweep(self):
        self.pin()
        with patch.object(cli, 'gate', return_value={'model': self.spec}):
            with self.assertRaisesRegex(ValueError, 'полного sweep'):
                cli.release()
        self.client.set_registered_model_alias.assert_not_called()

    def test_missing_local_selection_after_gate_cannot_hide_explicit_registry_mode(self):
        self.pin()
        def gate_then_remove_selection():
            cli.SELECTION.unlink()
            return {'model': self.spec}
        with patch.object(cli, 'gate', side_effect=gate_then_remove_selection):
            with self.assertRaisesRegex(ValueError, 'полного sweep'):
                cli.release()
        self.client.set_registered_model_alias.assert_not_called()

    def test_reason_is_required_before_any_registry_mutation(self):
        with patch('mlflow.register_model') as register:
            for reason in (None, '', '   '):
                with self.subTest(reason=reason), self.assertRaises(ValueError):
                    cli.prepare_candidate('child', reason)
            register.assert_not_called()

    def test_inputs_changing_during_validation_fail_before_registration(self):
        with (patch('src.common.provenance', side_effect=[self.current, {**self.current, 'auxiliary_sha256': 'changed'}]),
              patch('mlflow.register_model') as register):
            with self.assertRaisesRegex(ValueError, 'во время проверки'):
                cli.prepare_candidate('child', 'explicit assessment')
            register.assert_not_called()
        self.client.set_registered_model_alias.assert_not_called()


if __name__ == '__main__':
    unittest.main()
