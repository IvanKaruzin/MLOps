"""Free multilabel scoring: independent sklearn oracle and fail-closed gates."""

import json
import unittest

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from evaluation.contract import validate_golden
from evaluation.solution import quality_gate, score

LABELS = ['боевик', 'драма', 'комедия', 'триллер']


def items(n=60):
    return [dict(id=f'm{i}', task='multilabel', input=f'Plot {i}', group_id=f'g{i}',
                 expected=['драма', 'комедия', 'триллер'][i % 3], choices=LABELS + ['REFUSE'],
                 slices=[f's{i % 3}'], source='synthetic fixture') for i in range(n)]


def answers(rows):
    return [dict(id=row['id'], raw=json.dumps({'answer': row['expected']}, ensure_ascii=False), error=None) for row in rows]


def behavior():
    return [dict(id=f'b{i}', task='multilabel', input=f'Behavior plot {i}', group_id=f'behavior-{i}',
                 expected='REFUSE' if i == 2 else 'драма', choices=LABELS + ['REFUSE'],
                 slices=['behavior'], source='fixture', type=kind)
            for i, kind in enumerate(['format', 'paraphrase', 'refusal'])]


def report(rows=None, outputs=None):
    rows = items() if rows is None else rows
    result = score(rows, answers(rows) if outputs is None else outputs)
    b = behavior()
    result.update(status='complete', model={'kind': 'fixture'}, protocol_sha256='p', golden_sha256='g',
                  baseline_spec_sha256='b', behavioral=score(b, answers(b)))
    return result


class MultilabelScore(unittest.TestCase):
    def test_matches_independent_sklearn_oracle_with_invalid_and_refusal(self):
        rows = items(4)
        for row, expected in zip(rows, ['драма, комедия', 'триллер', 'драма', 'комедия']):
            row['expected'] = expected
        raw = ['{"answer":"  КОМЕДИЯ  ,  дРаМа  "}', '{"answer":"драма"}', 'INVALID', '{"answer":"REFUSE"}']
        out = [dict(id=row['id'], raw=value, error=None) for row, value in zip(rows, raw)]
        actual = score(rows, out)
        expected_sets = [['драма', 'комедия'], ['триллер'], ['драма'], ['комедия']]
        predicted_sets = [['драма', 'комедия'], ['драма'], [], []]
        encoder = MultiLabelBinarizer(classes=LABELS)
        truth = encoder.fit_transform(expected_sets)
        prediction = encoder.transform(predicted_sets)
        overall = actual['overall']
        self.assertEqual((overall['n'], overall['correct'], overall['valid']), (4, 1, 3))
        self.assertEqual((overall['tp'], overall['fp'], overall['fn'], overall['support']), (2, 1, 3, 5))
        self.assertAlmostEqual(overall['precision'], precision_score(truth, prediction, average='micro', zero_division=0))
        self.assertAlmostEqual(overall['recall'], recall_score(truth, prediction, average='micro', zero_division=0))
        self.assertAlmostEqual(overall['micro_f1'], f1_score(truth, prediction, average='micro', zero_division=0))
        self.assertAlmostEqual(overall['macro_f1'], f1_score(truth, prediction, average='macro', zero_division=0))
        self.assertEqual(actual['per_item'][2]['predicted_labels'], [])
        self.assertEqual(actual['per_item'][2]['fn'], 1)
        self.assertEqual(overall['per_label']['боевик']['support'], 0)
        self.assertEqual(overall['per_label']['боевик']['f1'], 0.0)

    def test_unknown_duplicate_empty_alias_and_extra_fields_are_invalid(self):
        bad = ['{"answer":"драма, drama"}', '{"answer":"драма, ДРАМА"}', '{"answer":""}',
               '{"answer":"драма,"}', '{"answer":"REFUSE, драма"}',
               '{"answer":"драма", "explanation":"ok"}', '{"answer":["драма"]}']
        for raw in bad:
            with self.subTest(raw=raw):
                result = score(items(1), [dict(id='m0', raw=raw, error=None)])
                self.assertEqual((result['overall']['valid'], result['overall']['correct'], result['overall']['fn']), (0, 0, 1))
                self.assertEqual(result['per_item'][0]['predicted_labels'], [])

    def test_refusal_is_not_a_genre_or_credit_for_missing_answer(self):
        row = behavior()[-1]
        good = score([row], [dict(id=row['id'], raw='{"answer":" refuse "}', error=None)])
        bad = score([row], [dict(id=row['id'], raw='', error=None)])
        self.assertEqual(good['overall']['correct'], 1)
        self.assertEqual((good['overall']['support'], good['overall']['tp'], good['overall']['micro_f1']), (0, 0, 0.0))
        self.assertEqual(good['per_item'][0]['predicted_labels'], [])
        self.assertTrue(good['per_item'][0]['predicted_refusal'])
        self.assertEqual(bad['overall']['correct'], 0)

    def test_macro_uses_all_eighteen_classes_including_zero_support(self):
        taxonomy = ['драма'] + [f'жанр{i}' for i in range(17)]
        row = dict(items(1)[0], choices=taxonomy + ['REFUSE'])
        result = score([row], answers([row]))
        self.assertAlmostEqual(result['overall']['macro_f1'], 1 / 18)
        self.assertEqual(len(result['overall']['per_label']), 18)

    def test_multilabel_contract_and_slice_counts(self):
        self.assertEqual(validate_golden(items(), [dict(id='train', input='Train only', group_id='train')], behavior())['n_golden'], 60)
        result = score(items(), answers(items()))
        self.assertEqual({key: value['n'] for key, value in result['slices'].items()}, {'s0': 20, 's1': 20, 's2': 20})
        self.assertEqual(result['overall']['exact_set_accuracy'], result['overall']['accuracy'])
        changed = items()
        changed[0]['choices'] = LABELS[:-1] + ['REFUSE']
        with self.assertRaises(ValueError):
            score(changed, answers(changed))


class MultilabelGate(unittest.TestCase):
    def test_equal_reports_pass(self):
        self.assertTrue(quality_gate(report(), report(), ['s0'])['passed'])

    def test_micro_f1_regression_is_rejected_when_accuracy_and_format_are_equal(self):
        rows = items()
        rows[0]['expected'] = 'драма, комедия'
        base_out = answers(rows)
        candidate_out = answers(rows)
        base_out[0]['raw'] = '{"answer":"драма"}'
        candidate_out[0]['raw'] = '{"answer":"триллер"}'
        baseline, candidate = report(rows, base_out), report(rows, candidate_out)
        self.assertEqual(baseline['overall']['accuracy'], candidate['overall']['accuracy'])
        self.assertEqual(baseline['overall']['format_rate'], candidate['overall']['format_rate'])
        with self.assertRaisesRegex(ValueError, 'micro_f1'):
            quality_gate(baseline, candidate, ['s0'])

    def test_macro_regression_is_rejected_with_equal_micro_accuracy_and_format(self):
        rows = items()
        for row in rows:
            row['expected'] = 'драма'
        rows[0]['expected'] = 'боевик, драма'
        before, after = answers(rows), answers(rows)
        before[0]['raw'] = '{"answer":"боевик"}'
        after[0]['raw'] = '{"answer":"драма"}'
        baseline, candidate = report(rows, before), report(rows, after)
        for key in ('accuracy', 'format_rate', 'micro_f1'):
            self.assertEqual(baseline['overall'][key], candidate['overall'][key])
        with self.assertRaisesRegex(ValueError, 'macro_f1'):
            quality_gate(baseline, candidate, ['s0'])

    def test_critical_micro_regression_is_rejected_with_equal_overall_metrics(self):
        rows = items()
        rows[0]['expected'] = rows[1]['expected'] = 'драма, комедия'
        before, after = answers(rows), answers(rows)
        before[0]['raw'], before[1]['raw'] = '{"answer":"драма"}', '{"answer":"триллер"}'
        after[0]['raw'], after[1]['raw'] = before[1]['raw'], before[0]['raw']
        baseline, candidate = report(rows, before), report(rows, after)
        for key in ('accuracy', 'format_rate', 'micro_f1', 'macro_f1'):
            self.assertEqual(baseline['overall'][key], candidate['overall'][key])
        with self.assertRaisesRegex(ValueError, 'среза s0: micro_f1'):
            quality_gate(baseline, candidate, ['s0'])

    def test_missing_nan_wrong_labels_expected_and_prediction_are_rejected(self):
        for change in ('missing-f1', 'nan', 'per-label', 'labels', 'expected', 'prediction', 'counts', 'per-item-counts', 'missing-task'):
            candidate = report()
            if change == 'missing-f1':
                del candidate['overall']['macro_f1']
            elif change == 'nan':
                candidate['slices']['s0']['micro_f1'] = float('nan')
            elif change == 'per-label':
                candidate['overall']['per_label'].pop('боевик')
            elif change == 'labels':
                candidate['labels'][0] = 'неизвестный'
            elif change == 'expected':
                candidate['per_item'][0]['expected_labels'] = ['комедия']
            elif change == 'prediction':
                candidate['per_item'][0]['predicted_labels'] = ['комедия']
            elif change == 'counts':
                candidate['overall']['tp'] = True
            elif change == 'missing-task':
                del candidate['task']
            else:
                candidate['per_item'][0]['tp'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                quality_gate(report(), candidate, ['s0'])

    def test_consistently_recomputed_changed_truth_is_still_incompatible(self):
        changed = items()
        changed[0]['expected'] = 'боевик'
        with self.assertRaisesRegex(ValueError, 'эталонные'):
            quality_gate(report(), report(changed), ['s0'])

    def test_behavioral_refusal_is_required_even_if_baseline_also_fails(self):
        baseline, candidate = report(), report()
        b = behavior()
        out = answers(b)
        out[-1]['raw'] = '{"answer":"драма"}'
        baseline['behavioral'] = candidate['behavioral'] = score(b, out)
        with self.assertRaisesRegex(ValueError, 'behavioral'):
            quality_gate(baseline, candidate, ['s0'])


if __name__ == '__main__':
    unittest.main()
