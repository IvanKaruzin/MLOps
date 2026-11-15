"""Conservation of all attempts, including rejected and truncated streams."""
import unittest
from hw11.client import summarize
from scripts.hw11_experiments import schedule


class AccountingTests(unittest.TestCase):
    def test_terminal_categories_and_success_quantiles(self):
        rows=[dict(outcome=name,pod='a',ttfc_seconds=1,full_seconds=2) for name in
              ['completed','completed','rejected','interrupted','timeout','connection_error','error','unsent']]
        result=summarize(rows)
        self.assertEqual(result['planned'],8)
        self.assertEqual(result['sent'],7)
        self.assertEqual(sum(result['counts'].values()),8)
        self.assertEqual(result['successful'],2)
        self.assertEqual(result['ttfc_p95_seconds'],1)
    def test_no_success_has_no_percentile(self):
        result=summarize([dict(outcome='rejected',pod='a')])
        self.assertIsNone(result['ttfc_p50_seconds'])
        self.assertIsNone(result['full_p95_seconds'])
    def test_workload_schedule_is_identical_and_skew_is_eighty_percent(self):
        first=schedule(120,2,'skewed')
        self.assertEqual(first,schedule(120,2,'skewed'))
        self.assertEqual(sum(r['client']==0 for r in first),192)
        self.assertEqual(len({r['client'] for r in first}),4)
        self.assertEqual(len(first),240)


if __name__=='__main__':unittest.main()
