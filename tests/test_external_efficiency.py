import importlib
import unittest


class TimingTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.find_spec('dual2pose.eval.profile_external_efficiency')
        self.assertIsNotNone(spec, 'external timing runner is not implemented')
        return importlib.import_module('dual2pose.eval.profile_external_efficiency')

    def test_warmup_excluded_and_synchronization_surrounds_calls(self):
        m = self.module()
        events = []
        result = m.time_calls(lambda: events.append('call'), lambda: events.append('sync'), 2, 3)
        self.assertEqual(len(result['samples_ms']), 3)
        self.assertEqual(events.count('call'), 5)
        self.assertEqual(events[-9:], ['sync', 'call', 'sync'] * 3)
        self.assertTrue(all(v >= 0 for v in result['samples_ms']))

    def test_invalid_budget_rejected(self):
        m = self.module()
        for warm, repeats in [(0, 2), (1, 1)]:
            with self.assertRaises(ValueError):
                m.time_calls(lambda: None, lambda: None, warm, repeats)


if __name__ == '__main__':
    unittest.main()
