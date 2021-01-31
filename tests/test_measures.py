from unittest import TestCase

from measures import Measures


class TestMeasures(TestCase):
    def test_accuracy(self):
        with self.assertRaises(ValueError):
            Measures.accuracy([1, 1, 0], [1, 1])
        with self.assertRaises(ValueError):
            Measures.accuracy([1, 1], [1, 1, 0])

        #  tp, tn, fp, fn, pr, r, f1, acc
        measures = 0, 0, 0, 0, 1, 1, 1, 1
        self.assertEqual(measures, Measures.accuracy([], []))

        #  tp, tn, fp, fn, pr, r, f1, acc
        measures = 1, 1, 0, 0, 1, 1, 1, 1
        self.assertEqual(measures, Measures.accuracy([1, 0], [1, 0]))

        #  tp, tn, fp, fn, pr, r, f1, acc
        measures = 0, 2, 0, 0, 1, 1, 1, 1
        self.assertEqual(measures, Measures.accuracy([0, 0], [0, 0]))

        #  tp, tn, fp, fn, pr, r, f1, acc
        measures = 0, 0, 1, 1, 0, 0, 0, 0
        self.assertEqual(measures, Measures.accuracy([1, 0], [0, 1]))

        tp, tn, fp, fn, pr, r, f1, acc = 5, 6, 2, 1, 0.71, 0.83, 0.77, 0.79
        r_tp, r_tn, r_fp, r_fn, r_pr, r_r, r_f1, r_acc = Measures.accuracy(
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        )
        self.assertEqual((tp, tn, fp, fn), (r_tp, r_tn, r_fp, r_fn))
        self.assertAlmostEqual(pr, r_pr, places=2)
        self.assertAlmostEqual(r, r_r, places=2)
        self.assertAlmostEqual(f1, r_f1, places=2)
        self.assertAlmostEqual(acc, r_acc, places=2)
