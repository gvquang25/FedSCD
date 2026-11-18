"""Train/validation isolation checks; run with python -m unittest test_tune_fedscd."""
import unittest

from tune_fedscd import split_indices


class SplitTests(unittest.TestCase):
    def test_disjoint_complete_reproducible_and_stratified(self):
        labels = [0] * 10 + [1] * 5 + [2]
        train, val = split_indices(labels, 0.2, 2006)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(sorted(train + val), list(range(len(labels))))
        self.assertEqual((train, val), split_indices(labels, 0.2, 2006))
        self.assertEqual([labels[i] for i in val].count(0), 2)
        self.assertEqual([labels[i] for i in val].count(1), 1)
        self.assertIn(15, train)

    def test_all_singletons_rejected(self):
        with self.assertRaises(ValueError):
            split_indices([0, 1, 2], 0.2, 2006)

    def test_each_class_retains_training_sample(self):
        labels = [0, 0, 1, 1]
        train, val = split_indices(labels, 0.9, 2006)
        self.assertEqual(sorted(labels[i] for i in train), [0, 1])
        self.assertEqual(sorted(labels[i] for i in val), [0, 1])


if __name__ == "__main__":
    unittest.main()
