import os
import unittest

from dataset.negatives import NegativeSamplesGenerator


class TestNegativeSamplesGenerator(unittest.TestCase):
    def test_delete_random_char(self):
        for _ in range(1000):
            self.assertEqual('', NegativeSamplesGenerator.delete_random_char(''))
            self.assertEqual('a', NegativeSamplesGenerator.delete_random_char('a'))
            w = NegativeSamplesGenerator.delete_random_char('ab')
            self.assertTrue(w == 'a' or w == 'b')
            w = NegativeSamplesGenerator.delete_random_char('abc')
            self.assertTrue(w == 'bc' or w == 'ac' or w == 'ab')
            w = NegativeSamplesGenerator.delete_random_char('abcd')
            self.assertEqual(3, len(w))

    def test_insert_random_char(self):
        for _ in range(1000):
            self.assertEqual('', NegativeSamplesGenerator.insert_random_char('', []))
            self.assertEqual('a', NegativeSamplesGenerator.insert_random_char('', ['a']))
            w = NegativeSamplesGenerator.insert_random_char('', ['a', 'b'])
            self.assertTrue(w == 'a' or w == 'b')
            w = NegativeSamplesGenerator.insert_random_char('a', ['b'])
            self.assertTrue(w == 'ab' or w == 'ba')
            w = NegativeSamplesGenerator.insert_random_char('abb', ['a', 'b'])
            self.assertEqual(4, len(w))

    def test_replace_random_char(self):
        for _ in range(1000):
            self.assertEqual('', NegativeSamplesGenerator.replace_random_char('', ['a']))
            self.assertEqual('a', NegativeSamplesGenerator.replace_random_char('a', ['a']))
            self.assertEqual('a', NegativeSamplesGenerator.replace_random_char('a', []))
            self.assertEqual('b', NegativeSamplesGenerator.replace_random_char('a', ['a', 'b']))
            w = NegativeSamplesGenerator.replace_random_char('a', ['a', 'b', 'c'])
            self.assertTrue(w == 'c' or w == 'b')
            w = NegativeSamplesGenerator.replace_random_char('abb', ['a', 'b'])
            self.assertEqual(3, len(w))
            w = NegativeSamplesGenerator.replace_random_char('abc', ['d', 'e'])
            self.assertEqual(3, len(w))

    def test_random_transposition(self):
        for _ in range(1000):
            self.assertEqual('', NegativeSamplesGenerator.random_transposition(''))
            self.assertEqual('a', NegativeSamplesGenerator.random_transposition('a'))
            self.assertEqual('ba', NegativeSamplesGenerator.random_transposition('ab'))
            w = NegativeSamplesGenerator.random_transposition('abc')
            self.assertTrue(w == 'bac' or w == 'acb')

    def test_generate(self):
        test_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
        with open(os.path.join(test_data_dir, 'numeral.dat')) as f:
            positive_samples = {w.strip().lower() for w in f.readlines()}
            alphabet = list({c for w in positive_samples for c in w})

            for _ in range(1000):
                negative_samples = NegativeSamplesGenerator.generate(positive_samples, alphabet)
                self.assertEqual(set(), positive_samples & negative_samples)
                self.assertEqual(len(positive_samples), len(negative_samples))


if __name__ == '__main__':
    unittest.main()
