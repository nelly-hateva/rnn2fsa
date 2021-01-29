import os
import pickle
import tempfile
import unittest

import numpy

from dataset.preprocessing import Preprocessing


class TestPreprocessing(unittest.TestCase):

    def test_shuffle_and_split_data(self):
        for _ in range(1000):
            train, dev, test = Preprocessing.shuffle_and_split_data(list(range(23)))
            self.assertEqual(18, len(train))
            self.assertEqual(len(train), len(set(train)))
            self.assertEqual(3, len(dev))
            self.assertEqual(len(dev), len(set(dev)))
            self.assertEqual(2, len(test))
            self.assertEqual(len(test), len(set(test)))
            self.assertEqual(set(), set(train) & set(dev))
            self.assertEqual(set(), set(train) & set(test))
            self.assertEqual(set(), set(dev) & set(test))

            train, dev, test = Preprocessing.shuffle_and_split_data(list(range(23)), train_size=0.5)
            self.assertEqual(12, len(train))
            self.assertEqual(len(train), len(set(train)))
            self.assertEqual(2, len(dev))
            self.assertEqual(len(dev), len(set(dev)))
            self.assertEqual(9, len(test))
            self.assertEqual(len(test), len(set(test)))
            self.assertEqual(set(), set(train) & set(dev))
            self.assertEqual(set(), set(train) & set(test))
            self.assertEqual(set(), set(dev) & set(test))

            train, dev, test = Preprocessing.shuffle_and_split_data(list(range(23)), train_size=0.6, dev_size=0.2)
            self.assertEqual(14, len(train))
            self.assertEqual(len(train), len(set(train)))
            self.assertEqual(4, len(dev))
            self.assertEqual(len(dev), len(set(dev)))
            self.assertEqual(5, len(test))
            self.assertEqual(len(test), len(set(test)))
            self.assertEqual(set(), set(train) & set(dev))
            self.assertEqual(set(), set(train) & set(test))
            self.assertEqual(set(), set(dev) & set(test))

    def test_pad_and_serialize_data(self):
        split = ['aaa', 'bb', 'c']
        positive_samples = {'aaa', 'c'}
        alphabet = {'a': 1, 'b': 2, 'c': 3}

        with tempfile.TemporaryDirectory() as tmp_dir:
            for max_word_length in range(1, 7):
                Preprocessing.pad_and_serialize_data(
                    split, 'split', positive_samples, alphabet, max_word_length, tmp_dir
                )

                split_data = numpy.load(os.path.join(tmp_dir, 'split.data.npy'), allow_pickle=True)
                expected_data = [[alphabet[c] for c in w] for w in split]
                expected_data = [w[:max_word_length] for w in expected_data]
                expected_data = [w + [0] * (max_word_length - len(w)) for w in expected_data]
                self.assertTrue((expected_data == split_data).all())
                split_length = numpy.load(os.path.join(tmp_dir, 'split.length.npy'), allow_pickle=True)
                expected_length = [min(3, max_word_length), min(2, max_word_length), min(1, max_word_length)]
                self.assertTrue((expected_length == split_length).all())
                split_labels = numpy.load(os.path.join(tmp_dir, 'split.labels.npy'), allow_pickle=True)
                self.assertTrue(([1, 0, 1] == split_labels).all())

    def test_preprocessing(self):

        test_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
        numeral_data = os.path.join(test_data_dir, 'numeral.dat')

        with tempfile.TemporaryDirectory() as tmp_dir:

            Preprocessing.preprocessing(numeral_data, tmp_dir)

            with open(os.path.join(tmp_dir, 'alphabet.dict'), 'rb') as f:
                alphabet = pickle.load(f)

                self.assertEqual(23, len(alphabet))
                self.assertEqual(list(range(1, 24)), list(alphabet.values()))

            train_data = numpy.load(os.path.join(tmp_dir, 'train.data.npy'), allow_pickle=True)
            train_length = numpy.load(os.path.join(tmp_dir, 'train.length.npy'), allow_pickle=True)
            train_labels = numpy.load(os.path.join(tmp_dir, 'train.labels.npy'), allow_pickle=True)
            dev_data = numpy.load(os.path.join(tmp_dir, 'dev.data.npy'), allow_pickle=True)
            dev_length = numpy.load(os.path.join(tmp_dir, 'dev.length.npy'), allow_pickle=True)
            dev_labels = numpy.load(os.path.join(tmp_dir, 'dev.labels.npy'), allow_pickle=True)
            test_data = numpy.load(os.path.join(tmp_dir, 'test.data.npy'), allow_pickle=True)
            test_length = numpy.load(os.path.join(tmp_dir, 'test.length.npy'), allow_pickle=True)
            test_labels = numpy.load(os.path.join(tmp_dir, 'test.labels.npy'), allow_pickle=True)

            with open(numeral_data) as f:
                positive_samples = {w.strip().lower() for w in f.readlines()}

            self.assertEqual(2 * len(positive_samples), len(train_data) + len(dev_data) + len(test_data))
            self.assertEqual(2 * len(positive_samples), len(train_length) + len(dev_length) + len(test_length))
            self.assertEqual(2 * len(positive_samples), len(train_labels) + len(dev_labels) + len(test_labels))

            self.assertEqual(
                len(positive_samples),
                numpy.count_nonzero(train_labels) + numpy.count_nonzero(dev_labels) + numpy.count_nonzero(test_labels)
            )
            self.assertEqual(
                len(positive_samples),
                len(numpy.where(train_labels == 0)[0])
                + len(numpy.where(dev_labels == 0)[0])
                + len(numpy.where(test_labels == 0)[0])
            )

            self.assertEqual(
                sum([len(w) for w in positive_samples]),
                train_length[train_labels.nonzero()].sum() + dev_length[dev_labels.nonzero()].sum()
                + test_length[test_labels.nonzero()].sum()
            )

            inv_alphabet = {v: k for k, v in alphabet.items()}

            def numpy_data_to_list(inv_alphabet_, numpy_data):
                samples = []
                for s in numpy_data.tolist():
                    try:
                        s = s[:s.index(0)]
                    except ValueError:
                        pass
                    samples.append(''.join([inv_alphabet_[c] for c in s]))
                return samples

            train_samples = numpy_data_to_list(inv_alphabet, train_data)
            self.assertEqual(len(train_samples), len(set(train_samples)))
            dev_samples = numpy_data_to_list(inv_alphabet, dev_data)
            self.assertEqual(len(dev_samples), len(set(dev_samples)))
            test_samples = numpy_data_to_list(inv_alphabet, test_data)
            self.assertEqual(len(test_samples), len(set(test_samples)))

            self.assertEqual(set(), set(train_samples) & set(dev_samples))
            self.assertEqual(set(), set(train_samples) & set(test_samples))
            self.assertEqual(set(), set(dev_samples) & set(test_samples))

            positive_samples_list = numpy.concatenate((
                train_data[train_labels.nonzero()],
                dev_data[dev_labels.nonzero()],
                test_data[test_labels.nonzero()]
            ), axis=0).tolist()
            self.assertEqual(len(positive_samples), len(positive_samples_list))

            for sample in positive_samples_list:
                try:
                    sample = sample[:sample.index(0)]
                except ValueError:
                    pass
                sample = ''.join([inv_alphabet[c] for c in sample])
                self.assertTrue(sample in positive_samples)

            negative_samples_list = numpy.concatenate((
                train_data[numpy.where(train_labels == 0)[0]],
                dev_data[numpy.where(dev_labels == 0)[0]],
                test_data[numpy.where(test_labels == 0)[0]]
            ), axis=0).tolist()
            self.assertEqual(len(positive_samples), len(negative_samples_list))

            for sample in negative_samples_list:
                try:
                    sample = sample[:sample.index(0)]
                except ValueError:
                    pass
                sample = ''.join([inv_alphabet[c] for c in sample])
                self.assertFalse(sample in positive_samples)


if __name__ == '__main__':
    unittest.main()
