import os

import numpy


class Stats:

    @staticmethod
    def print(data_dir):
        Stats.word_length_stats(data_dir)

    @staticmethod
    def word_length_stats(data_dir):
        def print_max_min_and_avg(name, data):
            print('{} samples minimum word length {}'.format(name, data.min()))
            print('{} samples maximum word length {}'.format(name, data.max()))
            print('{} samples mean word length {:.2f}'.format(name, numpy.mean(data)))
            print('{} samples variance word length {:.2f}'.format(name, numpy.var(data)))
            print('{} samples standard deviation word length {:.2f}'.format(name, numpy.std(data)))

        train_length = numpy.load(os.path.join(data_dir, 'train.length.npy'), allow_pickle=True)
        dev_length = numpy.load(os.path.join(data_dir, 'dev.length.npy'), allow_pickle=True)
        test_length = numpy.load(os.path.join(data_dir, 'test.length.npy'), allow_pickle=True)

        train_labels = numpy.load(os.path.join(data_dir, 'train.labels.npy'), allow_pickle=True)
        dev_labels = numpy.load(os.path.join(data_dir, 'dev.labels.npy'), allow_pickle=True)
        test_labels = numpy.load(os.path.join(data_dir, 'test.labels.npy'), allow_pickle=True)

        positive_samples_length = numpy.concatenate((
            train_length[train_labels.nonzero()], dev_length[dev_labels.nonzero()], test_length[test_labels.nonzero()]
        ))
        print_max_min_and_avg('Positive', positive_samples_length)

        negative_samples_length = numpy.concatenate((
            train_length[numpy.where(train_labels == 0)[0]],
            dev_length[numpy.where(dev_labels == 0)[0]],
            test_length[numpy.where(test_labels == 0)[0]]
        ))
        print_max_min_and_avg('Negative', negative_samples_length)


if __name__ == '__main__':
    Stats.print(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'), 'words'))
