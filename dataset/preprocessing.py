import os
import pickle
import random

import numpy

from .negatives import NegativeSamplesGenerator


class Preprocessing:

    @staticmethod
    def preprocessing(dictionary, output_dir):
        with open(dictionary) as f:
            positive_samples = {w.strip().lower() for w in f.readlines()}

            print('Positive samples size {0:,}'.format(len(positive_samples)))

            max_word_length = max([len(w) for w in positive_samples])
            print(f'Maximum word length from positive samples {max_word_length}')

            alphabet = {c for w in positive_samples for c in w}
            alphabet = list(alphabet)
            alphabet.sort()

            print(f'Alphabet size {len(alphabet)}')

            alphabet = {c: i + 1 for (i, c) in enumerate(alphabet)}

            with open(os.path.join(output_dir, 'alphabet.dict'), 'wb') as file:
                pickle.dump(alphabet, file)

            negative_samples = NegativeSamplesGenerator.generate(positive_samples, list(alphabet.keys()))
            print(f'Generated negative samples {len(negative_samples)}')

            train, dev, test = Preprocessing.shuffle_and_split_data(list(positive_samples) + list(negative_samples))

            print('Train size {0:,}'.format(len(train)))
            print('Dev size {0:,}'.format(len(dev)))
            print('Test size {0:,}'.format(len(test)))

            max_word_length = max([len(w) for w in train] + [len(w) for w in dev])
            print(f'Maximum word length from train and dev splits {max_word_length}')

            Preprocessing.pad_and_serialize_data(
                train, 'train', positive_samples, alphabet, max_word_length, output_dir
            )
            Preprocessing.pad_and_serialize_data(
                dev, 'dev', positive_samples, alphabet, max_word_length, output_dir
            )
            Preprocessing.pad_and_serialize_data(
                test, 'test', positive_samples, alphabet, max_word_length, output_dir
            )

    @staticmethod
    def shuffle_and_split_data(data, train_size=0.8, dev_size=0.1):
        random.shuffle(data)

        train = data[:round(len(data) * train_size)]
        dev = data[round(len(data) * train_size):round(len(data) * (train_size + dev_size))]
        test = data[round(len(data) * (train_size + dev_size)):]

        return train, dev, test

    @staticmethod
    def pad_and_serialize_data(split, split_name, positive_samples, alphabet, max_word_length, output_dir):
        x, length, y = [], [], []

        for w in split:
            len_w = len(w)
            w_x = [alphabet[c] for c in w]
            if len(w_x) < max_word_length:
                w_x.extend([0] * (max_word_length - len(w_x)))
            elif len(w_x) > max_word_length:
                w_x = w_x[:max_word_length]
                len_w = max_word_length
            x.append(w_x)
            length.append(len_w)
            y.append(1 if w in positive_samples else 0)

        numpy.save(os.path.join(output_dir, split_name + '.data.npy'), numpy.array(x))
        numpy.save(os.path.join(output_dir, split_name + '.length.npy'), numpy.array(length))
        numpy.save(os.path.join(output_dir, split_name + '.labels.npy'), numpy.array(y))
