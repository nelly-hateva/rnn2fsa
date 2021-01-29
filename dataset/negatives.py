import itertools
import random


class NegativeSamplesGenerator:
    @staticmethod
    def delete_random_char(s):
        if len(s) >= 2:
            pos = random.randint(0, len(s) - 1)
            return s[:pos] + s[pos + 1:]
        return s

    @staticmethod
    def insert_random_char(s, alphabet):
        if alphabet:
            pos = random.randint(0, len(s))
            return s[:pos] + random.choice(alphabet) + s[pos:]
        return s

    @staticmethod
    def replace_random_char(s, alphabet):
        if len(s) >= 1:
            pos = random.randint(0, len(s) - 1)
            choices = alphabet.copy()
            if s[pos] in choices:
                choices.remove(s[pos])
            if choices:
                return s[:pos] + random.choice(choices) + s[pos + 1:]
        return s

    @staticmethod
    def random_transposition(s):
        if len(s) >= 2:
            pos = random.randint(0, len(s) - 2)
            return s[:pos] + s[pos + 1] + s[pos] + s[pos + 2:]
        return s

    @staticmethod
    def generate(positive_samples, alphabet):
        negative_samples = set()

        for w in itertools.cycle(positive_samples):

            if len(negative_samples) == len(positive_samples):
                break

            if len(w) == 0:
                w1 = NegativeSamplesGenerator.insert_random_char(w, alphabet)
            elif len(w) == 1:
                if bool(random.getrandbits(1)):
                    w1 = NegativeSamplesGenerator.replace_random_char(w, alphabet)
                else:
                    w1 = NegativeSamplesGenerator.insert_random_char(w, alphabet)
            else:
                op = random.randint(1, 4)
                if op == 1:
                    w1 = NegativeSamplesGenerator.replace_random_char(w, alphabet)
                elif op == 2:
                    w1 = NegativeSamplesGenerator.insert_random_char(w, alphabet)
                elif op == 3:
                    w1 = NegativeSamplesGenerator.delete_random_char(w)
                else:
                    w1 = NegativeSamplesGenerator.random_transposition(w)

            if w1 not in positive_samples and w1 not in negative_samples:
                negative_samples.add(w1)

        return negative_samples
