class Measures:

    @staticmethod
    def accuracy(predictions, labels):
        if len(predictions) != len(labels):
            raise ValueError("predictions and labels lengths must be equal")

        if len(labels) == 0:
            return 0, 0, 0, 0, 1, 1, 1, 1

        tp, tn, fp, fn = 0, 0, 0, 0
        for prediction, label in zip(predictions, labels):
            if label == 1:
                if prediction == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if prediction == 1:
                    fp += 1
                else:
                    tn += 1

        if tp == 0:
            if fn == 0 and fp == 0:
                pr, r, f1 = 1, 1, 1
            else:
                pr, r, f1 = 0, 0, 0
        else:
            pr = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * ((pr * r) / (pr + r))

        acc = (tp + tn) / (tp + tn + fp + fn)

        return tp, tn, fp, fn, pr, r, f1, acc
