import tensorflow as tf
import numpy as np

from collections import Counter


class F1Score(tf.metrics.Metric):
    # TODO: add micro / macro averaging as a parameter
    def __init__(
        self,
        # num_classes, TODO: pass the number of classes?
        from_logits=False,
        averaging='micro', # can be 'micro' or 'macro'
        name='f1_score',
        **kwargs
    ):
        super(F1Score, self).__init__(name=name, **kwargs)

        self.from_logits = from_logits
        self.averaging = averaging

        if self.averaging != 'micro' and self.averaging != 'macro':
            raise ValueError('averaging must be "micro" or "macro"')

        self.reset_states()


    def update_state(self, y_true, y_pred, sample_weight=None):
        def update_confusion_matrix_counts(y_true, y_pred):
            # TODO: optimize with batching + tf operations on the GPU
            for target_seq, predicted_seq in zip(y_true.numpy(), y_pred.numpy()):
                target_counts = Counter(target_seq)
                predicted_counts = Counter(predicted_seq)

                # hits: count all tokens both inside 'predicted' and 'target'
                hits = target_counts & predicted_counts

                # false alarms: count all tokens inside 'predicted', but missing in 'target'
                false_alarms = predicted_counts - target_counts

                # misses: count all tokens inside 'target', but missing in 'predicted'
                misses = target_counts - predicted_counts

                # increment counters
                self.tp = self.tp + hits
                self.fp = self.fp + false_alarms
                self.fn = self.fn + misses

            return 0 # TODO: we artificially return a number to match Tout in tf.py_function

        # TODO: should we ignore padding tokens?
        # depadded_y_true = tf.RaggedTensor.from_tensor(y_true, padding=0)

        targets = y_true
        predictions = tf.argmax(y_pred, axis=-1) if self.from_logits else y_pred

        tf.py_function(
            update_confusion_matrix_counts,
            inp=[targets, predictions],
            Tout=tf.int32
        )


    @staticmethod
    def compute_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return f1, precision, recall


    def result(self):
        if self.averaging == 'micro':
            # the averaging is global
            f1, precision, recall = F1Score.compute_metrics(
                sum(self.tp.values()),
                sum(self.fp.values()),
                sum(self.fn.values()),
            )
        elif self.averaging == 'macro':
            # TODO: should we be computing metrics_per_class for all classes, regardless if they occurred at all? what about division by zero?
            # get all class ids for which there are any counts
            class_ids = list(
                set(self.tp.keys()).union(set(self.fp.keys())).union(set(self.fn.keys()))
            )

            # each row represents a class
            # every row has 3 columns with the computed metrics for the class (f1, precision, recall)
            metrics_per_class = np.asarray(list(map(
                lambda class_id: np.asarray(F1Score.compute_metrics(self.tp[class_id], self.fp[class_id], self.fn[class_id])),
                class_ids
            )))

            # compute mean by columns, i.e. the mean of f1 score for all classes, etc.
            f1, precision, recall = np.mean(metrics_per_class, axis = 0)

        # TODO: add support for weighted averaging

        return f1, precision, recall


    def reset_states(self):
        self.tp = Counter()
        self.fp = Counter()
        self.fn = Counter()

