import tensorflow as tf
import numpy as np

from collections import Counter


# TODO: document
def get_counts_per_class(sequence, num_classes):
    # the sorting is required to ensure that the returned unique values will be in sorted order
    unique_values, _idx, counts = tf.unique_with_counts(tf.sort(sequence))

    print(unique_values, counts)

    sparse_tensor = tf.SparseTensor(
        indices = tf.expand_dims(unique_values, axis = 1),
        values = counts,
        dense_shape = [num_classes]
    )

    return tf.sparse.to_dense(sparse_tensor)


def safe_divide(a, b):
    return tf.math.divide_no_nan(
        tf.cast(a, dtype=tf.float32),
        tf.cast(b, dtype=tf.float32)
    )


class F1Score(tf.metrics.Metric):
    def __init__(
        self,
        num_classes,
        from_logits=False,
        averaging='micro', # can be 'micro' or 'macro'
        name='f1_score',
        **kwargs
    ):
        super(F1Score, self).__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.from_logits = from_logits
        self.averaging = averaging

        assert self.averaging in ['micro', 'macro'], \
            'averaging must be "micro" or "macro"'

        self.init_shape = [self.num_classes]

        # allocate the zeros once for efficiency
        self.zeros = tf.zeros(self.init_shape, dtype=self.dtype)

        def _zero_wt_init(name):
            return self.add_weight(
                name,
                shape=[self.num_classes],
                initializer="zeros",
                dtype=self.dtype
            )

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")


    def non_negative_subtract(self, a, b):
        return tf.math.maximum(
            tf.math.subtract(a, b),
            self.zeros
        )


    def compute_confusion_matrix_tensors(self, target, predicted):
        target_counter = get_counts_per_class(target, num_classes = self.num_classes)
        predicted_counter = get_counts_per_class(predicted, num_classes = self.num_classes)

        # hits: count all tokens both inside 'predicted' and 'target'
        hits = tf.math.minimum(target_counter, predicted_counter)

        # false alarms: count all tokens inside 'predicted', but missing in 'target'
        false_alarms = self.non_negative_subtract(predicted_counter, target_counter)

        # misses: count all tokens inside 'target', but missing in 'predicted'
        misses = self.non_negative_subtract(target_counter, predicted_counter)

        return hits, false_alarms, misses


    def update_state(self, y_true, y_pred, sample_weight=None):
        # TODO: should we ignore padding tokens?
        # depadded_y_true = tf.RaggedTensor.from_tensor(y_true, padding=0)

        predictions = tf.argmax(y_pred, axis=-1) if self.from_logits else y_pred

        # reshape with shape = [-1] flattens the 2D tensors into 1D
        hits, false_alarms, misses = self.compute_confusion_matrix_tensors(
            target=tf.reshape(y_true, shape = [-1]),
            predicted=tf.reshape(predictions, shape = [-1]),
        )

        self.true_positives.assign_add(hits)
        self.false_positives.assign_add(false_alarms)
        self.false_negatives.assign_add(misses)


    @staticmethod
    def compute_metrics(tp, fp, fn):
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        return f1, precision, recall


    def result(self):
        if self.averaging == 'micro':
            # count globally, thus sum the counts of all classes and then compute the metrics
            f1, precision, recall = F1Score.compute_metrics(
                tf.math.reduce_sum(self.true_positives),
                tf.math.reduce_sum(self.false_positives),
                tf.math.reduce_sum(self.false_negatives),
            )
        elif self.averaging == 'macro':
            f1, precision, recall = tf.math.reduce_mean(
                F1Score.compute_metrics(
                    self.true_positives,
                    self.false_positives,
                    self.false_negatives,
                ),
                axis = 1 # compute the mean for each class
            )

        return f1, precision, recall


    def reset_states(self):
        self.true_positives.assign(self.zeros)
        self.false_positives.assign(self.zeros)
        self.false_negatives.assign(self.zeros)
