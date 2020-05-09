import tensorflow as tf

from collections import Counter


class F1Score(tf.metrics.Metric):
    # TODO: add micro / macro averaging as a parameter
    def __init__(
        self,
        from_logits=False,
        name='f1_score',
        **kwargs
    ):
        super(F1Score, self).__init__(name=name, **kwargs)

        self.from_logits = from_logits

        self.tp = 0
        self.fp = 0
        self.fn = 0

        self.true_positives = self.add_weight(name='tp', initializer='zeros')


    def update_state(self, y_true, y_pred, sample_weight=None):
        # TODO: should we ignore padding tokens?
        # depadded_y_true = tf.RaggedTensor.from_tensor(y_true, padding=0)

        def update_confusion_matrix_counts(y_true, y_pred):
            # TODO: optimize with batching + tf operations on the GPU
            for target_seq, predicted_seq in zip(y_true.numpy(), y_pred.numpy()):
                target_counts = Counter(target_seq)
                predicted_counts = Counter(predicted_seq)

                # hits: count all tokens both inside 'predicted' and 'target'
                true_positives = sum((target_counts & predicted_counts).values())

                # false alarms: count all tokens inside 'predicted', but missing in 'target'
                false_positives = sum((predicted_counts - target_counts).values())

                # misses: count all tokens inside 'target', but missing in 'predicted'
                false_negatives = sum((target_counts - predicted_counts).values())

                self.tp += true_positives
                self.fp += false_positives
                self.fn += false_negatives

            return 0 # TODO: we artificially return a number to match Tout in tf.py_function

        targets = y_true
        predictions = tf.argmax(y_pred, axis=-1) if self.from_logits else y_pred

        tf.py_function(
            update_confusion_matrix_counts,
            inp=[targets, predictions],
            Tout=tf.int32
        )


    def result(self):
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0

        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return f1, precision, recall


    def reset_states(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

