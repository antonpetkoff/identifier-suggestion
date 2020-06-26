import os
import time

from rouge_score import rouge_scorer

class Score:
    def __init__(self):
        self.score = {
            'rouge_1_p':  0.0,
            'rouge_1_r':  0.0,
            'rouge_1_f1': 0.0,
            'rouge_2_p':  0.0,
            'rouge_2_r':  0.0,
            'rouge_2_f1': 0.0,
            'rouge_3_p':  0.0,
            'rouge_3_r':  0.0,
            'rouge_3_f1': 0.0,
            'rouge_L_p':  0.0,
            'rouge_L_r':  0.0,
            'rouge_L_f1': 0.0,
        }

    def add(self, scores):
        self.score['rouge_1_p']  += scores['rouge1'].precision
        self.score['rouge_1_r']  += scores['rouge1'].recall
        self.score['rouge_1_f1'] += scores['rouge1'].fmeasure
        self.score['rouge_2_p']  += scores['rouge2'].precision
        self.score['rouge_2_r']  += scores['rouge2'].recall
        self.score['rouge_2_f1'] += scores['rouge2'].fmeasure
        self.score['rouge_3_p']  += scores['rouge3'].precision
        self.score['rouge_3_r']  += scores['rouge3'].recall
        self.score['rouge_3_f1'] += scores['rouge3'].fmeasure
        self.score['rouge_L_p']  += scores['rougeL'].precision
        self.score['rouge_L_r']  += scores['rougeL'].recall
        self.score['rouge_L_f1'] += scores['rougeL'].fmeasure


class RougeEvaluator:
    def __init__(self, sequence_transform_fn, batch_size):
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rouge3', 'rougeL'],
            use_stemmer=True,
        )
        self.sequence_transform_fn = sequence_transform_fn
        self.batch_size = batch_size
        self.use_cache = False
        self.target_batches = []
        self.prediction_batches = []


    def enable_cache(self):
        self.use_cache = True


    def disable_cache(self):
        self.use_cache = False


    def reset_state(self):
        self.prediction_batches = []
        if not self.use_cache:
            self.target_batches = []


    def add_batch(self, prediction_batch, target_batch):
        self.prediction_batches.append(prediction_batch)

        if not self.use_cache:
            self.target_batches.append(target_batch)


    # the single argument is a tuple of the predictions and targets
    # this makes it easier for parallel processing
    def evaluate_batch(self, batches):
        batch_id, (predictions, targets) = batches
        batch_score = Score()

        start_time = time.time()

        # accumulate method names for ROUGE evaluation
        for i in range(self.batch_size):
            # transform the raw predictions into strings of words suitable for ROUGE evaluation
            predicted_method_name = self.sequence_transform_fn(predictions[i].numpy())

            # don't forget to transform the target to text, too
            reference_method_name = self.sequence_transform_fn(
                # ignore the first token which is the start of sequence marker
                targets[i, 1:].numpy(),
            )

            # an example of the scores returned by rouge_score is:
            # {'rouge1': Score(precision=0.5, recall=0.44, fmeasure=0.47),
            #  'rouge2': Score(precision=0.29, recall=0.25, fmeasure=0.27),
            #  'rouge3': Score(precision=0.17, recall=0.14, fmeasure=0.15),
            #  'rougeL': Score(precision=0.5, recall=0.45, fmeasure=0.47)}
            scores = self.scorer.score(
                reference_method_name,
                predicted_method_name,
            )

            batch_score.add(scores)

        if (batch_id % 100) == 0:
            print(f'evaluation of batch {batch_id} took: {time.time() - start_time}')

        return batch_score


    def average_scores(self, scores):
        avg_scores = Score()
        total_length = len(scores) * self.batch_size

        # sum together all scores per score_type (e.g. rouge_3_f1, rouge_L_p, etc.)
        for score in scores:
            for score_type in score.score.keys():
                avg_scores.score[score_type] += score.score[score_type]

        # TODO: spare the .score redundancy
        for score_type in score.score.keys():
            avg_scores.score[score_type] /= total_length

        return avg_scores


    def evaluate(self):
        if len(self.prediction_batches) != len(self.target_batches):
            raise ValueError('The number of prediction and target batches must match')

        scores = list(map(
            self.evaluate_batch,
            enumerate(zip(self.prediction_batches, self.target_batches)),
        ))

        return self.average_scores(scores)
