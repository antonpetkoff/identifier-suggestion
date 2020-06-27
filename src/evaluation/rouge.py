import os
import time

from rouge_score import rouge_scorer

ROUGE_VARIATIONS = ['1', '2', '3', 'L']

class Score:
    def __init__(self):
        self.score = {}
        for i in ROUGE_VARIATIONS:
            self.score[f'rouge_{i}_p']  = 0.0
            self.score[f'rouge_{i}_r']  = 0.0
            self.score[f'rouge_{i}_f1'] = 0.0


    def add(self, scores):
        # iterate over each ROUGE variation that is evaluated
        for i in ROUGE_VARIATIONS:
            self.score[f'rouge_{i}_p']  += scores[f'rouge{i}'].precision
            self.score[f'rouge_{i}_r']  += scores[f'rouge{i}'].recall
            self.score[f'rouge_{i}_f1'] += scores[f'rouge{i}'].fmeasure


class RougeEvaluator:
    def __init__(self, sequence_transform_fn, batch_size):
        self.scorer = rouge_scorer.RougeScorer(
            [f'rouge{i}' for i in ROUGE_VARIATIONS],
            use_stemmer=True,
        )
        self.sequence_transform_fn = sequence_transform_fn
        self.batch_size = batch_size
        self.target_batches = []
        self.prediction_batches = []


    def reset_state(self):
        self.prediction_batches = []
        self.target_batches = []


    def add_batch(self, prediction_batch, target_batch):
        self.prediction_batches.append(prediction_batch)
        self.target_batches.append(target_batch)


    # the single argument is a tuple of the predictions and targets
    # this makes it easier for parallel processing
    def evaluate_batch(self, batches):
        batch_id, (predictions, targets) = batches
        batch_score = Score()

        start_time = time.time()

        # transform the raw predictions into strings of words suitable for ROUGE evaluation
        predicted_method_names = list(map(
            self.sequence_transform_fn,
            predictions.numpy(),
        ))

        # don't forget to transform the target to text, too
        reference_method_names = list(map(
            # ignore the first token which is the start of sequence marker
            lambda target: self.sequence_transform_fn(target[1:]),
            targets.numpy()
        ))

        # accumulate method names for ROUGE evaluation
        for i in range(self.batch_size):
            # an example of the scores returned by rouge_score is:
            # {'rouge1': Score(precision=0.5, recall=0.44, fmeasure=0.47),
            #  'rouge2': Score(precision=0.29, recall=0.25, fmeasure=0.27),
            #  'rouge3': Score(precision=0.17, recall=0.14, fmeasure=0.15),
            #  'rougeL': Score(precision=0.5, recall=0.45, fmeasure=0.47)}
            scores = self.scorer.score(
                reference_method_names[i],
                predicted_method_names[i],
            )
            batch_score.add(scores)

        if batch_id % 50 == 0:
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
