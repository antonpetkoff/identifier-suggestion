import tensorflow as tf
import numpy as np
import os
import time
import json
import math

from collections import Counter
from itertools import takewhile

from src.common.tokens import Common
from src.models.seq2seq.encoder import Encoder
from src.models.seq2seq.decoder import Decoder
from src.preprocessing.tokens import tokenize_method
from src.evaluation.rouge import RougeEvaluator


def prefix_dict_keys(dict, prefix):
    return {f'{prefix}{key}': value for key, value in dict.items()}


class Seq2Seq(tf.Module):
    def __init__(
        self,
        logger,
        checkpoint_dir,
        input_vocab_index,
        output_vocab_index,
        max_input_seq_length,
        max_output_seq_length,
        input_vocab_size,
        output_vocab_size,
        input_embedding_dim = 256,
        output_embedding_dim = 256,
        rnn_units = 1024,
        batch_size = 64,
        learning_rate = 0.001,
        dropout_rate = 0.0,
        patience = 3,
        min_delta = 0.001,
    ):
        self.logger = logger

        self.params = {
            'checkpoint_dir': checkpoint_dir,
            'max_input_seq_length': max_input_seq_length,
            'max_output_seq_length': max_output_seq_length,
            'input_vocab_size': input_vocab_size,
            'output_vocab_size': output_vocab_size,
            'input_embedding_dim': input_embedding_dim,
            'output_embedding_dim': output_embedding_dim,
            'rnn_units': rnn_units,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'patience': patience,
            'min_delta': min_delta,
        }

        self.input_vocab_index = input_vocab_index
        self.output_vocab_index = output_vocab_index
        self.reverse_input_index = dict(
            (i, token) for token, i in input_vocab_index.items()
        )
        self.reverse_output_index = dict(
            (i, token) for token, i in output_vocab_index.items()
        )
        self.eos_index = self.output_vocab_index[Common.EOS]

        self.encoder = Encoder(
            input_vocab_size=self.params['input_vocab_size'],
            embedding_dims=self.params['input_embedding_dim'],
            rnn_units=self.params['rnn_units'],
            batch_size=self.params['batch_size'],
            dropout_rate=self.params['dropout_rate'],
        )

        self.decoder = Decoder(
            output_vocab_size=self.params['output_vocab_size'],
            embedding_dims=self.params['output_embedding_dim'],
            rnn_units=self.params['rnn_units'],
            batch_size=self.params['batch_size'],
            dropout_rate=self.params['dropout_rate'],
        )

        # TODO: experiment with a custom learning rate scheduler
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.params['learning_rate'],
        )

        seq_transform_fn = lambda seq: self.convert_raw_prediction_to_text(seq, join_token=' ', camel_case=False)

        self.training_evaluator = RougeEvaluator(
            sequence_transform_fn=seq_transform_fn,
            batch_size=self.params['batch_size'],
        )

        self.test_evaluator = RougeEvaluator(
            sequence_transform_fn=seq_transform_fn,
            batch_size=self.params['batch_size'],
        )

        self.checkpoint = tf.train.Checkpoint(
            optimizer = self.optimizer,
            encoder = self.encoder,
            decoder = self.decoder,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            checkpoint_dir,
            max_to_keep = 3,
        )


    def call(
        self,
        inputs,
        encoder_hidden,
        training=False
    ):
        # TODO: differentiate between training and NOT training, if you add a Dropout layer

        input_batch, target_batch = inputs

        # feed forward through encoder
        encoder_outputs, encoder_hidden, _encoder_cell_state = self.encoder(
            input_batch,
            encoder_hidden,
            training,
        )

        # initialize the decoder's hidden state with the final hidden state of the encoder
        decoder_hidden = encoder_hidden

        # TODO: extract the <sos> token in a Common module

        # shape: (batch_size, 1), where the 1 is the single <start> timestep
        decoder_input = tf.expand_dims(
            [self.output_vocab_index[Common.SOS]] * self.params['batch_size'],
            axis = 1 # expand the last dimension, leave the batch_size in tact
        )

        # this list will accumulate the predictions of each timestep during teacher forcing
        # when all timestep predictions are accumulated, they are stacked along the timestep axis
        predictions = []

        target_seq_length = target_batch.shape[1]

        # feed forward through decoder using
        # the teacher forcing training algorithm - feeding the target as the next input
        # we iterate timesteps from 1, not from 0, because this way we shift the target sequences by one
        # i.e. if we have [<sos>, get, search, data, <eos>] as a target sequence
        # we first feed <sos> (index 0) into the decoder and expect it to predict "get" (index 1)
        # and on the next step the "get" (index 1) is fed into the decoder, expecting it to predict "search" (index 2)
        # and so on.
        for t in range(1, target_seq_length): # for each timestep in the output sequence
            # passing encoder_outputs to the decoder for a single timestep
            # notice that we overwrite decoder_hidden on every timestep
            timestep_predictions, decoder_hidden, _attention_weights = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
                training,
            )

            y_true = target_batch[:, t]

            predictions.append(timestep_predictions)

            # using teacher forcing
            # pass the true target/output token from timestep t as input to the decoder for timestep t+1
            decoder_input = tf.expand_dims(y_true, 1)

        # stack the predictions for all timesteps along the the timestep axis
        # after tf.stack the shape becomes [batch_size, output_seq_length - 1, 1, output_vocab_size]
        combined_predictions = tf.stack(predictions, axis = 1) # axis = 1 is the timestep axis

        # TODO: return attention_weights?

        return combined_predictions # [batch_size, output_seq_length - 1, output_vocab_size]


    def get_config(self):
        return self.params


    def build(self):
        self.encoder.build(input_shape=(self.params['batch_size'], self.params['max_input_seq_length']))

        self.decoder.set_state_from_encoder(
            hidden_state = tf.zeros((self.params['batch_size'], self.params['rnn_units'])),
            encoder_outputs = tf.zeros((
                self.params['batch_size'],
                self.params['max_input_seq_length'],
                self.params['rnn_units']
            ))
        )

        self.decoder.build(input_shape=(self.params['batch_size'], 1)) # the decoder processes one token at a time


    def summary(self):
        # TODO: build only if the models are not built
        self.build()
        self.encoder.summary()
        self.decoder.summary()


    def save_checkpoint(self):
        self.logger.log_message('Saving checkpoint...')

        save_path = self.checkpoint_manager.save()

        self.logger.persist_data(save_path)

        self.logger.log_message(f'Saved checkpoint: {save_path}')

        return save_path


    def save(self):
        self.save_checkpoint()

        config_filename = os.path.join(self.params['checkpoint_dir'], 'config.json')
        with open(config_filename, 'w') as f:
            json.dump(self.params, f)

        self.logger.persist_data(config_filename)
        self.logger.log_message('Done saving model')


    def restore_latest_checkpoint(self):
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint

        self.checkpoint.restore(latest_checkpoint)

        if latest_checkpoint:
            self.logger.log_message("Restored from {}".format(latest_checkpoint))
        else:
            self.logger.log_message("Initializing from scratch.")


    @staticmethod
    def restore(checkpoint_dir, logger, input_vocab_index, output_vocab_index):
        logger.log_message('Restoring model config')

        with open(os.path.join(checkpoint_dir, 'config.json')) as f:
            config = json.load(f)

        logger.log_message('Loaded model config: ', config)

        model = Seq2Seq(
            logger = logger,
            checkpoint_dir = checkpoint_dir,
            input_vocab_index = input_vocab_index,
            output_vocab_index = output_vocab_index,
            max_input_seq_length = config['max_input_seq_length'],
            max_output_seq_length = config['max_output_seq_length'],
            input_vocab_size = config['input_vocab_size'],
            output_vocab_size = config['output_vocab_size'],
            input_embedding_dim = config['input_embedding_dim'],
            output_embedding_dim = config['output_embedding_dim'],
            rnn_units = config['rnn_units'],
            batch_size = config['batch_size'],
            learning_rate = config.get('learning_rate', 0.001),
            dropout_rate = config.get('dropout_rate', 0.0),
            patience = config.get('patience', 3),
            min_delta = config.get('min_delta', 0.001),
        )

        model.restore_latest_checkpoint()

        model.build() # it is necessary to build the model before accessing its variables

        logger.log_message('Done restoring model')

        return model


    def loss_function(
        self,
        y_true, # shape: [batch_size, max_output_seq_length]
        y_pred, # shape: [batch_size, max_output_seq_length, output_vocab_size]
    ):
        # the CategoricalCrossentropy expects one-hot encoded y_true labels
        # that's why we use the Sparse version, because we don't need to one-hot encode the sequences
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, # y_pred will contain predictions for each class before you would pass it through softmax
            reduction='none' # do not sum or reduce the computed loss values in any way
        )

        # compute the actual losses
        loss = loss_fn(y_true=y_true, y_pred=y_pred)

        # skip loss calculation for padding sequences which contain only zeroes (i.e. when y = 0)
        # mask the loss when padding sequence appears in the output sequence
        # e.g.
        # [ <start>,transform, search, response, data, 0, 0, 0, 0, ..., <end>]
        # [ 1      , 234     , 3234  , 423     , 3344, 0, 0, 0 ,0, ..., 2 ]
        mask = tf.logical_not(tf.math.equal(y_true, 0)) # output 0 when y = 0, otherwise output 1
        mask = tf.cast(mask, dtype=loss.dtype)

        loss = mask * loss

        # all dimensions are reduced
        loss = tf.reduce_mean(loss)

        return loss


    def train_step(
        self,
        input_batch, # (batch_size, max_input_seq_length)
        target_batch, # (batch_size, max_output_seq_length)
        encoder_hidden, # (batch_size, encoder_rnn_units) TODO: do we need to pass this?
    ):
        # ignore the first timestep, because it is always the start of sequence token
        y_true = target_batch[:, 1:]

        loss = 0.0

        with tf.GradientTape() as tape:
            predictions = self.call(
                inputs = [input_batch, target_batch],
                encoder_hidden = encoder_hidden,
                training = True,
            )

            # compute loss
            loss = self.loss_function(y_true, predictions)

        # get the list of all trainable weights (variables)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        # differentiate loss with regard to variables (the parameters of the model)
        gradients = tape.gradient(loss, variables)

        # combine into pairs each variable with its gradient
        grads_and_vars = zip(gradients, variables)

        # adjust the weights / parameters with the computed gradients
        self.optimizer.apply_gradients(grads_and_vars)

        # select the most probable tokens from logits
        predictions = tf.argmax(predictions, axis=-1)

        return loss, predictions


    def train(self, X_train, Y_train, X_test, Y_test, epochs, on_epoch_end):
        self.restore_latest_checkpoint()

        num_samples = len(X_train)
        batch_size = self.params['batch_size']
        steps_per_epoch = num_samples // batch_size
        BUFFER_SIZE=5000 # TODO: expose as a parameter

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

        epochs_without_improvement = 0
        best_rouge_l_f1 = 0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0.0

            encoder_hidden_state = self.encoder.initialize_hidden_state()

            self.training_evaluator.reset_state()
            self.test_evaluator.reset_state()

            for (step, (input_batch, target_batch)) in enumerate(train_dataset.take(steps_per_epoch)):
                batch_loss, batch_predictions = self.train_step(
                    # the models expect tf.float32
                    tf.cast(input_batch, dtype=tf.float32),
                    tf.cast(target_batch, dtype=tf.float32),
                    encoder_hidden = encoder_hidden_state,
                )

                self.training_evaluator.add_batch(batch_predictions, target_batch)

                total_loss += batch_loss

                self.logger.log_data({
                    'batch': step,
                    'loss': round(batch_loss.numpy(), 3),
                })

                if (step + 1) % 10 == 0:
                    # TODO: just log the dict above? also format the numbers to 2 decimal places?
                    self.logger.log_message(
                        f'epoch {epoch} - batch {step + 1} - loss {batch_loss}'
                    )

            self.logger.log_message(f'epoch {epoch} training time: {time.time() - start_time} sec')

            start_time = time.time()
            train_rouge_scores = self.training_evaluator.evaluate()
            self.logger.log_message(f'epoch {epoch} evaluation on training data time: {time.time() - start_time} sec')

            self.logger.log_data({
                'epoch': epoch,
                'epoch_loss': total_loss / steps_per_epoch,
                **train_rouge_scores.score,
            })

            test_results = self.evaluate(X_test, Y_test, batch_size, epoch)

            on_epoch_end(epoch)

            # TODO: extract in on_epoch_end()?
            # save the whole model on every 3rd epoch
            if epoch % 3 == 0:
                save_path = self.checkpoint_manager.save()
                self.logger.log_message("epoch {} saved checkpoint: {}".format(epoch, save_path))

            # early stopping
            if test_results['test_rouge_L_f1'] > best_rouge_l_f1 + self.params['min_delta']:
                best_rouge_l_f1 = test_results['test_rouge_L_f1']
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement > self.params['patience']:
                self.logger.log_message(
                    f'Early stopping of training after {epochs_without_improvement} epochs without improvement'
                )
                self.logger.log_message(
                    f'Best epoch was: {best_epoch}, Best Rouge-L F1 was: {best_rouge_l_f1}'
                )
                return


    def convert_raw_prediction_to_text(
        self,
        raw_prediction,
        join_token='',
        camel_case=True
    ):
        prediction = [
            self.reverse_output_index.get(index, Common.OOV)
            for index in takewhile(lambda index: index != self.eos_index, raw_prediction)
        ]

        if camel_case:
            # convert tokens to camel case
            prediction = [prediction[0]] + [
                token[0].upper() + token[1:]
                for token in prediction[1:]
                if len(token) > 0
            ]

        return join_token.join(prediction)


    def evaluation_step(
        self,
        input_batch,
        target_batch,
    ):
        # apply teacher forcing
        # shift the output sequences with +1
        y_true = target_batch[:, 1:] # [batch_size, output_seq_length]

        # feed forward
        logits = self.call(
            inputs = [input_batch, target_batch],

            # TODO: do we need to initialize this hidden state everytime? move it inside call then?
            encoder_hidden = self.encoder.initialize_hidden_state(),
            training = False
        )
        # logits.shape is [batch_size, output_seq_length, output_vocab_size]

        loss = self.loss_function(
            y_true = y_true,
            y_pred = logits
        )

        # select the most probable tokens from logits
        predictions = tf.argmax(logits, axis=-1)

        return loss, predictions


    def evaluate(self, X_test, Y_test, batch_size, epoch):
        start_time = time.time()

        num_samples = len(X_test)
        steps_per_epoch = num_samples // batch_size
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

        total_loss = 0

        # go through the full test dataset
        for (input_batch, output_batch) in test_dataset:
            batch_loss, batch_predictions = self.evaluation_step(input_batch, output_batch)
            total_loss += batch_loss

            self.test_evaluator.add_batch(batch_predictions, output_batch)

        test_rouge_scores = self.test_evaluator.evaluate()

        self.logger.log_message(f'epoch {epoch} evaluation on test data time: {time.time() - start_time} sec')

        test_results = {
            'epoch': epoch,
            'epoch_test_loss': total_loss / steps_per_epoch,
            **prefix_dict_keys(test_rouge_scores.score, 'test_'),
        }

        self.logger.log_data(test_results)
        self.logger.log_message('epoch evaluation: ', test_results)

        return test_results


    # TODO: make it work with batches
    def predict_raw(
        self,
        input_sequence # preprocessed sequence with input token ids
    ):
        """Predict the output for the given input.

        Args:
            input_sequence (tensor): A fully preprocessed input sequence.

        Returns:
            A list with the predicted output token ids
            along with a matrix with attention weights for plotting.
        """

        start_of_seq_id = self.output_vocab_index[Common.SOS]
        end_of_seq_id = self.output_vocab_index[Common.EOS]

        hidden = self.encoder.initialize_hidden_state(batch_size = 1)

        encoder_outputs, encoder_hidden, _encoder_cell_state = self.encoder(
            tf.convert_to_tensor([input_sequence]), # the array brackets are needed for the batch_size dimension
            hidden,
            training=False,
        )

        # initialize the decoder hidden state with the hidden state of the encoder
        decoder_hidden = encoder_hidden

        # start decoding with the special start of sequence token
        decoder_input = tf.expand_dims([start_of_seq_id], axis = 0)

        result = []

        attention_rows = []

        for t in range(self.params['max_output_seq_length']):
            # if you use keyword args, then there will be an error
            # TypeError: __call__() missing 1 required positional argument: 'inputs'
            # oh well...
            predictions, decoder_hidden, attention_weights = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
                training=False
            )

            predicted_id = tf.argmax(predictions[0]).numpy()

            # the next decoder_input is the last predicted_id
            decoder_input = tf.expand_dims([predicted_id], axis = 0)

            result.append(predicted_id)

            # since the shape of attention_weights is (batch_size, max_input_seq_length, 1)
            # and batch_size is 1, then reshape the weights
            # and store them as a row in the attention plot at timestep t
            attention_rows.append(tf.reshape(attention_weights, shape = (-1, )))

            if predicted_id == end_of_seq_id:
                break

        attention_plot = tf.stack(attention_rows, axis = 0)

        # transpose the matrix so that each row is a token from the input sequence
        # and each column is a token from the output sequence
        attention_plot = tf.transpose(attention_plot)

        return result, attention_plot


    # TODO: add documentation
    def beam_search_predict_raw(self, input_sequence, k = 5, alpha = 0.7):
        start_of_seq_id = self.output_vocab_index[Common.SOS]
        end_of_seq_id = self.output_vocab_index[Common.EOS]

        encoder_outputs, encoder_hidden, _encoder_cell_state = self.encoder(
            tf.convert_to_tensor([input_sequence]), # the array brackets are needed for the batch_size dimension
            hidden=self.encoder.initialize_hidden_state(batch_size = 1),
            training=False,
        )

        # initialize the decoder hidden state with the hidden state of the encoder
        decoder_hidden = encoder_hidden

        best = []

        # start decoding with the special start of sequence token
        running_best = [
            # triplet: path with token ids, score, decoder_hidden_state
            [[start_of_seq_id], 0, decoder_hidden]
        ]

        # TODO: to how many timesteps should the search be limited?
        for _timestep in range(self.params['max_output_seq_length']):
            candidates = []

            # continue the beam search from where it left off on the last iteration
            for path, score, decoder_hidden in running_best:
                # the next decoder_input is the last predicted_id
                last_predicted_id = path[-1]
                decoder_input = tf.expand_dims([last_predicted_id], axis = 0)

                predictions, decoder_hidden, _attention_weights = self.decoder(
                    decoder_input,
                    decoder_hidden,
                    encoder_outputs,
                    training=False,
                )

                # the predictions are raw, so we use softmax to normalize to probabilities
                top_k_probabilities, top_k_indices = tf.math.top_k(
                    tf.nn.softmax(predictions[0]),
                    k = k
                )

                for top_token_id, probability in zip(top_k_indices.numpy(), top_k_probabilities.numpy()):
                    # ignore end of sequence tokens
                    if top_token_id == end_of_seq_id:
                        continue

                    candidates.append([
                        path + [top_token_id],
                        score + math.log(probability),
                        decoder_hidden
                    ])

            best_k_candidates = sorted(
                candidates,
                key = lambda triple: triple[1], # sort by score
                reverse = True # in descending order
            )[:k]

            # this iteration's top k candidates will be used to continue the search on the next step
            running_best = best_k_candidates

            # accumulate this iteration's top k candidates in the overall best selection
            best += [[path, score] for path, score, _decoder_hidden in best_k_candidates]

        # choose the final k best based on the score normalized by sequence length
        top_k = sorted(
            [
                [path, score / (len(path) ** alpha)]
                for path, score in best
            ],
            key = lambda tuple: tuple[1], # sort by the normalized score
            reverse = True # in descending order
        )[:k] # take only the top k

        # return the paths along their normalized scores
        return top_k


    # TODO: document that this function works with numpy arrays, not with TF tensors
    def predict(self, input_text):
        self.logger.log_message('Input text: ', input_text)

        tokens = tokenize_method(input_text)

        input_tokens = tokens

        self.logger.log_message('Tokenized text: ', tokens)

        encoded_tokens = np.array([
            self.input_vocab_index.get(token, 0)
            for token in tokens
        ])

        self.logger.log_message('Encoded tokens: ', encoded_tokens)

        raw_prediction, attention_weights = self.predict_raw(encoded_tokens)

        self.logger.log_message('Raw prediction: ', raw_prediction)

        output_tokens = [
            self.reverse_output_index.get(index, Common.OOV)
            for index in raw_prediction
        ]

        clean_raw_prediction = takewhile(
            lambda index: index != self.output_vocab_index[Common.EOS],
            raw_prediction
        )

        predicted_text = ''.join([
            self.reverse_output_index.get(index, Common.OOV)
            for index in clean_raw_prediction
        ])

        self.logger.log_message('Predicted text: ', predicted_text)

        return predicted_text, attention_weights, input_tokens, output_tokens


    # TODO: reduce duplication of predict methods
    def predict_beam_search(self, input_text):
        self.logger.log_message('Input text: ', input_text)

        tokens = tokenize_method(input_text)

        self.logger.log_message('Tokenized text: ', tokens)

        encoded_tokens = np.array([
            self.input_vocab_index.get(token, 0)
            for token in tokens
        ])

        self.logger.log_message('Encoded tokens: ', encoded_tokens)

        raw_predictions = self.beam_search_predict_raw(
            encoded_tokens
        )

        self.logger.log_message('Raw predictions with scores: ', raw_predictions)

        raw_predictions = [path for path, score in raw_predictions]

        self.logger.log_message('Raw predictions: ', raw_predictions)

        clean_raw_predictions = [
            takewhile(
                lambda index: index != self.output_vocab_index[Common.EOS],
                raw_prediction
            )
            for raw_prediction in raw_predictions
        ]

        predicted_texts = [
            ''.join([
                self.reverse_output_index.get(index, Common.OOV)
                for index in clean_raw_prediction
            ])
            for clean_raw_prediction in clean_raw_predictions
        ]

        self.logger.log_message('Predicted texts: ', predicted_texts)

        return predicted_texts
