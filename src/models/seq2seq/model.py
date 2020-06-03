import tensorflow as tf
import numpy as np
import wandb
import os
import time
import json

from collections import Counter
from itertools import takewhile

from src.models.seq2seq.encoder import Encoder
from src.models.seq2seq.decoder import Decoder
from src.metrics.f1_score import F1Score
from src.preprocessing.tokens import tokenize_method


class Seq2Seq(tf.Module):
    def __init__(
        self,
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
        eval_averaging = 'micro',
    ):
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
            'eval_averaging': eval_averaging,
        }

        self.input_vocab_index = input_vocab_index
        self.output_vocab_index = output_vocab_index
        self.reverse_input_index = dict(
            (i, token) for token, i in input_vocab_index.items()
        )
        self.reverse_output_index = dict(
            (i, token) for token, i in output_vocab_index.items()
        )

        self.encoder = Encoder(
            input_vocab_size=self.params['input_vocab_size'],
            embedding_dims=self.params['input_embedding_dim'],
            rnn_units=self.params['rnn_units'],
            batch_size=self.params['batch_size'],
        )

        self.decoder = Decoder(
            output_vocab_size=self.params['output_vocab_size'],
            embedding_dims=self.params['output_embedding_dim'],
            rnn_units=self.params['rnn_units'],
            batch_size=self.params['batch_size'],
        )

        # TODO: expose the optimizer as a hyper parameter? where do we give the learning rate? is it adaptive? can we log it?
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_metrics = {
            'sparse_categorical_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
            'f1_score': F1Score(
                num_classes=self.params['output_vocab_size'],
                from_logits=True,
                averaging=self.params['eval_averaging'], # TODO: micro vs macro averaging?
                dtype=tf.int32, # TODO: shouldn't the dtype be handled inside the metric?
            ),
        }

        # TODO: reduce code duplication with train_metrics
        self.test_metrics = {
            'sparse_categorical_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
            'f1_score': F1Score(
                num_classes=self.params['output_vocab_size'],
                from_logits=True,
                averaging=self.params['eval_averaging'], # TODO: micro vs macro averaging?
                dtype=tf.int32, # TODO: shouldn't the dtype be handled inside the metric?
            ),
        }

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


    # TODO: update call method with new model internals
    def call(self, inputs, training=False):
        # TODO: differentiate between training and NOT training, if you add a Dropout layer

        encoder_inputs, decoder_inputs = inputs

        # feed forward through encoder
        self.encoder.clear_initial_cell_state(batch_size=encoder_inputs.shape[0])
        encoder_outputs, encoder_hidden_state, encoder_memory_state = self.encoder(encoder_inputs)

        # feed forward through decoder

        # set up decoder memory from encoder output
        # this also sets up the decoder initial state based on the encoder last hidden and memory states
        self.decoder.setup_memory_and_initial_state(
            encoder_outputs=encoder_outputs,
            encoder_states=[encoder_hidden_state, encoder_memory_state],
        )

        # ignore hidden state and cell state from decoder RNN
        outputs, _, _ = self.decoder(decoder_inputs)

        return outputs.rnn_output # [batch_size, output_seq_length, output_vocab_size]


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
        save_path = self.checkpoint_manager.save()

        print('Saving checkpoint in wandb')
        wandb.save(save_path)

        return save_path


    def save(self):
        self.save_checkpoint()

        config_filename = os.path.join(self.params['checkpoint_dir'], 'config.json')
        with open(config_filename, 'w') as f:
            json.dump(self.params, f)

        wandb.save(config_filename)
        print('Done saving model')


    def restore_latest_checkpoint(self):
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint

        self.checkpoint.restore(latest_checkpoint)

        if latest_checkpoint:
            print("Restored from {}".format(latest_checkpoint))
        else:
            print("Initializing from scratch.")


    @staticmethod
    def restore(checkpoint_dir, input_vocab_index, output_vocab_index):
        print('Restoring model config')

        with open(os.path.join(checkpoint_dir, 'config.json')) as f:
            config = json.load(f)

        print('Loaded model config: ', config)

        model = Seq2Seq(
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
            eval_averaging = config['eval_averaging'],
        )

        model.restore_latest_checkpoint()

        model.build() # it is necessary to build the model before accessing its variables

        print('Done restoring model')

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
        batch_loss = 0.0

        with tf.GradientTape() as tape:
            loss = 0.0

            encoder_outputs, encoder_hidden, _encoder_cell_state = self.encoder(input_batch, encoder_hidden)

            # initialize the decoder's hidden state with the final hidden state of the encoder
            decoder_hidden = encoder_hidden

            # TODO: extract the <SOS> token in a Common module

            # shape: (batch_size, 1), where the 1 is the single <start> timestep
            decoder_input = tf.expand_dims(
                [self.output_vocab_index['<SOS>']] * self.params['batch_size'],
                axis = 1 # expand the last dimension, leave the batch_size in tact
            )

            target_seq_length = int(target_batch.shape[1])
            # Teacher forcing - feeding the target as the next input
            # we iterate timesteps from 1, not from 0, because this way we shift the target sequences by one
            # i.e. if we have [<SOS>, get, search, data, <EOS>] as a target sequence
            # we first feed <SOS> (index 0) into the decoder and expect it to predict "get" (index 1)
            # and on the next step the "get" (index 1) is fed into the decoder, expecting it to predict "search" (index 2)
            # and so on.
            for t in range(1, target_seq_length): # for each timestep in the output sequence
                # passing encoder_outputs to the decoder for a single timestep
                # notice that we overwrite decoder_hidden on every timestep
                predictions, decoder_hidden, _ = self.decoder(
                    decoder_input,
                    decoder_hidden,
                    encoder_outputs
                )

                y_true = target_batch[:, t]

                # add to loss
                loss += self.loss_function(y_true, predictions)

                # TODO: extract the forward pass inside call(), use tf.scatter_update to produce a single tensor with all timesteps
                # TODO: move the metrics state update after the forward pass
                # TODO: the first token <SOS> can be removed from both y_true and y_pred
                self.train_metrics['sparse_categorical_accuracy'].update_state(
                    y_true = y_true,
                    y_pred = predictions
                )

                self.train_metrics['f1_score'].update_state(
                    y_true = y_true,
                    y_pred = predictions,
                )

                # using teacher forcing
                # pass the true target/output token from timestep t as input to the decoder for timestep t+1
                decoder_input = tf.expand_dims(y_true, 1)

            batch_loss = loss / target_seq_length

        # get the list of all trainable weights (variables)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        # differentiate loss with regard to variables (the parameters of the model)
        gradients = tape.gradient(loss, variables)

        # combine into pairs each variable with its gradient
        grads_and_vars = zip(gradients, variables)

        # adjust the weights / parameters with the computed gradients
        self.optimizer.apply_gradients(grads_and_vars)

        return batch_loss


    def train(self, X_train, Y_train, X_test, Y_test, epochs, on_epoch_end):
        self.restore_latest_checkpoint()

        num_samples = len(X_train)
        batch_size = self.params['batch_size']
        steps_per_epoch = num_samples // batch_size
        BUFFER_SIZE=5000 # TODO: expose as a parameter

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0.0

            # reset accumulated metrics
            self.train_metrics['sparse_categorical_accuracy'].reset_states()
            self.train_metrics['f1_score'].reset_states()

            # TODO: is this necessary?
            encoder_hidden_state = self.encoder.initialize_hidden_state()

            for (step, (input_batch, target_batch)) in enumerate(train_dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(
                    # the models expect tf.float32
                    tf.cast(input_batch, dtype=tf.float32),
                    tf.cast(target_batch, dtype=tf.float32),
                    encoder_hidden = encoder_hidden_state,
                )

                sparse_categorical_accuracy = self.train_metrics['sparse_categorical_accuracy'].result()

                # TODO: compute accuracy
                f1, precision, recall = self.train_metrics['f1_score'].result()

                total_loss += batch_loss

                wandb.log({
                    'batch': step,
                    'loss': batch_loss,
                    'sparse_categorical_accuracy': sparse_categorical_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                })

                if (step + 1) % 10 == 0:
                    # TODO: just log the dict above? also format the numbers to 2 decimal places?
                    print(f'epoch {epoch} - batch {step + 1} - loss {batch_loss} - precision {precision} - recall {recall} - f1 {f1} - sparse_categorical_accuracy {sparse_categorical_accuracy}')

            print(f'epoch {epoch} training time: {time.time() - start_time} sec')
            wandb.log({
                'epoch': epoch,
                'epoch_loss': total_loss / steps_per_epoch,
                'epoch_sparse_categorical_accuracy': sparse_categorical_accuracy,
                'epoch_precision': precision,
                'epoch_recall': recall,
                'epoch_f1': f1,
            })

            self.evaluate(X_test, Y_test, batch_size, epoch)

            on_epoch_end()

            # TODO: extract in on_epoch_end()?
            # save the whole model on every 3rd epoch
            if epoch % 3 == 0:
                save_path = self.checkpoint_manager.save()
                print("epoch {} saved checkpoint: {}".format(epoch, save_path))


    def evaluation_step(
        self,
        input_batch,
        output_batch,
    ):
        # apply teacher forcing
        # ignore the <end> marker token for the decoder input
        decoder_input = output_batch[:, :-1]
        # shift the output sequences with +1
        decoder_output = output_batch[:, 1:] # [batch_size, output_seq_length]

        # feed forward
        logits = self.call(inputs=[input_batch, decoder_input], training = False)
        # logits.shape is [batch_size, output_seq_length, output_vocab_size]

        loss = self.loss_function(
            y_true = decoder_output,
            y_pred = logits
        )

        self.test_metrics['sparse_categorical_accuracy'].update_state(
            y_true=decoder_output,
            y_pred=logits
        )

        self.test_metrics['f1_score'].update_state(
            y_true = decoder_output,
            y_pred = logits,
        )

        return loss


    def evaluate(self, X_test, Y_test, batch_size, epoch):
        start_time = time.time()

        num_samples = len(X_test)
        steps_per_epoch = num_samples // batch_size
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

        # reset accumulated metrics
        self.test_metrics['sparse_categorical_accuracy'].reset_states()
        self.test_metrics['f1_score'].reset_states()
        total_loss = 0

        # go through the full test dataset
        for (input_batch, output_batch) in test_dataset:
            batch_loss = self.evaluation_step(input_batch, output_batch)
            total_loss += batch_loss

        sparse_categorical_accuracy = self.test_metrics['sparse_categorical_accuracy'].result()
        f1, precision, recall = self.test_metrics['f1_score'].result()

        print(f'epoch {epoch} evaluation time: {time.time() - start_time} sec')

        test_results = {
            'epoch': epoch,
            'epoch_test_loss': total_loss / steps_per_epoch,
            'epoch_test_sparse_categorical_accuracy': sparse_categorical_accuracy,
            'epoch_test_precision': precision,
            'epoch_test_recall': recall,
            'epoch_test_f1': f1,
        }

        # TODO: have better logging
        wandb.log(test_results)
        print('epoch evaluation: ', test_results)

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

        start_of_seq_id = self.output_vocab_index['<SOS>']
        end_of_seq_id = self.output_vocab_index['<EOS>']

        attention_plot = np.zeros((
            self.params['max_output_seq_length'],
            self.params['max_input_seq_length']
        ))

        hidden = self.encoder.initialize_hidden_state(batch_size = 1)

        encoder_outputs, encoder_hidden, _encoder_cell_state = self.encoder(
            tf.convert_to_tensor([input_sequence]), # the array brackets are needed for the batch_size dimension
            hidden
        )

        # initialize the decoder hidden state with the hidden state of the encoder
        decoder_hidden = encoder_hidden

        # start decoding with the special start of sequence token
        decoder_input = tf.expand_dims([start_of_seq_id], axis = 0)

        result = []

        for t in range(self.params['max_output_seq_length']):
            # if you use keyword args, then there will be an error
            # TypeError: __call__() missing 1 required positional argument: 'inputs'
            # oh well...
            predictions, decoder_hidden, attention_weights = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs
            )

            predicted_id = tf.argmax(predictions[0]).numpy()

            # the next decoder_input is the last predicted_id
            decoder_input = tf.expand_dims([predicted_id], axis = 0)

            result.append(predicted_id)

            # since the shape of attention_weights is (batch_size, max_input_seq_length, 1)
            # and batch_size is 1, then reshape the weights
            # and store them as a row in the attention plot at timestep t
            attention_plot[t] = tf.reshape(attention_weights, shape = (-1, )).numpy()

            if predicted_id == end_of_seq_id:
                break

        return result, attention_plot


    # TODO: document that this function works with numpy arrays, not with TF tensors
    def predict(self, input_text):
        print('Input text: ', input_text)

        tokens = tokenize_method(input_text)

        print('Tokenized text: ', tokens)

        encoded_tokens = np.array([
            self.input_vocab_index.get(token, 0)
            for token in tokens
        ])

        print('Encoded tokens: ', encoded_tokens)

        raw_prediction, attention_plot = self.predict_raw(encoded_tokens)

        print('Raw prediction: ', raw_prediction)

        clean_raw_prediction = takewhile(
            lambda index: index != self.output_vocab_index['<EOS>'],
            raw_prediction
        )

        predicted_text = ''.join([
            self.reverse_output_index.get(index, '<OOV>')
            for index in clean_raw_prediction
        ])

        print('Predicted text: ', predicted_text)

        return predicted_text, attention_plot
