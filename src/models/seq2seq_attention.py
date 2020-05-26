import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import wandb
import os
import time
import json

from collections import Counter
from itertools import takewhile

from src.metrics.f1_score import F1Score
from src.preprocessing.tokens import tokenize_method


class Encoder(tf.keras.Model):
    def __init__(
        self,
        input_vocab_size,
        embedding_dims,
        rnn_units,
        batch_size, # TODO: can we not pass the batch_size?
        *args,
        **kwargs,
    ):
        super().__init__(self, args, kwargs)

        self.config = {
            'input_vocab_size': input_vocab_size,
            'embedding_dims': embedding_dims,
            'rnn_units': rnn_units,
            'batch_size': batch_size,
        }

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.config['input_vocab_size'],
            output_dim=self.config['embedding_dims'],
            name='EncoderEmbedding',
        )

        self.encoder_rnn = tf.keras.layers.LSTM(
            self.config['rnn_units'],
            return_sequences=True,
            return_state=True,
            name='EncoderLSTM',
            # default kernel_initializer is 'glorot_uniform',
            # default recurrent_initializer is 'orthogonal'
            # default bias_initializer is 'zeros'
        )

        self.clear_initial_cell_state(batch_size=self.config['batch_size'])


    def clear_initial_cell_state(self, batch_size):
        self.encoder_initial_cell_state = [
            tf.zeros((batch_size, self.config['rnn_units'])),
            tf.zeros((batch_size, self.config['rnn_units'])),
        ]


    # annotating with @tf.function leads to None gradients while training
    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, None))])
    def call(self, input_batch):
        output, last_step_hidden_state, last_step_memory_state = self.encoder_rnn(
            self.embedding(input_batch),
            initial_state=self.encoder_initial_cell_state
        )

        return (
            output, # [batch_size, input sequence length, rnn_units]
            last_step_hidden_state, # [batch_size, rnn_units]
            last_step_memory_state # [batch_size, rnn_units]
        )

    def get_config(self):
        return self.config


class Decoder(tf.keras.Model):
    def __init__(
        self,
        max_output_seq_length,
        output_vocab_size,
        embedding_dims,
        rnn_units,
        dense_units,
        batch_size,
        *args,
        **kwargs,
    ):
        super().__init__(self, args, kwargs)

        self.config = {
            'max_output_seq_length': max_output_seq_length,
            'output_vocab_size': output_vocab_size,
            'embedding_dims': embedding_dims,
            'rnn_units': rnn_units,
            'dense_units': dense_units,
            'batch_size': batch_size,
        }

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.config['output_vocab_size'],
            output_dim=self.config['embedding_dims'],
            name='DecoderEmbedding'
        )

        # TODO: why isn't the activation softmax?
        self.dense_layer = tf.keras.layers.Dense(
            self.config['output_vocab_size'],
            name='DenseOutput'
        )

        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(
            self.config['rnn_units'],
            name='DecoderLSTMCell'
        )

        # TODO: why prefer Luong over tfa.seq2seq.BahdanauAttention or vice-versa?
        self.attention_mechanism = tfa.seq2seq.LuongAttention(
            self.config['dense_units'],
            memory = None,
            memory_sequence_length = self.config['batch_size'] * [self.config['max_output_seq_length']]
        )

        self.rnn_cell = tfa.seq2seq.AttentionWrapper(
            self.decoder_rnn_cell,
            self.attention_mechanism,
            attention_layer_size=self.config['dense_units'],
        )

        # TODO: isn't this sampler only for training? what if we need to pass a sampler for Beam Search?
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        self.decoder = tfa.seq2seq.BasicDecoder(
            self.rnn_cell,
            sampler=self.sampler,
            output_layer=self.dense_layer,
        )

        self.setup_memory_and_initial_state() # setup memory with zeros, since we don't have encoder outputs


    def build_decoder_initial_state(
        self,
        batch_size,
        encoder_state,
    ):
        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_size,
            dtype=tf.float32, # TODO: do we need this dtype at all?
        )

        # TODO: why clone? do we clone the encoder_state? what's going on here?
        return decoder_initial_state.clone(cell_state=encoder_state)


    def setup_memory_and_initial_state(
        self,
        encoder_outputs=None,
        encoder_states=None,
        batch_size=None
    ):
        self.attention_mechanism.setup_memory(
            encoder_outputs if encoder_outputs is not None else tf.zeros((
                self.config['batch_size'],
                200, # TODO: provide input sequence length
                self.config['rnn_units'],
            ))
        )

        self.decoder_initial_state = self.build_decoder_initial_state(
            # the batch size can be different when in prediction mode
            batch_size if batch_size is not None else self.config['batch_size'],

            # [last step activations, last memory_state] of encoder is passed as input to decoder Network
            encoder_state = encoder_states if encoder_states is not None else [
                tf.zeros((self.config['batch_size'], self.config['rnn_units'])),
                tf.zeros((self.config['batch_size'], self.config['rnn_units'])),
            ],
        )

    # annotating with @tf.function leads to None gradients while training
    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, None))])
    def call(self, input_batch):
        # TODO: document that the memory must be set up with encoder outputs before calling call()

        outputs, hidden_state, cell_state = self.decoder(
            self.embedding(input_batch),
            initial_state=self.decoder_initial_state,

            # TODO: don't we know the BATCH_SIZE already inside the decoder? should we?
            # output sequence length - 1 because of teacher forcing
            sequence_length=self.config['batch_size'] * [self.config['max_output_seq_length'] - 1]
        )

        return outputs, hidden_state, cell_state


    def get_config(self):
        return self.config


class Seq2SeqAttention(tf.Module):
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
        dense_units = 1024,
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
            'dense_units': dense_units,
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
            max_output_seq_length=self.params['max_output_seq_length'],
            output_vocab_size=self.params['output_vocab_size'],
            embedding_dims=self.params['output_embedding_dim'],
            rnn_units=self.params['rnn_units'],
            dense_units=self.params['dense_units'],
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
        self.decoder.build(input_shape=(self.params['batch_size'], self.params['max_output_seq_length']))


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

        model = Seq2SeqAttention(
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
            dense_units = config['dense_units'],
            batch_size = config['batch_size'],
            eval_averaging = config['eval_averaging'],
        )

        model.restore_latest_checkpoint()

        model.build() # it is necessary to build the model before accessing its variables

        print('Done restoring model')

        return model


    # TODO: add documentation
    def loss_function(
        self,
        y_pred, # shape: [batch_size, Ty, output_vocab_size]
        y # shape: [batch_size, Ty]
    ):
        # TODO: why Sparse instead of non-sparse?
        # TODO: what does from_logits mean? is it the output of a softmax layer?  i guess it means that we have a distribution, i.e. one dimension more
        # TODO: what does the reduction parameter do?
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
        )

        # compute the actual losses
        loss = loss_fn(
            y_true=y,
            y_pred=y_pred
        )

        # skip loss calculation for padding sequences which contain only zeroes (i.e. when y = 0)
        # mask the loss when padding sequence appears in the output sequence
        # e.g.
        # [ <start>,transform, search, response, data, 0, 0, 0, 0, ..., <end>]
        # [ 1      , 234     , 3234  , 423     , 3344, 0, 0, 0 ,0, ..., 2 ]
        mask = tf.logical_not(tf.math.equal(y, 0)) # output 0 when y = 0, otherwise output 1
        mask = tf.cast(mask, dtype=loss.dtype)

        loss = mask * loss
        loss = tf.reduce_mean(loss) # TODO: on which axises is the reduction done?

        return loss


    # TODO: what will happen if we annotate with @tf.function?
    def train_step(
        self,
        input_batch,
        output_batch,
    ):
        loss = 0.0

        with tf.GradientTape() as tape:
            # apply teacher forcing
            # ignore the <end> marker token for the decoder input
            decoder_input = output_batch[:, :-1]
            # shift the output sequences with +1
            decoder_output = output_batch[:, 1:] # [batch_size, output_seq_length]

            # feed forward
            logits = self.call(inputs=[input_batch, decoder_input], training = True)
            # logits.shape is [batch_size, output_seq_length, output_vocab_size]

            loss = self.loss_function(logits, decoder_output)

            self.train_metrics['sparse_categorical_accuracy'].update_state(
                y_true=decoder_output,
                y_pred=logits
            )

            self.train_metrics['f1_score'].update_state(
                y_true = decoder_output,
                y_pred = logits,
            )

        # get the list of all trainable weights (variables)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        # differentiate loss with regard to variables (the parameters of the model)
        gradients = tape.gradient(loss, variables)

        # combine into pairs each variable with its gradient
        grads_and_vars = zip(gradients, variables)

        # adjust the weights / parameters with the computed gradients
        self.optimizer.apply_gradients(grads_and_vars)

        return loss


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

            for (step, (input_batch, output_batch)) in enumerate(train_dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(
                    # the models expect tf.float32
                    tf.cast(input_batch, dtype=tf.float32),
                    tf.cast(output_batch, dtype=tf.float32),
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
        logits = self.call(inputs=[input_batch, decoder_input], training = True)
        # logits.shape is [batch_size, output_seq_length, output_vocab_size]

        loss = self.loss_function(logits, decoder_output)

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


    def predict_raw(self, input_sequences):
        """Predict the outputs for the given inputs.

        Args:
            input_batch (tensor): A fully preprocessed batch of input sequences.

        Returns:
            A tensor with the raw predicted output sequences as numbers.
        """

        start_token_index = self.output_vocab_index['<SOS>']
        end_token_index = self.output_vocab_index['<EOS>']

        # compute the size of input sequences batch
        inference_batch_size = input_sequences.shape[0]

        # feed forward input sequences through the encoder
        self.encoder.clear_initial_cell_state(batch_size = inference_batch_size)
        encoder_outputs, encoder_hidden_state, encoder_memory_state = self.encoder(input_sequences)

        # initialize decoder

        # TODO: how is this different from? [start_token_index] * inference_batch_size
        start_tokens = tf.fill(
            [inference_batch_size],
            start_token_index
        )

        end_token = end_token_index

        # TODO: add a training=False argument to self.decoder.call()

        # the sampler is initialized inside the basic decoder below
        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        # TODO: understand why and explain? isn't ([[start]] * batch_size) the same?
        # TODO: understand how samplers work
        # a new decoder is created because we use a different embedding sampler
        decoder_instance = tfa.seq2seq.BasicDecoder(
            cell=self.decoder.rnn_cell,
            sampler=greedy_sampler,
            output_layer=self.decoder.dense_layer
        )

        # TODO: load variable from checkpoint?
        # instead of feeding forward through the decoder embedding layer
        # we use an EmbeddingSampler
        decoder_embedding_matrix = self.decoder.embedding.variables[0]

        # setup self.decoder.decoder_initial_state
        self.decoder.setup_memory_and_initial_state(
            encoder_outputs=encoder_outputs,
            encoder_states=[encoder_hidden_state, encoder_memory_state],
            batch_size=inference_batch_size
        )

        # the kwargs being passed here are passed to the sampler itself
        # and the first two values of the returned tuple are returned by the sampler initialization
        _first_finished, first_inputs, first_state = decoder_instance.initialize(
            decoder_embedding_matrix,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=self.decoder.decoder_initial_state,
        )

        # TODO: decide on a maximum output sequence length
        # inference can produce output sequences longer than
        # the limited length output sequences used during training
        maximum_iterations = 2 * self.params['max_output_seq_length']

        inputs = first_inputs
        state = first_state

        # TODO: prefer TF tensors over NumPy arrays
        predictions = np.empty((inference_batch_size, 0), dtype = np.int32)
        for step in range(maximum_iterations):
            outputs, next_state, next_inputs, _finished = decoder_instance.step(
                step,
                inputs,
                state
            )

            inputs = next_inputs
            state = next_state
            outputs = np.expand_dims(outputs.sample_id, axis = -1)
            predictions = np.append([predictions, outputs], axis = -1)

        return predictions


    def predict(self, input_text):
        print('Input text: ', input_text)

        tokens = tokenize_method(input_text)

        print('Tokenized text: ', tokens)

        encoded_tokens = np.array([
            self.input_vocab_index.get(token, 0)
            for token in tokens
        ])

        print('Encoded tokens: ', encoded_tokens)

        raw_predictions = self.predict_raw(input_sequences=tf.constant([encoded_tokens]))

        # TODO: document that this function works with numpy arrays, not with TF tensors
        raw_prediction = raw_predictions.numpy()[0]

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

        return predicted_text
