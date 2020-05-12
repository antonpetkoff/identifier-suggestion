import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import wandb
import os
import time

from collections import Counter

from src.metrics.f1_score import F1Score


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

    def build(self, input_shape):
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

        self.set_initial_cell_state()

        # we must add our own layers and weights, and then call super().build()
        super().build(input_shape)


    def set_initial_cell_state(self, initial_cell_state=None):
        self.encoder_initial_cell_state = [
            tf.zeros((self.config['batch_size'], self.config['rnn_units'])),
            tf.zeros((self.config['batch_size'], self.config['rnn_units'])),
        ] if initial_cell_state is None else initial_cell_state


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
        batch_size
    ):
        super().__init__()

        # TODO: have a separate parameter for the decoder embedding output dim
        self.decoder_embedding = tf.keras.layers.Embedding(
            input_dim=output_vocab_size,
            output_dim=embedding_dims,
        )

        # TODO: softmax output classifier?
        self.dense_layer = tf.keras.layers.Dense(
            output_vocab_size
        )

        # TODO: rename to decoder_rnn_cell
        self.decoder_rnncell = tf.keras.layers.LSTMCell(
            rnn_units
        )

        # TODO: isn't this sampler only for training? what if we need to pass a sampler for Beam Search?
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        # TODO: why no memory and what does this memory do?
        self.attention_mechanism = self.build_attention_mechanism(
            dense_units,
            None, # TODO: use named parameters
            batch_size * [max_output_seq_length] # TODO: what is this parameter?
        )

        # TODO: didn't we build a decoder RNN cell above already?
        self.rnn_cell = self.build_rnn_cell(
            dense_units,
            batch_size
        )

        # TODO: is there a fancier decoder?
        self.decoder = tfa.seq2seq.BasicDecoder(
            self.rnn_cell,
            sampler=self.sampler,
            output_layer=self.dense_layer,
        )

    def build_attention_mechanism(
        self,
        units, # TODO: the attention is over what? and what are units here?
        memory,
        memory_sequence_length
    ):
        # TODO: why prefer Luong over Bahdanau or vice-versa?
        return tfa.seq2seq.LuongAttention( # can be replaced with tfa.seq2seq.BahdanauAttention
            units,
            memory=memory,
            memory_sequence_length=memory_sequence_length
        )

    def build_rnn_cell(self, dense_units, batch_size):
        rnn_cell = tfa.seq2seq.AttentionWrapper(
            self.decoder_rnncell,
            self.attention_mechanism,
            attention_layer_size=dense_units # TODO: extract hyper parameters
        )

        return rnn_cell

    def build_decoder_initial_state(
        self,
        batch_size,
        encoder_state,
        dtype
    ):
        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_size,
            dtype=dtype
        )

        # TODO: why clone? do we clone the encoder_state? what's going on here?
        return decoder_initial_state.clone(cell_state=encoder_state)


class Seq2SeqAttention():
    def __init__(
        self,
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

        self.metrics = {
            'sparse_categorical_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
            'f1_score': F1Score(
                num_classes=self.params['output_vocab_size'],
                from_logits=True,
                averaging=self.params['eval_averaging'], # TODO: micro vs macro averaging?
                dtype=tf.int32, # TODO: shouldn't the dtype be handled inside the metric?
            ),
        }


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


    def train_step(
        self,
        input_batch,
        output_batch,
        encoder_initial_cell_state
    ):
        loss = 0.0

        with tf.GradientTape() as tape:
            # TODO: extract the feed forward pass as a method?

            # feed forward through encoder
            self.encoder.set_initial_cell_state(encoder_initial_cell_state)
            a, a_tx, c_tx = self.encoder(input_batch)

            # apply teacher forcing
            # ignore the <end> marker token for the decoder input
            decoder_input = output_batch[:, :-1]
            # shift the output sequences with +1
            decoder_output = output_batch[:, 1:] # [batch_size, output_seq_length]

            # feed forward through decoder
            decoder_emb_inp = self.decoder.decoder_embedding(decoder_input)

            # TODO: how often should we setup this memory?
            # set up decoder memory from encoder output
            self.decoder.attention_mechanism.setup_memory(a)

            decoder_initial_state = self.decoder.build_decoder_initial_state(
                self.params['batch_size'], # TODO: this parameter should be known already
                # [last step activations, last memory_state] of encoder is passed as input to decoder Network
                encoder_state=[a_tx, c_tx], # TODO: use more appropriate names
                dtype=tf.float32, # TODO: isn't the dtype always tf.float32?
            )

            # ignore hidden state and cell state from decoder RNN
            outputs, _, _ = self.decoder.decoder(
                decoder_emb_inp,
                initial_state=decoder_initial_state,

                # TODO: don't we know the BATCH_SIZE already inside the decoder? should we?
                # output sequence length - 1 because of teacher forcing
                sequence_length=self.params['batch_size'] * [self.params['max_output_seq_length'] - 1]
            )

            # TODO: please, argument why logits? and what is the type of outputs? i expected only tensors
            logits = outputs.rnn_output # [batch_size, output_seq_length, output_vocab_size]

            loss = self.loss_function(logits, decoder_output)


            # update evaluation metrics
            # TODO: evaluate on the train set during training or after training? performance and correctness-wise?

            self.metrics['sparse_categorical_accuracy'].update_state(
                y_true=decoder_output,
                y_pred=logits
            )

            self.metrics['f1_score'].update_state(
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


    def initialize_initial_state(self):
        # TODO: use random or Xavier initialization?
        # TODO: can we initialize all model parameters at once? or we need to initialize only a part of the parameters?
        # TODO: why return a list? instead return a tensor with one more dimension set to 2
        return [
            tf.zeros((self.params['batch_size'], self.params['rnn_units'])),
            tf.zeros((self.params['batch_size'], self.params['rnn_units']))
        ]


    def train(self, X_train, Y_train, X_test, Y_test, epochs, on_epoch_end):
        num_samples = len(X_train)
        batch_size = self.params['batch_size']
        steps_per_epoch = num_samples // batch_size
        BUFFER_SIZE=5000 # TODO: expose as a parameter

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0.0

            # reset accumulated metrics
            self.metrics['sparse_categorical_accuracy'].reset_states()
            self.metrics['f1_score'].reset_states()

            encoder_initial_cell_state = self.initialize_initial_state()

            for (step, (input_batch, output_batch)) in enumerate(train_dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(
                    input_batch,
                    output_batch,
                    encoder_initial_cell_state # TODO: shouldn't we persist this state through training steps?
                )

                sparse_categorical_accuracy = self.metrics['sparse_categorical_accuracy'].result()

                # TODO: compute accuracy
                f1, precision, recall = self.metrics['f1_score'].result()

                total_loss += batch_loss

                # TODO: add a custom validation step

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

            # TODO: evaluate(test_set)

            print(f'epoch {epoch} time: {time.time() - start_time} sec')
            wandb.log({
                'epoch': epoch,
                'epoch_loss': total_loss / steps_per_epoch,
                'epoch_sparse_categorical_accuracy': sparse_categorical_accuracy,
                'epoch_precision': precision,
                'epoch_recall': recall,
                'epoch_f1': f1,
            })
            on_epoch_end()


    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        print('Saving encoder weights locally')
        self.encoder.save_weights(os.path.join(save_dir, 'encoder.h5'))
        print('Saving decoder weights locally')
        self.decoder.save_weights(os.path.join(save_dir, 'decoder.h5'))
        print('Saved model weights locally')

        print('Saving model with wandb')
        wandb.save(os.path.join(save_dir, '*'))
        print('Done saving model')


    def predict(
        self,
        input_sequences,
        start_token_index,
        end_token_index,
    ):
        """Predict the outputs for the given inputs.

        Args:
            input_batch (tensor): A fully preprocessed batch of input sequences.

        Returns:
            A tensor with the raw predicted output sequences as numbers.
        """

        # compute the size of input sequences batch
        inference_batch_size = input_sequences.shape[0]

        # TODO: why do we initialize the encoder with zeroes? does it matter for inference?
        self.encoder.set_initial_cell_state([
            tf.zeros((inference_batch_size, self.params['rnn_units'])),
            tf.zeros((inference_batch_size, self.params['rnn_units'])),
        ])

        # feed forward input sequences through the encoder
        a, a_tx, c_tx = self.encoder(input_sequences)

        # initialize decoder

        # TODO: how is this different from? [start_token_index] * inference_batch_size
        start_tokens = tf.fill(
            [inference_batch_size],
            start_token_index
        )

        end_token = end_token_index

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

        self.decoder.attention_mechanism.setup_memory(a)

        # TODO: load variable from checkpoint?
        # instead of feeding forward through the decoder embedding layer
        # we use an EmbeddingSampler
        decoder_embedding_matrix = self.decoder.decoder_embedding.variables[0]

        decoder_initial_state = self.decoder.build_decoder_initial_state(
            inference_batch_size,
            encoder_state=[a_tx, c_tx],
            dtype=tf.float32 # TODO: isn't this known by default?
        )

        # the kwargs being passed here are passed to the sampler itself
        # and the first two values of the returned tuple are returned by the sampler initialization
        _first_finished, first_inputs, first_state = decoder_instance.initialize(
            decoder_embedding_matrix,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,
        )

        # TODO: decide on a maximum output sequence length
        # inference can produce output sequences longer than
        # the limited length output sequences used during training
        maximum_iterations = 2 * self.params['max_output_seq_length']

        inputs = first_inputs
        state = first_state
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
            predictions = np.append(predictions, outputs, axis = -1)

        return predictions