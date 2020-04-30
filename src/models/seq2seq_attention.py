import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

# import os
# import io
# import itertools

BATCH_SIZE = 64
BUFFER_SIZE = len(X_train)
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
embedding_dims = 256
rnn_units = 1024
dense_units = 1024
Dtype = tf.float32   #used to initialize DecoderCell Zero state

Tx = len(X_train) # TODO: what is Tx?

class Encoder(tf.keras.Model):
    def __init__(
        self,
        input_vocab_size,
        embedding_dims,
        rnn_units
    ):
        super().__init__()

        self.encoder_embedding = tf.keras.layers.Embedding(
            input_dim=input_vocab_size,
            output_dim=embedding_dims,
        )

        # TODO: rename to encoder_rnn
        self.encoder_rnnlayer = tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True,
            return_state=True
        )

class Decoder(tf.keras.Model):
    def __init__(
        self,
        output_vocab_size,
        embedding_dims,
        rnn_units
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
            BATCH_SIZE * [Tx] # TODO: what is this parameter?
        )

        # TODO: didn't we build a decoder RNN cell above already?
        self.rnn_cell = self.build_rnn_cell(
            BATCH_SIZE
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

    def build_rnn_cell(self, batch_size):
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
        input_vocab_size,
        output_vocab_size,
        embedding_dims,
        rnn_units,
    ):
        self.encoder = Encoder(
            input_vocab_size,
            embedding_dims,
            rnn_units
        )

        self.decoder = Decoder(
            output_vocab_size,
            embedding_dims,
            rnn_units
        )

        # TODO: expose the optimizer as a hyper parameter? where do we give the learning rate? is it adaptive? can we log it?
        self.optimizer = tf.keras.optimizers.Adam()


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
            # TODO: extract the feed forward pass as a method

            # feed forward through encoder
            # TODO: put accurate names everywhere below
            encoder_emb_inp = self.encoder.encoder_embedding(input_batch)

            a, a_tx, c_tx = self.encoder.encoder_rnnlayer(
                encoder_emb_inp,
                initial_state=encoder_initial_cell_state
            )

            # apply teacher forcing
            # ignore the <end> marker token for the decoder input
            decoder_input = output_batch[:, :-1]
            # shift the output sequences with +1
            decoder_output = output_batch[:, 1:]

            # feed forward through decoder
            decoder_emb_inp = self.decoder.decoder_embedding(decoder_input)

            # set up decoder memory from encoder output
            self.decoder.attention_mechanism.setup_memory(a)

            decoder_initial_state = self.decoder.build_decoder_initial_state(
                BATCH_SIZE, # TODO: this parameter should be known already
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
                sequence_length=BATCH_SIZE * [Ty - 1]
            )

            # TODO: please, argument why logits? and what is the type of outputs? i expected only tensors
            logits = outputs.rnn_output

            loss = self.loss_function(logits, decoder_output)

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
        return [
            tf.zeros((BATCH_SIZE, rnn_units)),
            tf.zeros((BATCH_SIZE, rnn_units))
        ]


    def train(self, dataset, epochs):
        # TODO: maybe better define the dataset itself here? because we need to know its shapes

        for i in range(1, epochs + 1):
            encoder_initial_cell_state = self.initialize_initial_state()
            total_loss = 0.0

            for (batch, (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(
                    input_batch,
                    output_batch,
                    encoder_initial_cell_state # TODO: shouldn't we persist this state through training steps?
                )

                total_loss += batch_loss

                # TODO: integrate with wandb to save model checkpoints and metrics

                if (batch + 1) % 5 == 0:
                    print(f'total loss: {batch_loss.numpy()}, epoch {i}, batch {batch + 1}')
