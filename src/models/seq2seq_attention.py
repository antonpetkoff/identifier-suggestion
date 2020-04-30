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

        # TODO: why Sparse instead of non-sparse?
        # TODO: from_logits is before the softmax layer? what does it mean?
        # TODO: what does the reduction parameter do?
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
        )

