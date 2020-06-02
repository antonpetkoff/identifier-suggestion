import tensorflow as tf
import tensorflow_addons as tfa


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
