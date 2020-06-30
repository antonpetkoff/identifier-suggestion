import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(
        self,
        input_vocab_size,
        embedding_dims,
        rnn_units,
        batch_size,
        bidirectional,
        *args,
        **kwargs,
    ):
        super().__init__(self, args, kwargs)

        self.config = {
            'input_vocab_size': input_vocab_size,
            'embedding_dims': embedding_dims,
            'rnn_units': rnn_units,
            'batch_size': batch_size,
            'bidirectional': bidirectional,
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

        if self.config['bidirectional']:
            self.encoder_rnn = tf.keras.layers.Bidirectional(
                self.encoder_rnn,
                merge_mode='concat',
            )


    def initialize_hidden_state(self, batch_size=None):
        bsz = batch_size or self.config['batch_size']

        return [
            tf.zeros((bsz, self.config['rnn_units'])),
            tf.zeros((bsz, self.config['rnn_units'])),
        ]


    def call(self, input_batch, hidden = None):
        if hidden is None:
            hidden = self.initialize_hidden_state()

        output, last_step_hidden_state, last_step_memory_state = self.encoder_rnn(
            self.embedding(input_batch),
            initial_state = hidden,
        )

        return (
            output, # [batch_size, input sequence length, rnn_units]
            last_step_hidden_state, # [batch_size, rnn_units]
            last_step_memory_state # [batch_size, rnn_units]
        )

    def get_config(self):
        return self.config
