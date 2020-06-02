import tensorflow as tf


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
