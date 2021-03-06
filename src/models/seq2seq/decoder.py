import tensorflow as tf


from src.models.seq2seq.attention import BahdanauAttention

class Decoder(tf.keras.Model):
    def __init__(
        self,
        output_vocab_size,
        embedding_dims,
        rnn_units,
        batch_size,
        dropout_rate,
        *args,
        **kwargs,
    ):
        super().__init__(self, args, kwargs)

        self.config = {
            'output_vocab_size': output_vocab_size,
            'embedding_dims': embedding_dims,
            'rnn_units': rnn_units,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
        }

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.config['output_vocab_size'],
            output_dim=self.config['embedding_dims'],
            name='DecoderEmbedding'
        )

        self.decoder_rnn = tf.keras.layers.LSTM(
            self.config['rnn_units'],
            return_sequences = True,
            return_state = True,
            # recurrent_initializer is 'glorot_uniform' by default
            name='DecoderLSTM',
            dropout=self.config['dropout_rate'],
        )

        self.output_layer = tf.keras.layers.Dense(
            self.config['output_vocab_size'],
            name='DenseOutput'
        )

        self.attention = BahdanauAttention(self.config['rnn_units'])


    # this method is added for flexibility, so that we can call build() in the main model
    def set_state_from_encoder(self, hidden_state, encoder_outputs):
        self.hidden_state = hidden_state
        self.encoder_outputs = encoder_outputs


    def call(
        self,
        input_batch,
        hidden_state=None,
        encoder_outputs=None,
        training=False,
    ):
        if hidden_state is None:
            hidden_state = self.hidden_state

        if encoder_outputs is None:
            encoder_outputs = self.encoder_outputs

        # if you use keyword args, then there will be an error
        # TypeError: __call__() missing 1 required positional argument: 'inputs'
        # oh well...
        context_vector, attention_weights = self.attention(
            hidden_state, # query
            encoder_outputs # values
        )

        # shape of x is (batch_size, 1, embedding_dim), where 1 is the timestep, i.e. one timestep
        x = self.embedding(input_batch)

        # shape after concatenation is (batch_size, 1, embedding_dim + hidden_size)
        decoder_input = tf.concat(
            [
                tf.expand_dims(context_vector, 1),
                x
            ],
            axis = -1
        )

        # decoder output shape is (batch_size, 1, rnn_units)
        decoder_output, decoder_hidden_state, _decoder_cell_state = self.decoder_rnn(
            decoder_input,
            training=training,
        )

        # squash the timestep dimension of the decoder_output, because we have a single timestep
        # thus the shape becomes (batch_size, rnn_units)
        decoder_output = tf.reshape(
            decoder_output,
            (-1, decoder_output.shape[2])
        )

        # produce raw predictions over the output vocabulary
        # afterwards the most probable tokens will be sampled with argmax
        # the output_logits shape is (batch_size, output_vocab_size)
        output_logits = self.output_layer(decoder_output)

        return output_logits, decoder_hidden_state, attention_weights


    def get_config(self):
        return self.config
