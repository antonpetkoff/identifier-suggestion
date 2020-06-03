import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)


    def call(
        self,
        inputs, # query (i.e. hiddent state) shape is (batch_size, hidden_size)
        values # values (i.e. encoder outputs) shape is (batch_size, max_input_seq_length, hidden_size)
    ):
        # the first argument must be named "inputs" in order to resolve the following error
        # TypeError: __call__() missing 1 required positional argument: 'inputs'
        query = inputs

        # add a dimension for timesteps to the query
        # shape is (batch_size, 1, hidden_size)
        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(
            tf.nn.tanh(
                self.W1(query_with_time_axis) + self.W2(values)
            )
        )

        # shape is (batch_size, max_input_seq_length, 1)
        attention_weights = tf.nn.softmax(score, axis = 1)

        # apply (multiply) the attention weights to the encoder timesteps
        context_vector = attention_weights * values

        # sum over all timesteps, to produce a single context vector with the size of the hidden state
        context_vector = tf.reduce_sum(context_vector, axis = 1)

        return (
            context_vector, # shape is (batch_size, hidden_size)
            attention_weights # shape is (batch_size, max_input_seq_length, 1)
        )
