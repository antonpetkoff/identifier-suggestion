import tensorflow as tf

from base import BaseModel
from tensorflow.keras import layers

class Seq2Seq(BaseModel):
    def __init__(self, params):
        self.params = {
            max_input_length: 100,
            max_output_length: method_name_subtokens_with_start_and_end.apply(len).max(), # 10
            # TODO: limit the input vocabulary size for now
            input_vocab_size: 10000 # the number of method body tokens,
            input_embedding_dim: 50,

            output_vocab_size: len(output_vocabulary), # used for the softmax layer, like num_classes,
            output_embedding_dim: 50,

            latent_dim: 128 # encoder-decoder latent space dimensions,

            epochs: 10,
            batch_size: 64,
        }
        self.training = self.build_training_model()
        # TODO: should we build the inference model after the training model is fitted?
        self.inference = self.build_inference_model()


    def build_training_model(self):
        encoder_inputs = layers.Input(shape=(None, ), name='encoder_inputs')
        encoder_embeddeding = layers.Embedding(
            input_dim=input_vocab_size,
            output_dim=input_embedding_dim,
            name='encoder_embedding'
        )(encoder_inputs)

        # Return states in addition to output
        _encoder_output, state_h, state_c = layers.LSTM(
            latent_dim,
            return_state=True,
            name='encoder_lstm'
        )(encoder_embeddeding)

        # Pass the 2 states to a new LSTM layer, as initial state
        encoder_states = [state_h, state_c]

        decoder_input = layers.Input(shape=(None, ), name='decoder_input')
        decoder_embeddeding = layers.Embedding(
            input_dim=output_vocab_size,
            output_dim=output_embedding_dim,
            name='decoder_embeddeding'
        )(decoder_input)

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = layers.LSTM(
            latent_dim,
            return_sequences=True,
            return_state=True,
            name='decoder_lstm'
        )

        decoder_outputs, _, _ = decoder_lstm(
            decoder_embeddeding,
            initial_state=encoder_states
        )

        decoder_dense = layers.Dense(
            output_vocab_size,
            activation='softmax',
            name='softmax'
        )
        output = decoder_dense(decoder_outputs)

        return {
            'model': tf.keras.Model([encoder_inputs, decoder_input], output),
            'encoder_inputs': encoder_inputs,
            'encoder_state': encoder_states,
            'decoder_lstm': decoder_lstm,
            'decoder_dense': decoder_dense,
        }

    # train_x is a dataframe with columns head.word, tail.word, sentence
    def fit_data_while_training(self, train_x, train_y):
        # self.train_features = self.transform(train_x)
        # self.train_labels = self.transform_labels(train_y)

        # TODO: do data preprocessing
        training_inputs = [tensor_encoder_inputs, tensor_decoder_inputs]
        training_labels = tf.sparse.to_dense(tensor_decoder_outputs)

        self.training.model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
        )

        training_history = self.training.model.fit(
            training_inputs,
            training_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )

        return training_history


    def build_inference_model(self):
        # inference mode (sampling):
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        encoder_model = Model(
            self.training.encoder_inputs,
            self.training.encoder_states
        )

        decoder_state_input_h = Input(shape=(self.params.latent_dim,))
        decoder_state_input_c = Input(shape=(self.params.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        # we ignore the decoder LSTM's outputs
        _, state_h, state_c = self.training.model.decoder_lstm(
            decoder_inputs,
            initial_state=decoder_states_inputs
        )

        decoder_states = [state_h, state_c]
        decoder_outputs = self.training.decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )

        return {
            'encoder_model': encoder_model,
            'decoder_model': decoder_model,

            # Reverse-lookup token index to decode sequences back to
            # something readable.
            'reverse_input_char_index': dict(
                (i, char) for char, i in input_token_index.items()
            ),
            'reverse_target_char_index': dict(
                (i, char) for char, i in target_token_index.items()
            )
        }

    def predict(self, input_seq):
        # features = self.transform(test_x)
        # predictions = self.model.predict(features)
        # return self.label_encoder.inverse_transform(predictions)
        # Encode the input as state vectors.

        states_value = self.inference.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.inference.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
            len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence



    # transforms the input train_x or test_x examples into features for the model
    # df is a dataframe with columns head.word, tail.word, sentence
    def transform(self, df):
        print('Tokenizing sentences...')
        df_tokenized = df['sentence'].apply(self.preprocess)
        print(df_tokenized.head())

        print('Averaging word embeddings...')
        df_vectors = df_tokenized.apply(self.average)
        print(df_vectors.head())

        vectors = np.asarray(flatten(df_vectors)).reshape(-1, self.word_embeddings_dim)
        print('Shape of transformed input: {}'.format(vectors.shape))

        return vectors


    def transform_labels(self, df):
        print('Fitting label encoder...')
        self.label_encoder.fit(df)
        print(self.label_encoder.classes_)

        print('Transforming labels...')
        labels = self.label_encoder.transform(df)
        print('Shape of transformed labels: {}'.format(labels.shape))

        return labels


    def preprocess(self, text):
        # tokenize
        word_tokens = self.tokenizer.tokenize(text)

        # clean stop words and lower cases
        return [
            word.lower()
            for word in word_tokens
            if not word in self.stop_words
        ]


    @staticmethod
    def load_word_embeddings(path):
        with open(path) as f:
            word_vec = json.load(f)

        word_embeddings = {
            obj['word']: np.asarray(obj['vec'])
            for obj in word_vec
        }
        return word_embeddings


    @staticmethod
    def average_embeddings(word_embeddings, word_embeddings_dim, words):
        embeddings = [
            word_embeddings[word]
            for word in words
            if word in word_embeddings
        ]

        if len(embeddings) > 0:
            return np.average(embeddings, axis=0)
        else:
            return np.zeros(word_embeddings_dim)

    def get_grid_params(self):
        return {
            'max_iter': (5, 100),
            'solver': ('liblinear', 'sag'),
        }
