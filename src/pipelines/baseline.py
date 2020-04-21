import argparse
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense

# custom imports
from src.utils.random import set_random_seeds
from src.evaluation.sequence import compute_f1_score
from src.preprocessing.tokens import tokenize_method_body, get_subtokens
from src.preprocessing.sequence import preprocess_sequences

# import tqdm and enable it for pandas for progress_apply
from tqdm import tqdm
tqdm.pandas()

set_random_seeds(1)

parser = argparse.ArgumentParser(description='Baseline Seq2Seq model')

# data files
parser.add_argument('--file_data_raw', type=str, help='Raw data file used for model training', required=True)
parser.add_argument('--file_model_output', type=str, help='Model output file name', required=True)
parser.add_argument('--dir_preprocessed_data', type=str, help='Directory for preprocessed data', required=True)

# hyper parameters
parser.add_argument('--max_input_length', type=int, help='Max input sequence length', required=True)

# TODO: should we include <start> and <end>? we will add the <start> and <end> tokens which will effectively increase the seq length to 10 during training and prediction
parser.add_argument('--max_output_length', type=int, help='Max output sequence length', required=True)
parser.add_argument('--input_vocab_size', type=int, help='Input vocabulary size', required=True)
parser.add_argument('--input_embedding_dim', type=int, help='Input embedding dimensionality', required=True)

# TODO: maybe we should name this max_output_vocab_size, since we can compute it from the input data set?
parser.add_argument('--output_vocab_size', type=int, help='Output vocabulary size', required=True)
parser.add_argument('--output_embedding_dim', type=int, help='Output embedding dimensionality', required=True)
parser.add_argument('--latent_dim', type=int, help='Encoder-Decoder latent space dimensionality', required=True)
parser.add_argument('--learning_rate', type=float, help='Learning Rate', required=True)
parser.add_argument('--epochs', type=int, help='Number of training epochs', required=True)
parser.add_argument('--batch_size', type=int, help='Batch Size', required=True)
parser.add_argument('--random_seed', type=int, help='Random Seed', required=True)


SEQ_START_TOKEN = '\t'
SEQ_END_TOKEN = '\n'
PAD_TOKEN = ' '

def seq2seq(args, input_texts, target_texts):
    # Vectorize the data.
    # add the PAD_TOKEN to both vocabularies in order to pad the sequences
    input_vocabulary = set(PAD_TOKEN)
    target_vocabulary = set(PAD_TOKEN)

    for input_text, target_text in zip(input_texts, target_texts):
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = [SEQ_START_TOKEN] + target_text + [SEQ_END_TOKEN]

        for token in input_text:
            if token not in input_vocabulary:
                input_vocabulary.add(token)
        for token in target_text:
            if token not in target_vocabulary:
                target_vocabulary.add(token)

    # compute core hyperparameters of the model

    input_vocabulary = sorted(list(input_vocabulary))
    target_vocabulary = sorted(list(target_vocabulary))
    num_encoder_tokens = len(input_vocabulary)
    num_decoder_tokens = len(target_vocabulary)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)

    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    encoder_seq_length = min(max_encoder_seq_length, args.max_input_length)
    decoder_seq_length = min(max_decoder_seq_length, args.max_output_length)

    input_token_index = dict(
        [(token, i) for i, token in enumerate(input_vocabulary)]
    )
    target_token_index = dict(
        [(token, i) for i, token in enumerate(target_vocabulary)]
    )

    # allocate tensors in the format accepted by the model for the training data
    encoder_input_data = np.zeros(
        (len(input_texts), encoder_seq_length, num_encoder_tokens),
        dtype='float32'
    )
    decoder_input_data = np.zeros(
        (len(input_texts), decoder_seq_length, num_decoder_tokens),
        dtype='float32'
    )
    decoder_target_data = np.zeros(
        (len(input_texts), decoder_seq_length, num_decoder_tokens),
        dtype='float32'
    )

    # encode the training data into tensors of numbers for the model
    # i is for sample_id
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        # process inputs...
        # t is for timestemp
        # take at most the first N tokens where N = encoder_seq_length
        for t, token in enumerate(input_text[:encoder_seq_length]):
            encoder_input_data[i, t, input_token_index[token]] = 1.

        # pad any remaining placeholders with the PAD_TOKEN
        encoder_input_data[i, t + 1:, input_token_index[PAD_TOKEN]] = 1.

        # process targets...
        # take at most the first N tokens where N = decoder_seq_length
        for t, token in enumerate(target_text[:decoder_seq_length]):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[token]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[token]] = 1.

        # pad any remaining placeholders with the PAD_TOKEN
        decoder_input_data[i, t + 1:, target_token_index[PAD_TOKEN]] = 1.
        decoder_target_data[i, t:, target_token_index[PAD_TOKEN]] = 1.

    # Define the Seq2Seq model in Keras
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(args.latent_dim, return_state=True)
    _encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(args.latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logdir = f'reports/logs/tensorboard/baseline-{timestamp}'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.2,
        callbacks=[tensorboard_callback]
    )

    # Save model
    model.save(args.file_model_output)

    # TODO: load the model for inference mode below

    # Inference mode / sampling
    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(args.latent_dim,))
    decoder_state_input_c = Input(shape=(args.latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_token_index = dict(
        (i, token) for token, i in input_token_index.items()
    )
    reverse_target_token_index = dict(
        (i, token) for token, i in target_token_index.items()
    )

    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states
    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index[SEQ_START_TOKEN]] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value
            )

            # Sample a token
            # TODO: why argmax?
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_token_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length or find stop character.
            # TODO: the name decoder_seq_length is misleading
            if (sampled_char == SEQ_END_TOKEN or len(decoded_sentence) > decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            # Feed in this sampled token into the decoder to sample the next token
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence


    for seq_index in range(100):
        # Take one sequence (part of the training set) for trying out decoding.
        input_seq = encoder_input_data[seq_index : seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)



def filter_out_string_literals(seq):
    return [token for token in seq if not token.startswith('"')]


def preprocess_data(args):
    df_path = os.path.join(args.dir_preprocessed_data, 'sequences.h5')
    input_vocab_path = os.path.join(args.dir_preprocessed_data, 'input_vocab_index.json')
    output_vocab_path = os.path.join(args.dir_preprocessed_data, 'output_vocab_index.json')

    files_exist = all(map(
        os.path.isfile,
        [df_path, input_vocab_path, output_vocab_path]
    ))

    print(f'files_exist: {files_exist}')

    if not files_exist:
        # Preprocess raw data
        df, input_vocab_index, output_vocab_index = preprocess_sequences(
            csv_filename=args.file_data_raw,
            max_input_seq_length=args.max_input_length,
            max_output_seq_length=args.max_output_length,
        )

        # Save preprocessed data
        os.makedirs(args.dir_preprocessed_data, exist_ok=True)

        df.to_hdf(df_path, key='data', mode='w')

        with open(input_vocab_path, 'w') as f:
            json.dump(input_vocab_index, f)

        with open(output_vocab_path, 'w') as f:
            json.dump(output_vocab_index, f)

    df = pd.read_hdf(df_path, key='data')

    return df


def get_dataset(df):
    dataset_inputs = tf.data.Dataset.from_tensor_slices(df['inputs'].values)
    dataset_outputs = tf.data.Dataset.from_tensor_slices(df['outputs'].values)

    return tf.data.Dataset.zip((dataset_inputs, dataset_outputs))

def main():
    args = parser.parse_args()
    # TODO: persist configuration in experiment folter

    df = preprocess_data(args)

    dataset = get_dataset(df)

    return # TODO: REMOVEME

    df = pd.read_csv(args.file_data_raw).dropna().head(1000)
    print(f'loaded dataset of size {len(df)}')

    # dataset
    df['body_tokens'] = df['body'] \
        .progress_apply(tokenize_method_body) \
        .progress_apply(filter_out_string_literals) # ignore string literals, because they increase the vocabulary size too much
    input_texts = df[df.body_tokens.str.len() > 0]['body_tokens'] # remove invalid methods which cannot be parsed

    # the output sequences are marked with a <start> and <end> special tokens
    target_texts = df['method_name'].progress_apply(get_subtokens)

    # print(input_texts.head(5))
    # print('-------')
    # print(target_texts.head(5))
    seq2seq(args, input_texts, target_texts)


if __name__ == "__main__":
    main()
