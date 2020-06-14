import os

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# import tqdm and enable it for pandas for progress_apply
from tqdm import tqdm
tqdm.pandas()

from src.common.tokens import Common
from src.preprocessing.tokens import tokenize_method, split_subtokens
from src.utils.pandas import lists_to_series


def preprocess_sequences(
    dir_data,
    max_input_seq_length = 200,
    max_output_seq_length = 8,
    max_input_vocab_size = 20000,
    max_output_vocab_size = 15000,
):
    """This function preprocesses input and output sequences for seq2seq models.

    The preprocessing steps include:
    - Reading the input file
    - Cleaning/filtering of empty samples
    - Tokenization of the sequences
    - Marking the sequences with special <sos>, <eos> tokens
    - Building vocabularies
        - Including <sos>, <eos>, <pad> special tokens
        - Saving these vocabularies to files
    - Encoding the tokens to numbers, based on the vocabularies
    - Pad, align and cut, using <pad> special tokens
    - Save final sequences to binary data files

    Args:
        csv_filename (str): CSV filename which includes the raw data.
        max_input_seq_length (int): Maximum input sequence length.
        max_output_seq_length (int): Maximum output sequence length.

    Returns:
        A pandas data frame with the preprocessed inputs and outputs sequences
        in columns with the names 'inputs' and 'outputs'.
        Also the input and output vocabulary indices.
    """

    print('Reading input files')
    # Reading the input files
    df_train = pd.read_hdf(os.path.join(dir_data, 'train.h5'), key='data')
    df_validation = pd.read_hdf(os.path.join(dir_data, 'validation.h5'), key='data')
    df_test = pd.read_hdf(os.path.join(dir_data, 'test.h5'), key='data')

    def pick_columns_and_mark_outputs(df):
        # Keep only the inputs and outputs
        df = df[['tokenized_body', 'tokenized_method_name']]

        # Rename columns
        df = df.rename(columns={'tokenized_body': 'inputs', 'tokenized_method_name': 'outputs'})

        # Ensure output sequences are at most max_output_seq_length long
        # and annotate output sequences with <sos> and <eos>
        print('Adding <start> and <end> markers to output sequences')
        df['outputs'] = df['outputs'].progress_apply(
            lambda seq: [Common.SOS] + seq[:max_output_seq_length] + [Common.EOS]
        )

        return df

    df_train = pick_columns_and_mark_outputs(df_train)
    df_validation = pick_columns_and_mark_outputs(df_validation)
    df_test = pick_columns_and_mark_outputs(df_test)

    # we need to combine the rows (samples) from all data set splits
    # so that the full vocabularies can be built
    df_combined = pd.concat([df_train, df_validation, df_test], axis=0)

    def get_vocab_index(df, max_vocab_size):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=max_vocab_size,
            filters='',
            lower=False,
            oov_token=Common.OOV,
        )

        tokenizer.fit_on_texts(df.values)

        return tokenizer

    # Build vocabularies and their indices
    print('Building input vocabulary')
    input_tokenizer = get_vocab_index(
        df_combined['inputs'],
        max_vocab_size=max_input_vocab_size
    )
    input_vocab_index = input_tokenizer.word_index

    print('Building output vocabulary')
    output_tokenizer = get_vocab_index(
        df_combined['outputs'],
        max_vocab_size=max_output_vocab_size
    )
    output_vocab_index = output_tokenizer.word_index

    def encode_and_pad(df):
        print('Encoding input sequences into numbers')
        # TODO: can tokenizer.texts_to_sequences be applied on all samples at once?
        # Encode sequences to numbers
        df['inputs'] = df['inputs'].progress_apply(
            lambda seq: np.concatenate(input_tokenizer.texts_to_sequences(seq))
        )

        print('inputs after tokenizer', df['inputs'].head(3))

        print('Encoding output sequences into numbers')
        df['outputs'] = df['outputs'].progress_apply(
            lambda seq: np.concatenate(output_tokenizer.texts_to_sequences(seq))
        )

        print('outputs after tokenizer', df['outputs'].head(3))

        print('Padding and aligning input sequences')
        # Pad and align sequences
        df['inputs'] = tf.keras.preprocessing.sequence.pad_sequences(
            df['inputs'],
            maxlen=max_input_seq_length,
            truncating='post',
            padding='post',
            value=0, # index of PAD token
            dtype='int32',
        ).tolist()

        print('inputs after padding', df['inputs'].head(3))

        print('Padding and aligning output sequences')
        df['outputs'] = tf.keras.preprocessing.sequence.pad_sequences(
            df['outputs'],
            maxlen=max_output_seq_length,
            truncating='post',
            padding='post',
            value=0, # index of PAD token
            dtype='int32',
        ).tolist()

        print('Outputs after padding: ', df['outputs'].head(3))

        return df

    print('Encoding and padding training data...')
    df_train = encode_and_pad(df_train)

    print('Encoding and padding validation data...')
    df_validation = encode_and_pad(df_validation)

    print('Encoding and padding testing data...')
    df_test = encode_and_pad(df_test)

    print('Done preprocessing')

    # TODO: write tests which ensure that we have correctly formatted the preprocessed data

    return df_train, df_validation, df_test, input_vocab_index, output_vocab_index
