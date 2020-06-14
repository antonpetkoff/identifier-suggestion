import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# import tqdm and enable it for pandas for progress_apply
from tqdm import tqdm
tqdm.pandas()

from src.preprocessing.tokens import tokenize_method, split_subtokens
from src.utils.pandas import lists_to_series


OOV_TOKEN = '<oov>'
SEQ_START_TOKEN = '<sos>'
SEQ_END_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'


def preprocess_sequences(
    csv_filename,
    max_input_seq_length = 200,
    max_output_seq_length = 8,
    max_input_vocab_size = 30000,
    max_output_vocab_size = 10000,
    random_seed = 1,
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
    df = pd.read_csv(csv_filename)

    # Cleaning, filtering the data
    df = df.dropna()

    print('Tokenizing input method bodies')
    # Tokenize and filter input sequences
    df['inputs'] = df['body'].progress_apply(tokenize_method)

    print('Tokenizing output sequences')
    # Tokenize output sequences
    df['outputs'] = df['method_name'].progress_apply(split_subtokens)

    # Filter out the samples which cannot be tokenized
    df = df[df.inputs.str.len() > 0]

    # Keep only the inputs and outputs
    df = df[['inputs', 'outputs']]

    # Ensure output sequences are at most max_output_seq_length long
    # and annotate output sequences with <sos> and <eos>
    print('Adding <start> and <end> markers to output sequences')
    df['outputs'] = df['outputs'].progress_apply(
        lambda seq: [SEQ_START_TOKEN] + seq[:max_output_seq_length] + [SEQ_END_TOKEN]
    )

    def get_vocab_index(df, max_vocab_size):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=max_vocab_size,
            filters='',
            lower=False,
            oov_token=OOV_TOKEN,
        )

        tokenizer.fit_on_texts(df.values)

        return tokenizer

    # Build vocabularies and their indices
    print('Building input vocabulary')
    input_tokenizer = get_vocab_index(
        df['inputs'],
        max_vocab_size=max_input_vocab_size
    )
    input_vocab_index = input_tokenizer.word_index

    print('Building output vocabulary')
    output_tokenizer = get_vocab_index(
        df['outputs'],
        max_vocab_size=max_output_vocab_size
    )
    output_vocab_index = output_tokenizer.word_index

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

    print('Shuffling the final dataset')
    # shuffle the samples so that we don't have only unit tests at the beginning
    shuffle(df, random_state=random_seed)

    print('Splitting the data into train/validation/test datasets')

    X_train, X_test, y_train, y_test = train_test_split(
        df['inputs'],
        df['outputs'],
        test_size=0.2,
        random_state=random_seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25, # 0.25 x 0.8 = 0.2
        random_state=random_seed
    )

    df_train = pd.DataFrame({
        'inputs': X_train,
        'outputs': y_train,
    })

    df_validation = pd.DataFrame({
        'inputs': X_val,
        'outputs': y_val,
    })

    df_test = pd.DataFrame({
        'inputs': X_test,
        'outputs': y_test,
    })

    print('Done preprocessing')

    # TODO: write tests which ensure that we have correctly formatted the preprocessed data

    return df_train, df_validation, df_test, input_vocab_index, output_vocab_index
