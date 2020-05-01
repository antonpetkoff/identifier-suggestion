import pandas as pd
import tensorflow as tf

# import tqdm and enable it for pandas for progress_apply
from tqdm import tqdm
tqdm.pandas()

from src.preprocessing.tokens import tokenize_method, split_subtokens
from src.utils.pandas import lists_to_series


OOV_TOKEN = '<OOV>'
SEQ_START_TOKEN = '<SOS>'
SEQ_END_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'


def preprocess_sequences(
    csv_filename,
    max_input_seq_length = 200,
    max_output_seq_length = 8,
    max_input_vocab_size = 30000,
    max_output_vocab_size = 10000,
):
    """This function preprocesses input and output sequences for seq2seq models.

    The preprocessing steps include:
    - Reading the input file
    - Cleaning/filtering of empty samples
    - Tokenization of the sequences
    - Marking the sequences with special <SOS>, <EOS> tokens
    - Building vocabularies
        - Including <SOS>, <EOS>, <PAD> special tokens
        - Saving these vocabularies to files
    - Encoding the tokens to numbers, based on the vocabularies
    - Pad, align and cut, using <PAD> special tokens
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

    # Reading the input files
    df = pd.read_csv(csv_filename)

    # TODO: REMOVE the head(1000)
    # Cleaning, filtering the data
    df = df.dropna().head(1000)

    # Tokenize and filter input sequences
    df['inputs'] = df['body'].progress_apply(tokenize_method)

    # Tokenize output sequences
    df['outputs'] = df['method_name'].progress_apply(split_subtokens)

    # Filter out the samples which cannot be tokenized
    df = df[df.inputs.str.len() > 0]

    # Keep only the inputs and outputs
    df = df[['inputs', 'outputs']]

    # Ensure output sequences are at most max_output_seq_length long
    # and annotate output sequences with <SOS> and <EOS>
    df['outputs'] = df['outputs'].progress_apply(
        lambda seq: [SEQ_START_TOKEN] + seq[:max_output_seq_length] + [SEQ_END_TOKEN]
    )

    def get_vocab_index(df, max_vocab_size):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=max_vocab_size,
            filters='',
            lower=False,
            oov_token='<OOV>',
        )

        tokenizer.fit_on_texts(df.values)

        return tokenizer.word_index

    # Build vocabularies and their indices
    input_vocab_index = get_vocab_index(
        df['inputs'],
        max_vocab_size=max_input_vocab_size
    )
    output_vocab_index = get_vocab_index(
        df['outputs'],
        max_vocab_size=max_output_vocab_size
    )

    # TODO: save to vocabularies to JSON now?

    # TODO: use tokenizer fit_on_sequences
    # Encode sequences to numbers
    df['inputs'] = df['inputs'].progress_apply(
        lambda seq: [input_vocab_index[token] for token in seq]
    )

    df['outputs'] = df['outputs'].progress_apply(
        lambda seq: [output_vocab_index[token] for token in seq]
    )

    # Pad and align sequences
    df['inputs'] = tf.keras.preprocessing.sequence.pad_sequences(
        df['inputs'],
        maxlen=max_input_seq_length,
        truncating='post',
        padding='post',
        value=0, # index of PAD token
        dtype='int32',
    ).tolist()

    print('inputs', df['inputs'].head(5))

    df['outputs'] = tf.keras.preprocessing.sequence.pad_sequences(
        df['outputs'],
        maxlen=max_output_seq_length,
        truncating='post',
        padding='post',
        value=0, # index of PAD token
        dtype='int32',
    ).tolist()

    print('outputs', df['outputs'].head(5))

    # TODO: write tests which ensure that we have correctly formatted the preprocessed data

    return df, input_vocab_index, output_vocab_index
