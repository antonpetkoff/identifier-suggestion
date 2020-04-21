import pandas as pd

# import tqdm and enable it for pandas for progress_apply
from tqdm import tqdm
tqdm.pandas()

from src.preprocessing.tokens import tokenize_method_body, get_subtokens
from src.utils.pandas import lists_to_series

SEQ_START_TOKEN = '<SOS>'
SEQ_END_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'
STRING_LITERAL_TOKEN = '<STR>'


def replace_string_literals(seq):
    return [
        STRING_LITERAL_TOKEN if token.startswith('"') else token
        for token in seq
    ]


def preprocess_sequences(
    csv_filename,
    max_output_seq_length = 8,
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
        max_output_seq_length (int): Maximum output sequence length.

    Returns:
        A pandas data frame with the preprocessed input and output sequences.
    """

    # Reading the input files
    df = pd.read_csv(csv_filename).head(1000) # TODO: remove head(1000)

    # Cleaning, filtering the data
    df = df.dropna()

    # Tokenize and filter input sequences
    df['inputs'] = df['body'] \
        .progress_apply(tokenize_method_body) \
        .progress_apply(replace_string_literals) # ignore string literals, because they increase the vocabulary size too much

    # Tokenize output sequences
    df['outputs'] = df['method_name'].progress_apply(get_subtokens)

    # Filter out the samples which cannot be tokenized
    df = df[df.inputs.str.len() > 0]

    # Keep only the inputs and outputs
    df = df[['inputs', 'outputs']]

    # Ensure output sequences are at most max_output_seq_length long
    # and annotate output sequences with <SOS> and <EOS>
    df['outputs'] = df['outputs'].progress_apply(
        lambda seq: [SEQ_START_TOKEN] + seq[:max_output_seq_length] + [SEQ_END_TOKEN]
    )

    # Build vocabularies and their indices
    input_vocab = set(lists_to_series(df['inputs'].values).unique())
    input_vocab.add(PAD_TOKEN)
    input_vocab_index = { token: index for index, token in enumerate(input_vocab) }

    output_vocab = set(lists_to_series(df['inputs'].values).unique())
    output_vocab.add(PAD_TOKEN)
    output_vocab_index = { token: index for index, token in enumerate(output_vocab) }

    # TODO: save to vocabularies to JSON now?

    print(output_vocab_index)

    # TODO: write tests which ensure that we have correctly formatted the preprocessed data

    # TODO: the output sequences are marked with a <start> and <end> special tokens

    print(len(df), df.head())

    return df
