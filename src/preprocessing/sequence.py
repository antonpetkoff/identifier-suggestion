import pandas as pd

# import tqdm and enable it for pandas for progress_apply
from tqdm import tqdm
tqdm.pandas()

SEQ_START_TOKEN = '<SOS>'
SEQ_END_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'

def preprocess_sequences(csv_filename):
    """This function preprocesses input and output sequences for seq2seq models.

    The preprocessing steps include:
    - Reading the input file
    - Cleaning/filtering of empty samples
    - Tokenization of the sequences
    - Marking the sequences with special <SOS>, <EOS> tokens
    - Building a vocabularies
        - Including <SOS>, <EOS>, <PAD> special tokens
        - Saving these vocabularies to files
    - Encoding the tokens to numbers, based on the vocabularies
    - Pad, align and cut, using <PAD> special tokens
    - Save final sequences to binary data files

    Args:
        csv_filename (str): CSV filename which includes the raw data.

    Returns:
        A pandas data frame with the preprocessed input and output sequences.
    """
    df = pd.read_csv(csv_filename)

    print(df.head())

    return df
