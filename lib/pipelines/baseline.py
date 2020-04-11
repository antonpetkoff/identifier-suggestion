import argparse
import pandas as pd
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

from tqdm import tqdm

from lib.evaluation.sequence import compute_f1_score
from lib.preprocessing.tokens import tokenize_method_body, get_subtokens

tqdm.pandas()

parser = argparse.ArgumentParser(description='Baseline Seq2Seq model')

# data files
parser.add_argument('--file_data_raw', type=str, help='Raw data file', required=True)

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


SEQ_START = '\t'
SEQ_END = '\n'


def seq2seq(args, input_texts, target_texts):
    # Vectorize the data.
    input_vocabulary = set()
    target_vocabulary = set()

    for input_text, target_text in zip(input_texts, target_texts):
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = [SEQ_START] + target_text + [SEQ_END]

        for token in input_text:
            # ignore string literals, because they increase the vocabulary size too much
            if not token.startswith('"') and token not in input_vocabulary:
                input_vocabulary.add(token)
        for token in target_text:
            if token not in target_vocabulary:
                target_vocabulary.add(token)

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

    print(target_token_index)

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


def main():
    args = parser.parse_args()
    #print(args) # TODO: persist configuration in experiment folter

    df = pd.read_csv(args.file_data_raw).dropna().head(1000)
    print(f'loaded dataset of size {len(df)}')

    # dataset
    df['body_tokens'] = df['body'].progress_apply(tokenize_method_body)
    input_texts = df[df.body_tokens.str.len() > 0]['body_tokens'] # remove invalid methods which cannot be parsed

    # the output sequences are marked with a <start> and <end> special tokens
    target_texts = df['method_name'].progress_apply(get_subtokens)

    # print(input_texts.head(5))
    # print('-------')
    # print(target_texts.head(5))
    seq2seq(args, input_texts, target_texts)


if __name__ == "__main__":
    main()
