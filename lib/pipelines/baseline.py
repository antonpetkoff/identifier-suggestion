import argparse
import pandas as pd

from lib.evaluation.sequence import compute_f1_score

parser = argparse.ArgumentParser(description='Baseline Seq2Seq model')

# data files
parser.add_argument('--file_data_raw', type=str, help='Raw data file', required=True)

# hyper parameters
parser.add_argument('--max_input_length', type=int, help='Max input sequence length', required=True)
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


def main():
    args = parser.parse_args()
    print(args)

    df = pd.read_csv(args.file_data_raw).dropna().head(1000)
    print(f'loaded dataset of size {len(df)}')

    # TODO: preprocess data from raw to model format?
    # TODO: use data iterator

    f1_score = compute_f1_score('transformSearchResponse', 'modifySearchResponseData')
    print(f'F1 score is {f1_score}')


if __name__ == "__main__":
    main()
