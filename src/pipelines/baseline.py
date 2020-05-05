import argparse
import json
import os
import sys
import resource
import wandb
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

from src.models.seq2seq_attention import Seq2SeqAttention

# import tqdm and enable it for pandas for progress_apply
from tqdm import tqdm
tqdm.pandas()

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(verbose=True)


RANDOM_SEED = 1

set_random_seeds(RANDOM_SEED)


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def limit_memory():
    ALLOWED_FREE_MEMORY = 0.8 # percentage of the free memory to allow for usage
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard_memory_limit = get_memory() * 1024 * ALLOWED_FREE_MEMORY
    print(f'Setting new hard memory limit to: {new_hard_memory_limit}')
    resource.setrlimit(resource.RLIMIT_AS, (new_hard_memory_limit, hard))


parser = argparse.ArgumentParser(description='Baseline Seq2Seq model')

# data files
parser.add_argument('--file_data_raw', type=str, help='Raw data file used for model training', required=True)
parser.add_argument('--file_model_dir', type=str, help='Model output directory name', required=True)
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


def preprocess_data(args):
    df_path = os.path.join(args.dir_preprocessed_data, 'sequences.h5')
    input_vocab_path = os.path.join(args.dir_preprocessed_data, 'input_vocab_index.json')
    output_vocab_path = os.path.join(args.dir_preprocessed_data, 'output_vocab_index.json')

    files_exist = all(map(
        os.path.isfile,
        [df_path, input_vocab_path, output_vocab_path]
    ))

    if not files_exist:
        print('Preprocessed files not found. Preprocessing...')
        # Preprocess raw data
        df, input_vocab_index, output_vocab_index = preprocess_sequences(
            csv_filename=args.file_data_raw,
            max_input_seq_length=args.max_input_length,
            max_output_seq_length=args.max_output_length,
            max_input_vocab_size=args.input_vocab_size,
            max_output_vocab_size=args.output_vocab_size,
            random_seed=RANDOM_SEED,
        )

        print('Done preprocessing. Saving...')

        # Save preprocessed data
        os.makedirs(args.dir_preprocessed_data, exist_ok=True)

        df.to_hdf(df_path, key='data', mode='w')

        with open(input_vocab_path, 'w') as f:
            json.dump(input_vocab_index, f)

        with open(output_vocab_path, 'w') as f:
            json.dump(output_vocab_index, f)

    print('Loading preprocessed files...')

    with open(input_vocab_path) as f:
        input_vocab_index = json.load(f)

    print('Loaded input vocabulary.')

    with open(output_vocab_path) as f:
        output_vocab_index = json.load(f)

    print('Loaded output vocabulary.')

    df = pd.read_hdf(df_path, key='data')

    print('Loaded preprocessed files.')

    return df, input_vocab_index, output_vocab_index


def run(args):
    print('Experiment parameters: ', args)

    os.makedirs('./reports', exist_ok=True)
    wandb.init(dir='./reports', config=args)

    # TODO: persist configuration in experiment folter

    df, _input_vocab_index, output_vocab_index = preprocess_data(args)

    model = Seq2SeqAttention(
        max_input_seq_length=args.max_input_length,
        max_output_seq_length=args.max_output_length,
        input_vocab_size=args.input_vocab_size,
        output_vocab_size=args.output_vocab_size,
        input_embedding_dim=args.input_embedding_dim,
        output_embedding_dim=args.output_embedding_dim,
        rnn_units=args.latent_dim,
        dense_units=args.latent_dim, # TODO: expose as a hyper parameter
        batch_size=args.batch_size,
    )

    # TODO: expose callback and use the model to predict a sample of 10 sequences.
    # TODO: Log the predictions as text tables to observe the progress of the training

    model.train(
        X_train=np.stack(df['inputs'].values),
        Y_train=np.stack(df['outputs'].values),
        epochs=args.epochs
    )

    model.save(save_dir=args.file_model_dir)

    test_inputs = np.stack(df['inputs'].head(2))

    print('Test inputs: ', test_inputs)

    predictions = model.predict(
        input_sequences=test_inputs,
        start_token_index=output_vocab_index['<SOS>'],
        end_token_index=output_vocab_index['<EOS>'],
    )

    print('Predictions: ', predictions)


if __name__ == '__main__':
    limit_memory() # limit maximun memory usage to half
    try:
        args = parser.parse_args()
        run(args)
    except MemoryError:
        sys.stderr.write('\nERROR: Memory Limit Exception\n')
        sys.exit(1)
