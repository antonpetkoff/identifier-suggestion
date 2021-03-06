import argparse
import json
import os
import sys
import resource
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from itertools import takewhile

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense

# custom imports
from src.utils.logger import Logger
from src.utils.random import set_random_seeds
from src.evaluation.sequence import compute_f1_score
from src.preprocessing.tokens import tokenize_method_body, get_subtokens
from src.preprocessing.sequence import preprocess_sequences
from src.visualization.plot import plot_attention_weights
from src.utils.strings import seq_to_camel_case
from src.common.tokens import Common

from src.models.seq2seq import Seq2Seq

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
parser.add_argument('--dir_data', type=str, help='Directory of the dataset', required=True)
parser.add_argument('--file_checkpoint_dir', type=str, help='Model checkpoint directory name', required=True)
parser.add_argument('--dir_preprocessed_data', type=str, help='Directory for preprocessed data', required=True)
parser.add_argument('--experiment_name', type=str, help='Name of the experiment', required=True)

# hyper parameters
parser.add_argument('--max_input_length', type=int, help='Max input sequence length', required=True)
parser.add_argument('--max_output_length', type=int, help='Max output sequence length', required=True)
parser.add_argument('--input_vocab_size', type=int, help='Input vocabulary size', required=True)
parser.add_argument('--input_embedding_dim', type=int, help='Input embedding dimensionality', required=True)

parser.add_argument('--output_vocab_size', type=int, help='Output vocabulary size', required=True)
parser.add_argument('--output_embedding_dim', type=int, help='Output embedding dimensionality', required=True)
parser.add_argument('--latent_dim', type=int, help='Encoder-Decoder latent space dimensionality', required=True)
parser.add_argument('--learning_rate', type=float, help='Learning Rate - a non-negative float, e.g. 0.001, 0.01, etc.', required=True)

parser.add_argument('--dropout_rate', type=float, help='Dropout rate of LSTM linear activations. Varies from 0.0 to 1.0, where 0.0 means no dropout', required=True)
parser.add_argument('--evaluation_dataset', choices=['validation', 'test'], help='Type of dataset to use for evaluation. Can be: test or validation', required=True)
parser.add_argument('--epochs', type=int, help='Maximum number of training epochs', required=True)
parser.add_argument('--early_stopping_patience', type=int, help='Maximum number of epochs without improvement before stopping training', required=True)
parser.add_argument('--early_stopping_min_delta', type=float, help='Minimum amount of improvement required in the score for early stopping', required=True)
parser.add_argument('--batch_size', type=int, help='Batch Size', required=True)
parser.add_argument('--random_seed', type=int, help='Random Seed', required=True)


def preprocess_data(args, logger):
    df_train_path = os.path.join(args.dir_preprocessed_data, 'sequences.train.h5')
    df_validation_path = os.path.join(args.dir_preprocessed_data, 'sequences.validation.h5')
    df_test_path = os.path.join(args.dir_preprocessed_data, 'sequences.test.h5')
    input_vocab_path = os.path.join(args.dir_preprocessed_data, 'input_vocab_index.json')
    output_vocab_path = os.path.join(args.dir_preprocessed_data, 'output_vocab_index.json')

    files_exist = all(map(
        os.path.isfile,
        [df_train_path, df_validation_path, df_test_path, input_vocab_path, output_vocab_path]
    ))

    if not files_exist:
        logger.log_message('Preprocessed files not found. Preprocessing...')
        # Preprocess raw data
        df_train, df_validation, df_test, input_vocab_index, output_vocab_index = preprocess_sequences(
            dir_data=args.dir_data,
            max_input_seq_length=args.max_input_length,
            max_output_seq_length=args.max_output_length,
            max_input_vocab_size=args.input_vocab_size,
            max_output_vocab_size=args.output_vocab_size,
        )

        logger.log_message('Done preprocessing. Saving...')

        # Save preprocessed data
        os.makedirs(args.dir_preprocessed_data, exist_ok=True)

        df_train.to_hdf(df_train_path, key='data', mode='w')
        df_validation.to_hdf(df_validation_path, key='data', mode='w')
        df_test.to_hdf(df_test_path, key='data', mode='w')

        with open(input_vocab_path, 'w') as f:
            json.dump(input_vocab_index, f)

        with open(output_vocab_path, 'w') as f:
            json.dump(output_vocab_index, f)

    logger.log_message('Loading preprocessed files...')

    with open(input_vocab_path) as f:
        input_vocab_index = json.load(f)

    logger.log_message('Loaded input vocabulary.')

    with open(output_vocab_path) as f:
        output_vocab_index = json.load(f)

    logger.log_message('Loaded output vocabulary.')

    df_train = pd.read_hdf(df_train_path, key='data')
    df_validation = pd.read_hdf(df_validation_path, key='data')
    df_test = pd.read_hdf(df_test_path, key='data')

    logger.log_message('Loaded preprocessed files.')

    return df_train, df_validation, df_test, input_vocab_index, output_vocab_index


def run(args):
    os.makedirs(args.file_checkpoint_dir, exist_ok=True)
    os.makedirs('./reports', exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    logger = Logger(
        experiment_config = args,
        wandb_save_dir = './reports',
        image_save_dir = f'./reports/figures/train-{timestamp}',
        data_save_dir = f'./reports/dumps/train-{timestamp}',
    )

    logger.log_message('Experiment parameters: ', args)

    df_train, df_validation, df_test, input_vocab_index, output_vocab_index = preprocess_data(args, logger)

    evaluation_dataset = df_test if args.evaluation_dataset == 'test' else df_validation

    model = Seq2Seq(
        logger=logger,
        checkpoint_dir=args.file_checkpoint_dir,
        input_vocab_index=input_vocab_index,
        output_vocab_index=output_vocab_index,
        max_input_seq_length=args.max_input_length,
        max_output_seq_length=args.max_output_length,
        input_vocab_size=args.input_vocab_size,
        output_vocab_size=args.output_vocab_size,
        input_embedding_dim=args.input_embedding_dim,
        output_embedding_dim=args.output_embedding_dim,
        rnn_units=args.latent_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )

    model.summary()

    model.save()

    # TODO: extract this evaluation logic as a callback
    reverse_input_index = dict(
        (i, token) for token, i in input_vocab_index.items()
    )
    reverse_output_index = dict(
        (i, token) for token, i in output_vocab_index.items()
    )

    test_samples = df_train.sample(10)
    test_inputs = np.stack(test_samples['inputs'])
    test_outputs = np.stack(test_samples['outputs'])

    # convert raw inputs to human-readable text strings
    input_texts = [
        ' '.join([
            reverse_input_index.get(index, Common.OOV)
            for index in test_input
            if index != 0 # without padding
        ])
        for test_input in test_inputs
    ]

    def on_epoch_end(epoch):
        predicted_texts = []

        for sample_id, test_input in enumerate(test_inputs):
            predicted_token_ids, attention_weights = model.predict_raw(input_sequence = test_input)

            input_tokens = [
                reverse_input_index.get(index, Common.OOV)
                for index in test_input
                if index != 0
            ]

            eos_key = output_vocab_index[Common.EOS]
            index_of_first_end_of_seq = predicted_token_ids.index(eos_key) if eos_key in predicted_token_ids else (len(predicted_token_ids) - 1)
            output_tokens = [
                reverse_output_index.get(token_id, Common.OOV)
                for token_id in predicted_token_ids[:(index_of_first_end_of_seq + 1)]
            ]

            logger.log_attention_heatmap(
                attention_weights.numpy(),
                input_tokens,
                output_tokens,
                save_name = f'id-{sample_id}-epoch-{epoch}',
                save_to_wandb = False, # temporarily disable wandb heatmaps until W&B fix the dashboards
            )

            predicted_texts.append(
                '' if len(output_tokens) == 1 else ''.join(seq_to_camel_case(output_tokens[:-1])) # without the <eos> marker
            )

        # log tables
        expected_texts = [
            model.convert_raw_prediction_to_text(seq[1:]) # ignore <sos> marker in testing data
            for seq in test_outputs
        ]
        logger.log_examples_table(input_texts, predicted_texts, expected_texts)

    model.train(
        X_train=np.stack(df_train['inputs'].values),
        Y_train=np.stack(df_train['outputs'].values),
        X_test=np.stack(evaluation_dataset['inputs'].values),
        Y_test=np.stack(evaluation_dataset['outputs'].values),
        epochs=args.epochs,
        on_epoch_end=on_epoch_end,
    )

    model.save()


if __name__ == '__main__':
    limit_memory() # limit maximun memory usage to half
    try:
        args = parser.parse_args()
        run(args)
    except MemoryError:
        sys.stderr.write('\nERROR: Memory Limit Exception\n')
        sys.exit(1)
