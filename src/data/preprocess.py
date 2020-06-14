import argparse
import os

import multiprocessing as mp
import numpy as np
import pandas as pd

from src.preprocessing.tokens import tokenize_method, split_subtokens
from src.utils.random import set_random_seeds

# import tqdm and enable it for pandas for progress_apply
from tqdm import tqdm
tqdm.pandas()

RANDOM_SEED = 1
set_random_seeds(RANDOM_SEED)


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description='Data set preprocessing - cleaning, tokenization, splitting, shuffling'
    )

    parser.add_argument(
        '--dir_input_data',
        type=str,
        help='Raw data directory with a CSV for each repository',
        required=True
    )

    parser.add_argument(
        '--dir_output',
        type=str,
        help='Target directory where the preprocessed data will be stored',
        required=True
    )

    parser.add_argument(
        '--num_jobs',
        type=int,
        help='The number of parallel jobs',
        default=4,
        required=True
    )

    parser.add_argument(
        '--tokenization_level',
        type=str,
        help='The tokenization level to use - character, subtoken or token',
        required=True
    )

    return parser


def split_train_test_val(df):
    # The split is shuffled

    # Split the data set into 80% for training and the remaining 20% for testing and validation
    training_mask = np.random.rand(len(df)) < 0.8
    training = df[training_mask]
    remaining = df[~training_mask]

    # split the remaining 20% into equal sets for validation and testing
    validation_mask = np.random.rand(len(remaining)) < 0.5
    validation = remaining[validation_mask]
    testing = remaining[~validation_mask]

    return training, validation, testing


def preprocess_repository(csv_filename):
    print(f'Preprocessing {csv_filename}')

    # Reading the input files
    df = pd.read_csv(csv_filename)

    # Cleaning, filtering the data
    df = df.dropna()

    print(f'Tokenizing {csv_filename}')

    # TODO: choose tokenization method
    # Tokenize and filter input sequences
    df['tokenized_body'] = df['body'].progress_apply(tokenize_method)

    # Tokenize output sequences
    df['tokenized_method_name'] = df['method_name'].progress_apply(split_subtokens)

    # Filter out the samples which cannot be tokenized
    df = df[df.tokenized_body.str.len() > 0]

    # Drop the original columns
    df.drop(['body', 'method_name'], axis=1, inplace=True)

    print(f'Splitting {csv_filename}')

    return split_train_test_val(df)


def combine_and_shuffle(subsets):
    # subsets is a list of dataframes

    # combine the rows from all subsets (i.e. from all repositories) into a single data frame
    df = pd.concat(subsets, axis=0)

    # shuffle the data frame in place without re-indexing it
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def preprocess_data(args):
    repositories_dir = args.dir_input_data
    repo_csv_files = os.listdir(repositories_dir)

    job_specs = map(
        # wrap file names into tuples, because they are passed as argument lists
        lambda repo: (os.path.join(repositories_dir, repo),),
        repo_csv_files
    )

    print('Preprocessing the repository CSV files: ', repo_csv_files)

    subsets = [] # a list of triples (training, validation, testing) sets
    with mp.Pool(args.num_jobs) as pool:
        subsets = pool.starmap(preprocess_repository, job_specs)

    print('Done preprocessing each repository')

    save_dir = os.path.join(args.dir_output, args.tokenization_level)
    os.makedirs(save_dir, exist_ok=True)

    for dataset_id, dataset_type in enumerate(['train', 'validation', 'test']):
        selected_subsets = map(lambda triplet: triplet[dataset_id], subsets)

        print(f'Combining and shuffling {dataset_type} dataset')

        df = combine_and_shuffle(selected_subsets)

        save_path = os.path.join(save_dir, f'{dataset_type}.h5')

        print(f'Saving {save_path}')

        # save the dataframe
        df.to_hdf(save_path, key='data', mode='w')

    print('Done preprocessing')


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    preprocess_data(args)


if __name__ == '__main__':
    main()