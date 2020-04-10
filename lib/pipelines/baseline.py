import argparse
import pandas as pd

from lib.evaluation.sequence import compute_f1_score

parser = argparse.ArgumentParser(description='Baseline Seq2Seq model')
parser.add_argument('--file_data_raw', type=str, help='Raw data file', required=True)

def main():
    args = parser.parse_args()
    print(args)

    df = pd.read_csv(args.file_data_raw).dropna().head(1000)
    print(f'loaded dataset of size {len(df)}')

    f1_score = compute_f1_score('transformSearchResponse', 'modifySearchResponseData')
    print(f'F1 score is {f1_score}')



if __name__ == "__main__":
    main()
