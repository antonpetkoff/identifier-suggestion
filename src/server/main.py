import argparse
import os
import json

from datetime import datetime

from flask import Flask
from flask import request

from src.models.seq2seq import Seq2Seq
from src.visualization.plot import plot_attention_weights
from src.utils.logger import Logger


parser = argparse.ArgumentParser(description='Seq2Seq model server')

# data files
parser.add_argument('--file_checkpoint_dir', type=str, help='Model output directory name', required=True)
parser.add_argument('--vocab_path', type=str, help='Directory with input and output vocabularies', required=True)


def load_vocabularies(vocab_path):
    input_vocab_path = os.path.join(vocab_path, 'input_vocab_index.json')
    output_vocab_path = os.path.join(vocab_path, 'output_vocab_index.json')

    with open(input_vocab_path) as f:
        input_vocab_index = json.load(f)

    print('Loaded input vocabulary.')

    with open(output_vocab_path) as f:
        output_vocab_index = json.load(f)

    print('Loaded output vocabulary.')

    return input_vocab_index, output_vocab_index


def get_current_timestamp():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def initialize_model(args):
    input_vocab_index, output_vocab_index = load_vocabularies(
        vocab_path=args.vocab_path
    )

    timestamp = get_current_timestamp()

    logger = Logger(
        experiment_config = args,
        wandb_save_dir = None,
        image_save_dir = f'./reports/figures/serve-{timestamp}'
    )

    model = Seq2Seq.restore(
        args.file_checkpoint_dir,
        logger,
        input_vocab_index,
        output_vocab_index
    )

    return model, logger


args = parser.parse_args()
model, logger = initialize_model(args)
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/predict')
def predict():
    input_text = request.args.get('input')

    print(f'input_text: {input_text}')

    prediction, attention_weights, input_tokens, output_tokens = model.predict(input_text)

    print('prediction: ', prediction)

    # log attention weights for interpretability
    # logger.log_attention_heatmap(
    #     attention_weights,
    #     input_tokens,
    #     output_tokens,
    #     save_name=get_current_timestamp()
    # )

    predictions = model.predict_beam_search(input_text=input_text)

    print(f'predictions: {predictions}')

    return {'predictions': predictions}


if __name__ == '__main__':
    app.run()
