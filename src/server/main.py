import argparse
import os
import json

from flask import Flask
from flask import request

from src.models.seq2seq_attention import Seq2SeqAttention


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



def initialize_model(args):
    input_vocab_index, output_vocab_index = load_vocabularies(
        vocab_path=args.vocab_path
    )

    model = Seq2SeqAttention.restore(
        args.file_checkpoint_dir,
        input_vocab_index,
        output_vocab_index
    )

    return model


args = parser.parse_args()
model = initialize_model(args)
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/predict')
def predict():
    input_text = request.args.get('input')

    print(f'input_text: {input_text}')

    prediction = model.predict(input_text=input_text)

    print(f'prediction: {prediction}')

    return {'prediction': prediction}


if __name__ == '__main__':
    app.run()