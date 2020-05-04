# Identifier Suggestion

Models for source code identifier suggestion built by learning from Big Code.

## Setup

### Requirements

1. Have Python 3.7 or above installed.

2. Have a Unix-like terminal with `bash`, `coreutils` and some other utilities like `wget` (required by some .sh scripts).

### Installation

1. Run `pip install -r requirements/dev.txt` to install all Python package dependencies.

### Installation with a virtual environment

For example, you can use the standard `venv`:

1. Run `python3 -m venv .venv` to create a virtual environment.

1. Run `source .venv/bin/activate` to activate the virtual environment. This step must be executed on every new terminal session.

1. Run `pip install -r requirements/dev.txt` to install all Python package dependencies.

## Project Structure

Follows the [Cookiecutter Data Science project structure](https://drivendata.github.io/cookiecutter-data-science/).

```text
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── requirements.txt   <- The requirements file for reproducing this environment.
                          The list with dependencies and their versions
```

## Tasks

Fundamentals:

- [ ] Dataset creation - organize project structure to make it more easily reproducible
  - [ ] Write scripts for downloading the source code repositories (with wget)
  - [ ] Write scripts for generating the datasets
  - [ ] Add a way to download the preprocessed data for training (e.g. S3, Drive)

- [x] Create data input pipeline with tf.data Dataset API
  - [x] Preprocess raw text sequences into padded number sequences (IO efficient)
    - [x] Read
    - [x] Filter
    - [x] Tokenize
    - [x] Build vocabulary
      - [x] Save vocabulary to file
    - [x] Encode to numbers
    - [x] Pad, align and cut
    - [x] Shuffle
    - [x] Save final sequences to binary data files
  - [x] Create tensors for training (memory intensive)
    - [x] Load the data
    - [x] Use the Dataset API to batch, shuffle and repeat

- [ ] Evaluation
  - [ ] Split a test set for evaluation after/while training
  - [ ] Split a validation set for hyperparameter optimization
  - [ ] Add evaluation metrics
    - [ ] Accuracy
    - [ ] Precision
    - [ ] Recall
    - [ ] F1 score
    - [ ] BLEU
      - [ ] BLEU score for different sequence lengths? [see Extensions here](https://machinelearningmastery.com/encoder-decoder-recurrent-neural-network-models-neural-machine-translation/)
  - [ ] Run evaluation on test set after each epoch
  - [ ] Log evaluation metrics with `weights and biases`

- [ ] Make a notebook for Google Colab to train on Tesla K80
  - [ ] Make your repo public
  - [ ] Clone it in the notebook
  - [ ] Authenticate and fetch credentials for `wandb` and other services
  - [ ] Fetch the preprocessed data
    - [ ] Find where to host the dataset files - S3, Drive, or ?
  - [ ] Run the training script
    - [ ] Save checkpoints
    - [ ] Restore from checkpoints, if the training fails

- [ ] Add Beam Search Decoder for making multiple suggestions

- [ ] Regularization
  - [ ] L2 Regularization
  - [ ] [Dropout vs Batch Normalization](https://arxiv.org/abs/1502.03167)
  - [ ] LayerNorm vs BatchNorm

- [x] Fixate all random seeds for reproducible results

- [ ] Run one full experiment
  - [ ] Describe the experiment - setup, expectations (hypothesis) vs results, goals, architecture, meaning of parameters, evaluation
  - [ ] Log training and evaluation
  - [x] Save configuration
  - [ ] Save checkpoints of model weights
  - [x] Make predictions

- [ ] Visualizations
  - [ ] Plot embeddings
    - [ ] Reduce the embedding matrices with t-SNE or other dimensionality reduction algorithms
    - [ ] Make a 2D/3D Plot with a good (interpretable) sample
    - [ ] Log/Upload the plot in `wandb`

  - [ ] Heat map of Attention weights during a single prediction. [see here](https://www.researchgate.net/figure/Heatmaps-of-attention-weights-a-i-j_fig1_316184919)

  - [ ] What other visualizations can be useful?

Modelling and Feature Engineering (Creative):

- [ ] Are there better evaluation metrics than accuracy and BLEU for our task?
  - [ ] Are there differentiable ones which we can use as loss functions?
  - [ ] Can we incorporate synonym sets in the loss function for looser evaluation?

- [ ] Does performance improve when using:
  - [ ] Subtoken splitting
  - [ ] Attention for the decoder
  - [ ] Bi-directional RNN for the encoder
  - [ ] Custom Binary/Categorical features
  - [ ] AST features (AST path embeddings)

Optional:

- [ ] Add types with [typing](https://docs.python.org/3/library/typing.html)
