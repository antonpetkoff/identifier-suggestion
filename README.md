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

- [ ] Serve the trained model for predictions
  - [ ] Make the Seq2Seq class a tf.Module with @tf.functions with signatures so that it can be serialized
  - [ ] Ensure the Encoder and Decoder models are properly (de)serialized inside the module
  - [ ] Test the SavedModel through the saved_model_cli
  - [ ] Save the model to disk on every 2nd epoch
  - [ ] Serve the SavedModel from a Docker container
  - [ ] Preprocess the raw input text for the model

- [ ] Implement an IDE suggestion plugin/extension for VSCode using LSP (Language Server Protocol)
  - [ ] Provide the method body by making a selection with the cursor
  - [ ] Query the served model by HTTP to receive a list of suggestions

- [ ] Add Beam Search Decoder for making multiple suggestions

- [ ] Start writing your thesis
  - [ ] Document experiments

- [ ] Increase Dataset to at least 1 million method samples for training
  - [ ] Select a set of Java repositories
  - [ ] Write scripts for downloading the source code repositories at specific hashes
  - [ ] Fetch only the Java code from these repositories
  - [ ] Analyze the distribution of the data
  - [ ] Split test files and everything else
  - [ ] Write scripts for extracting Java methods
  - [ ] Split the dataset into train/test/validation sets
  - [ ] Preprocess data
    - [ ] Extract the preprocessing step as a method in the model

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

- [x] Evaluation
  - [x] Add evaluation metrics
    - [x] Accuracy
    - [x] Precision
    - [x] Recall
    - [x] F1 score
    - [ ] Define the token position strictness of the confusion matrix metrics
    - [ ] Order-aware metrics
      - [ ] Some type of edit distance
      - [ ] BLEU
        - [ ] BLEU score for different sequence lengths? [see Extensions here](https://machinelearningmastery.com/encoder-decoder-recurrent-neural-network-models-neural-machine-translation/)
  - [x] Run evaluation on test set after each epoch
    - [x] Can we run the evaluation in parallel on the CPU while the model trains on the GPU? Will not do it for now
  - [x] Log evaluation metrics with `weights and biases`

- [x] Make a notebook for Google Colab to train on Tesla K80
  - [x] Make your repo public
  - [x] Clone it in the notebook
  - [x] Authenticate and fetch credentials for `wandb` and other services
  - [x] Fetch the preprocessed data from Drive
  - [x] Run the training script
  - [ ] Save checkpoints
  - [ ] Restore from checkpoints, if the training fails

- [x] Log model summary - architecture, parameter counts, shapes

- [ ] Initialization
  - [ ] Will Xavier or Random Normal initialization improve training time?

- [ ] Regularization
  - [ ] L2 Regularization
  - [ ] [Dropout vs Batch Normalization](https://arxiv.org/abs/1502.03167)
  - [ ] LayerNorm vs BatchNorm

- [x] Fixate all random seeds for reproducible results

- [ ] Run one full experiment
  - [ ] Describe the experiment - setup, expectations (hypothesis) vs results, goals, architecture, meaning of parameters, evaluation
  - [x] Log training and evaluation
  - [x] Save configuration
  - [ ] Save checkpoints of model weights
  - [x] Make predictions
    - [ ] Stop predicting elements once you hit the <EOS> marker

- [ ] Visualizations
  - [x] Log predictions into text tables for transparency on how the model performs
  - [ ] Plot embeddings
    - [ ] Reduce the embedding matrices with t-SNE or other dimensionality reduction algorithms
    - [ ] Make a 2D/3D Plot with a good (interpretable) sample
    - [ ] Log/Upload the plot in `wandb`

  - [ ] Heat map of Attention weights during a single prediction. [see here](https://www.researchgate.net/figure/Heatmaps-of-attention-weights-a-i-j_fig1_316184919)

  - [ ] Can we log layer weights / activations / gradients for debugging? [see here](https://www.quora.com/Are-there-any-examples-of-tensorflow-that-shows-how-to-monitor-the-jacobian-and-or-the-hessian)
    - [ ] [Tutorial](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/)
    - [ ] [Question](https://stackoverflow.com/questions/42315202/understanding-tensorboard-weight-histograms)

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
  - [ ] Transformer
  - [ ] AST features (AST path embeddings)

Optional:

- [ ] Add types with [typing](https://docs.python.org/3/library/typing.html)
