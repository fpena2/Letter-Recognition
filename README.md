# Letter-Recognition

Training different classification models in order to classify different sets of letters.

## Requirements

- Python 3.10+
- Python Libraries
  - pandas
  - numpy
  - matplotlib
  - sklearn

Use the following command to automatically install all dependencies:

    pip install requirements.txt

# How to use

Simply clone this repository and run the following commands:

    python main.py

Once models are trained, navigate to the "/data/" directory and run the following command in order to generate performance charts:

    python process.py

## Expected Input

Modify the variable `PAIR` inside `main.py`.

- `PAIR = 1` train and test models on the letters `H` & `K`
- `PAIR = 2` train and test models on the letters `M` & `Y`
- `PAIR = 3` train and test models on the letters `A` & `B`
- `PAIR = 4` train and test the multi-class model on the letters `H`, `K`, `M`, `Y`, `A` & `B`

## Expected Output

1. Various CSV files with statistics about each model inside the `/data/` and `/test/` directory
2. Various plots after running `/data/process.py`
   - Examples:  
     ![KNN-Example](/data/KNN.png)
     ![SVC-Example](/test/SVC.png)

# Folder structure

### `/data/`

This folder will contain the raw output of the models: accuracy, training, and testing times, etc.

### `/test/`

This folder will contain results after models are tested using the final validation set (10% of the input data set).

# Libraries/Classes

### `const.py`

This file is used for pre-processing the input [dataset](https://archive.ics.uci.edu/ml/datasets/letter+recognition) file (`letters-recognition.data`)

### `models.py`

This library include all the implementations of the models used.
The `Model` class include a function for each classifier.
