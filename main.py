import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from libs.const import LETTER_PAIRS
from libs.models import KNN, ANN

# Split features and response
buffer = LETTER_PAIRS[1]
_X = buffer.loc[:, buffer.columns != "lettr"].to_numpy()
_Y = buffer["lettr"].to_numpy()

# Save 10% of data for testing purposes
_XTrain, _XTest, _YTrain, _YTest = train_test_split(
    _X, _Y, test_size=0.10, random_state=42
)

# Split data (5-fold cross-validation)
index = 0
crossValidation = KFold(n_splits=5, random_state=43, shuffle=True)
for train_index, test_index in crossValidation.split(_XTrain, _YTrain):
    x_train, x_test = _XTrain[train_index], _XTrain[test_index]
    y_train, y_test = _YTrain[train_index], _YTrain[test_index]
    KNN(index, x_train, y_train, x_test, y_test)
    ANN(index, x_train, y_train, x_test, y_test)
    index += 1
