import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split, KFold
from const import LETTER_PAIRS

# Split features and response
buffer = LETTER_PAIRS[1]
_X = buffer.loc[:, buffer.columns != "lettr"].to_numpy()
_Y = buffer["lettr"].to_numpy()

# Save 10% of data for testing purposes
_XTrain, _XTest, _yTrain, _yTest = train_test_split(
    _X, _Y, test_size=0.10, random_state=42
)

# Split data (5-fold cross-validation)
crossValidation = KFold(n_splits=5, random_state=43, shuffle=True)
for train_index, test_index in crossValidation.split(_XTrain, _yTrain):
    X_train, X_test = _XTrain[train_index], _XTrain[test_index]
    y_train, y_test = _yTrain[train_index], _yTrain[test_index]
    # print(y_train)
    # print(y_test)
    # print()


class KNN:
    def __init__(self, x_train, y_train) -> None:
        self.x_train = x_train
        self.y_train = y_train

    # Train the models
    def train(self):
        neighborsPoll = range(2, 10, 2)
        weights = ["uniform", "distance"]
        algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
        for neighbor in neighborsPoll:
            for weight in weights:
                for algorithm in algorithms:
                    clf = neighbors.KNeighborsClassifier(
                        n_neighbors=neighbor, weights=weight, algorithm=algorithm
                    )
                    clf.fit(self.x_train, self.y_train)

    def test_prediction(self):
        pass

    # print(X_test.iloc[0])
    # print(clf.predict(X_test))
    # print(y_test)
