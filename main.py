import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from const import LETTER_PAIRS

buffer = LETTER_PAIRS[1]
X = buffer.loc[:, buffer.columns != "lettr"].to_numpy()
Y = buffer["lettr"].to_numpy()

# print(buffer)

# Split data 5-fold cross-validation
crossValidation = KFold(n_splits=5, random_state=43, shuffle=True)
for train_index, test_index in crossValidation.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    print(len(y_train))
    print(len(y_test))
    print()


# Train the model
# n_neighbors = 15
# weights = "uniform"
# clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
# clf.fit(X, y)

# print(X_test.iloc[0])
# print(clf.predict(X_test))
# print(y_test)
