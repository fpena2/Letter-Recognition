import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from const import LETTER_PAIRS

buffer = LETTER_PAIRS[1]
X = buffer.loc[:, buffer.columns != "lettr"]
y = buffer["lettr"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# Train the model
n_neighbors = 15
weights = "uniform"
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(X, y)

# print(X_test.iloc[0])
# print(clf.predict(X_test))
# print(y_test)
