from sklearn import neighbors
from sklearn.neural_network import MLPClassifier


class KNN:
    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # Structure to hold trained models
        self.models = {}
        # Calls
        self.train()
        self.test_prediction()

    # Train the models
    def train(self):
        neighborsPoll = range(2, 8, 2)  # 2, 4, 8
        weights = ["uniform", "distance"]
        algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
        for neighbor in neighborsPoll:
            for weight in weights:
                for algorithm in algorithms:
                    clf = neighbors.KNeighborsClassifier(
                        n_neighbors=neighbor,
                        weights=weight,
                        algorithm=algorithm,
                    )
                    clf.fit(self.x_train, self.y_train)
                    # Store the model in a structure
                    self.models[(neighbor, weight, algorithm)] = clf

    def test_prediction(self):
        for entry in self.models:
            clf = self.models[entry]
            if entry[0] == 2:
                accuracy = clf.score(self.x_test, self.y_test)
                print(f"K={entry[0]} Weight={entry[1]} Algorithm={entry[2]}")
                print(round(accuracy, 3))


class ANN:
    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # Structure to hold trained models
        self.models = {}
        # Calls
        self.train()
        self.test_prediction()

    def train(self):
        activations = ["identity", "logistic", "tanh", "relu"]
        hidden_layers = [10, 100, 1000]
        iterations = range(200, 800, 200)  # 200, 600, 800
        for activation in activations:
            for layers in hidden_layers:
                for iteration in iterations:
                    clf = MLPClassifier(
                        max_iter=iteration,
                        hidden_layer_sizes=(layers,),
                        activation=activation,
                    )
                    clf.fit(self.x_train, self.y_train)
                    self.models[(activation, layers, iteration)] = clf

    def test_prediction(self):
        for entry in self.models:
            clf = self.models[entry]
            if entry[0] == "relu":
                accuracy = clf.score(self.x_test, self.y_test)
                print(f"Activation={entry[0]} Layers={entry[1]} Iterations={entry[2]}")
                print(round(accuracy, 3))
