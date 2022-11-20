from sklearn import neighbors
from sklearn.neural_network import MLPClassifier


class KNN:
    def __init__(self, index, x_train, y_train, x_test, y_test) -> None:
        self.index = index
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # Misc
        if self.index == 0:
            self.handle = open("KNN.txt", "w")
        else:
            self.handle = open("KNN.txt", "a")

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
            accuracy = clf.score(self.x_test, self.y_test)
            # K,Weight,Algorithm
            print(
                f"{self.index},KNN,{entry[0]},{entry[1]},{entry[2]},{round(accuracy, 3)}",
                file=self.handle,
            )


class ANN:
    def __init__(self, index, x_train, y_train, x_test, y_test) -> None:
        self.index = index
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # Misc
        if self.index == 0:
            self.handle = open("ANN.txt", "w")
        else:
            self.handle = open("ANN.txt", "a")
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
            accuracy = clf.score(self.x_test, self.y_test)
            # Name,Activation,Layers,Iterations
            print(
                f"{self.index},ANN,{entry[0]},{entry[1]},{entry[2]},{round(accuracy, 3)}",
                file=self.handle,
            )


class SVM:
    def __init__(self, index, x_train, y_train, x_test, y_test) -> None:
        self.index = index
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # Misc
        if self.index == 0:
            self.handle = open("SVM.txt", "w")
        else:
            self.handle = open("SVM.txt", "a")
        # Structure to hold trained models
        self.models = {}
        # Calls
        # self.train()
        # self.test_prediction()
