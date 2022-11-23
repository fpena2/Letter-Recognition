from sklearn import neighbors
from sklearn.neural_network import MLPClassifier


class Model:
    def __init__(self, name, index, x_train, y_train, x_test, y_test) -> None:
        self.name = name
        self.index = index
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # Misc
        if self.index == 0:
            self.handle = open(f"./data/{name}.txt", "w")
        else:
            self.handle = open(f"./data/{name}.txt", "a")

        # Structure to hold trained models
        self.models = {}
        self.results = {}

        # Calls
        if name == "KNN":
            self.train_KNN()
        elif name == "ANN":
            self.train_ANN()
        else:
            pass

        # Test and output the results of the trained model
        self.test(self.x_test, self.y_test)
        self.print_results()

    """
    Helper functions 
    """

    def print_results(self):
        results = {key: self.results[key] for key in sorted(self.results.keys())}
        for entry in results:
            print(
                # entry[0] removed for sorting purposes
                f"{entry[1]},{entry[2]},{entry[3]},{entry[4]},{results[entry]}",
                file=self.handle,
            )

    """
    Function to train KNN models 
    """

    def train_KNN(self):
        neighborsPoll = [2, 20, 200]
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

    """
    Function to train ANN models
    """

    def train_ANN(self):
        activations = ["identity", "logistic", "tanh", "relu"]
        hidden_layers = [10, 100, 1000]
        iterations = [200, 600, 800]
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

    """
    Function to test trained models
    """

    def test(self, x_test, y_test):
        for entry in self.models:
            clf = self.models[entry]
            accuracy = round(clf.score(x_test, y_test), 5)
            # Modify key structure and save to dictionary
            self.results[(self.index, self.name) + entry] = accuracy
