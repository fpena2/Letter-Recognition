import time
from statistics import mean
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


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
        self.times = []

        # Calls
        if "KNN" in name:
            self.train_KNN()
        elif "ANN" in name:
            self.train_ANN()
        elif "GAUS" in name:
            self.train_Gaussian()
        else:
            print("Unknown Model Name")

        # Test and output the results of the trained model
        self.test(self.x_test, self.y_test)
        self.print_results()

    """
    Function to train KNN models 
    """

    def train_KNN(self):
        neighborsPoll = [2, 16, 64, 256, 512]
        weights = ["uniform", "distance"]
        algorithms = ["ball_tree", "kd_tree", "brute"]
        leaf_sizes = [2, 16, 64, 256, 512]
        for neighbor in neighborsPoll:
            for weight in weights:
                for algorithm in algorithms:
                    for leafs in leaf_sizes:
                        t0 = time.time()
                        clf = neighbors.KNeighborsClassifier(
                            n_neighbors=neighbor,
                            weights=weight,
                            algorithm=algorithm,
                            leaf_size=leafs,
                        )
                        clf.fit(self.x_train, self.y_train)
                        t1 = time.time()
                        self.times.append(t1-t0)
                        # Store the model in a structure
                        self.models[(neighbor, weight, algorithm, leafs, None)] = clf

    """
    Function to train ANN models
    """

    def train_ANN(self):
        activations = ["identity", "logistic", "tanh", "relu"]
        hidden_layers = [2, 16, 64, 256, 512]
        iterations = [200, 600, 800]
        solvers = ["lbfgs", "sgd", "adam"]
        learning_rates = ["constant", "invscaling", "adaptive"]
        for activation in activations:
            for layers in hidden_layers:
                for iteration in iterations:
                    for solver in solvers:
                        for learning_rate in learning_rates:
                            clf = MLPClassifier(
                                max_iter=iteration,
                                hidden_layer_sizes=(layers,),
                                activation=activation,
                                solver=solver,
                                learning_rate=learning_rate,
                                random_state=42,  # for reproducibility
                            )
                            clf.fit(self.x_train, self.y_train)
                            self.models[
                                (activation, layers, iteration, solver, learning_rate)
                            ] = clf

    """
    Function to train Gaussian Process models
    """

    def train_Gaussian(self):
        kernels = [1.0, 3.0, 10.0]
        iterations = [2, 20, 80]
        for kernel in kernels:
            for iteration in iterations:
                clf = GaussianProcessClassifier(
                    kernel=kernel * RBF(1.0),
                    max_iter_predict=iteration,
                    random_state=42,  # for reproducibility
                )
                clf.fit(self.x_train, self.y_train)
                self.models[(kernel, iteration, None, None, None)] = clf

    """
    Function to test trained models
    """

    def test(self, x_test, y_test):
        for entry in self.models:
            clf = self.models[entry]
            accuracy = round(clf.score(x_test, y_test), 5)
            # Modify key structure and save to dictionary
            self.results[(self.index, self.name) + entry] = accuracy

    """
    Helper functions 
    """

    def print_results(self):
        results = {key: self.results[key] for key in sorted(self.results.keys())}
        for entry in results:
            # Iterate over the tuple
            output = ""
            for e in entry:
                output += f"{e},"
            output += f"{results[entry]},{min(self.times)},{mean(self.times)},{max(self.times)}"

            print(
                # entry[0] removed for sorting purposes
                # f"{entry[1]},{entry[2]},{entry[3]},{entry[4]},{entry[5]},{entry[6]},{}",
                output,
                file=self.handle,
            )
