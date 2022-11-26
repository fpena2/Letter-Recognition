import time
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    RationalQuadratic,
    DotProduct,
)
from sklearn.svm import SVC


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
        if "KNN" in name:
            self.train_KNN()
        elif "ANN" in name:
            self.train_ANN()
        elif "GAUS" in name:
            self.train_Gaussian()
        elif "SVC" in name:
            self.train_SVC()
        else:
            print("Unknown Model Name")

        # Test and output the results of the trained model
        self.test(self.x_test, self.y_test)
        self.print_results()

    """
    Function to train KNN models 
    """

    def train_KNN(self):
        weights = ["uniform", "distance"]
        neighborsPoll = [2, 16, 64, 256, 512]
        for weight in weights:
            for neighbor in neighborsPoll:
                t0 = time.time()
                clf = neighbors.KNeighborsClassifier(
                    n_neighbors=neighbor,
                    weights=weight,
                )
                clf.fit(self.x_train, self.y_train)
                t = time.time() - t0
                # Store the model in a structure
                self.models[(weight, neighbor, None, t)] = clf

    """
    Function to train SVC models
    """

    def train_SVC(self):
        kernels = ["linear", "poly", "rbf"]
        regularization = [1.0, 2.0, 16.0, 64.0, 256.0, 512.0]
        for kernel in kernels:
            for c in regularization:
                t0 = time.time()
                clf = SVC(
                    C=c,
                    kernel=kernel,
                    random_state=42,  # for reproducibility
                )
                clf.fit(self.x_train, self.y_train)
                t = time.time() - t0
                self.models[(kernel, c, None, t)] = clf

    """
    Function to train Gaussian Process models
    """

    def train_Gaussian(self):
        kernels = {
            "ConstantKernel": ConstantKernel(),
            "Matern": Matern(),
            "RationalQuadratic": RationalQuadratic(),
            "DotProduct": DotProduct(),
            "RBF": RBF(),
        }

        for kernel in kernels:
            t0 = time.time()
            clf = GaussianProcessClassifier(
                kernel=kernels[kernel],
                random_state=42,  # for reproducibility
            )
            clf.fit(self.x_train, self.y_train)
            t = time.time() - t0
            self.models[(kernel, None, None, t)] = clf

    """
    Function to train ANN models
    """

    def train_ANN(self):
        activations = ["logistic", "tanh", "relu"]
        hidden_layers = [2, 16, 64, 256, 512]
        for activation in activations:
            for layers in hidden_layers:
                t0 = time.time()
                clf = MLPClassifier(
                    hidden_layer_sizes=(layers,),
                    activation=activation,
                    random_state=42,  # for reproducibility
                )
                clf.fit(self.x_train, self.y_train)
                t = time.time() - t0
                self.models[(activation, layers, None, t)] = clf

    """
    Function to test trained models
    """

    def test(self, x_test, y_test):
        self.results = {}
        for entry in self.models:
            clf = self.models[entry]
            t0 = time.time()
            accuracy = round(clf.score(x_test, y_test), 5)
            t = time.time() - t0
            # Modify key structure and save to dictionary
            self.results[(self.index, self.name) + entry + (t,)] = accuracy

    """
    Helper functions 
    """

    def print_results(self):
        results = self.results
        for entry in results:
            print(
                # entry[0] removed for sorting purposes
                f"{entry[1]},{entry[2]},{entry[3]},{entry[4]},{results[entry]},{round(entry[5], 4)},{round(entry[6], 4)}",
                file=self.handle,
            )
