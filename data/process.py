import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


LABELS = ["name", "entry1", "entry2", "entry3", "accuracy", "trainTime", "testTime"]
final_LABELS = [
    "name",
    "entry1",
    "entry2",
    "min",
    "mean",
    "max",
    "trainTime",
    "testTime",
]

"""
"""


def process_files(file_list, entry1_list, entry2_list, outFile):
    handle = open(outFile, "w")
    for inFile in file_list:
        df = pd.read_csv(inFile, names=LABELS)
        for entry1 in entry1_list:
            for entry2 in entry2_list:
                buf = df[df["entry1"] == entry1]
                if entry2 != None:
                    buf = buf[buf["entry2"] == entry2]
                # Get averages for each entry
                _min = buf["accuracy"].min()
                _mean = buf["accuracy"].mean()
                _max = buf["accuracy"].max()
                _train = buf["trainTime"].mean()
                _test = buf["testTime"].mean()
                # Print to file
                s = f"{inFile},{entry1},{entry2},{_min},{_mean},{_max},{_train},{_test}"
                print(s, file=handle)


"""
"""


def process_graph(inFile, file_list, entry1_list, rows, cols, xname, figname):
    # Configure graph
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4)

    n = 0
    df = pd.read_csv(inFile, names=final_LABELS)
    for aFile in file_list:
        for entry in entry1_list:
            # Change plotting location
            n += 1
            ax = plt.subplot(rows, cols, n)
            # Filter by file
            buf = df[df["name"] == aFile]
            # Filter by swept parameters and also pull the X range from the data frame
            if entry != None:
                buf = buf[buf["entry1"] == entry]
                x = buf["entry2"].astype(str).tolist()
            else:
                # Entry will be None for models with a single swept parameter
                x = buf["entry1"].astype(str).tolist()

            # Pull the min, mean, and max lists from the data frame
            _min = buf["min"]
            _mean = buf["mean"]
            _max = buf["max"]
            yerr = [_mean - _min, _max - _mean]

            # Create the plot
            ax.errorbar(x, _mean, yerr=yerr, fmt="o")

            # Style the plot
            name = aFile[aFile.rfind("/") + 1 : aFile.rfind(".")]
            ax.set_title(f"{name} - Param 1 = {entry}")
            ax.set_ylabel("Accuracy")
            ax.set_xlabel(f"Param 2 = {xname}")

    plt.savefig(f"./graphing/{figname}.png", bbox_inches="tight")


"""
"""


def process_graph_bar(inFile, file_list, entry1_list, rows, cols, xname, figname):
    # Configure graph
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4)
    width = 0.35

    n = 0
    df = pd.read_csv(inFile, names=final_LABELS)
    for aFile in file_list:
        for entry in entry1_list:
            # Change plotting location
            n += 1
            ax = plt.subplot(rows, cols, n)
            # Filter by file
            buf = df[df["name"] == aFile]
            # Filter by swept parameters and also pull the X range from the data frame
            if entry != None:
                buf = buf[buf["entry1"] == entry]
                xValues = buf["entry2"].astype(str).tolist()
            else:
                # Entry will be None for models with a single swept parameter
                xValues = buf["entry1"].astype(str).tolist()
            xRange = np.arange(len(xValues))

            _train = buf["trainTime"]
            _test = buf["testTime"]

            rects1 = ax.bar(xRange - width / 2, _train, width, label="Train Time")
            rects2 = ax.bar(xRange + width / 2, _test, width, label="Test Time")
            ax.bar_label(rects1, padding=3)
            ax.bar_label(rects2, padding=3)

            # Style the chart
            name = aFile[aFile.rfind("/") + 1 : aFile.rfind(".")]
            ax.set_title(f"{name} - Param 1 = {entry}")
            ax.set_ylabel("Time (Seconds)")
            ax.set_xlabel(f"Param 2 = {xname}")
            ax.set_xticks(xRange, xValues)
            # Remove bar values
            for child in plt.gca().get_children():
                if isinstance(child, matplotlib.text.Annotation):
                    child.remove()
            ax.legend()

    plt.savefig(f"./graphing/{figname}_Timing.png", bbox_inches="tight")


def KNN():
    file_list = ["./data/KNN.txt", "./data/KNN_PCA.txt"]
    entry1_list = ["uniform", "distance"]
    entry2_list = [2, 16, 64, 256, 512]
    outFile = "./graphing/KNNs.csv"
    process_files(file_list, entry1_list, entry2_list, outFile)
    process_graph(outFile, file_list, entry1_list, 2, 2, "Neighbors", "KNNs")
    file_list = ["./test/KNN.txt", "./test/KNN_PCA.txt"]
    outFile = "./test/KNNs.csv"
    process_files(file_list, entry1_list, entry2_list, outFile)
    process_graph_bar(outFile, file_list, entry1_list, 2, 2, "Neighbors", "KNNs")


def SVC():
    file_list = ["./data/SVC.txt", "./data/SVC_PCA.txt"]
    entry1_list = ["linear", "poly", "rbf"]
    entry2_list = [1.0, 2.0, 16.0, 64.0, 256.0, 512.0]
    outFile = "./graphing/SVCs.csv"
    process_files(file_list, entry1_list, entry2_list, outFile)
    process_graph(outFile, file_list, entry1_list, 3, 3, "Regularization", "SVCs")
    process_graph_bar(outFile, file_list, entry1_list, 3, 3, "Regularization", "SVCs")


def GAUS():
    file_list = ["./data/GAUS.txt", "./data/GAUS_PCA.txt"]
    outFile = "./graphing/GAUS.csv"
    entry1_list = [
        "ConstantKernel",
        "Matern",
        "RationalQuadratic",
        "DotProduct",
        "RBF",
    ]
    entry2_list = [None]
    process_files(file_list, entry1_list, entry2_list, outFile)
    entry1_list = [None]
    process_graph(outFile, file_list, entry1_list, 2, 1, "Kernel", "GAUSs")
    process_graph_bar(outFile, file_list, entry1_list, 2, 1, "Kernel", "GAUSs")


def ANN():
    file_list = ["./data/ANN.txt", "./data/ANN_PCA.txt"]
    entry1_list = ["logistic", "tanh", "relu"]
    entry2_list = [2, 16, 64, 256, 512]
    outFile = "./graphing/ANNs.csv"
    process_files(file_list, entry1_list, entry2_list, outFile)
    process_graph(outFile, file_list, entry1_list, 3, 3, "Hidden Layers", "ANNs")
    process_graph_bar(outFile, file_list, entry1_list, 3, 3, "Hidden Layers", "ANNs")


KNN()
SVC()
GAUS()
ANN()
