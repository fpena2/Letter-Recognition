import pandas as pd
import matplotlib.pyplot as plt

LABELS = ["name", "entry1", "entry2", "entry3", "accuracy", "trainTime", "testTime"]
final_LABELS = ["name", "entry1", "entry2", "min", "mean", "max"]


def process_files(file_list, entry1_list, entry2_list, outFile):
    handle = open(outFile, "w")
    for inFile in file_list:
        df = pd.read_csv(inFile, names=LABELS)
        for entry1 in entry1_list:
            for entry2 in entry2_list:
                buf = df[df["entry1"] == entry1]
                if entry2 != None:
                    buf = buf[buf["entry2"] == entry2]

                _min, _mean, _max = (
                    buf["accuracy"].min(),
                    buf["accuracy"].mean(),
                    buf["accuracy"].max(),
                )
                # Print to file
                s = f"{inFile},{entry1},{entry2},{_min},{_mean},{_max}"
                print(s, file=handle)


def process_graph(inFile, file_list, entry1_list, rows, cols, xname, figname):
    df = pd.read_csv(inFile, names=final_LABELS)
    # Configure graph
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4)

    n = 0
    for aFile in file_list:
        for entry in entry1_list:
            n += 1
            name = aFile[aFile.rfind("/") + 1 : aFile.rfind(".")]
            ax = plt.subplot(rows, cols, n)
            buf = df[df["name"] == aFile]
            if entry != None:
                buf = buf[buf["entry1"] == entry]
                x = buf["entry2"].astype(str).tolist()
            else:
                x = buf["entry1"].astype(str).tolist()

            _min, _mean, _max = (
                buf["min"],
                buf["mean"],
                buf["max"],
            )

            ax.errorbar(
                x,
                _mean,
                yerr=[_mean - _min, _max - _mean],
                fmt="o",
            )
            ax.set_title(f"{name} - Param 1 = {entry}")
            ax.set_ylabel("Accuracy")
            ax.set_xlabel(f"Param 2 = {xname}")
    plt.savefig(f"./graphing/{figname}.png", bbox_inches="tight")


#
file_list = ["./data/KNN.txt", "./data/KNN_PCA.txt"]
entry1_list = ["uniform", "distance"]
entry2_list = [2, 16, 64, 256, 512]
outFile = "./graphing/KNNs.csv"
process_files(file_list, entry1_list, entry2_list, outFile)
process_graph(outFile, file_list, entry1_list, 2, 2, "Neighbors", "KNNs")

#
file_list = ["./data/SVC.txt", "./data/SVC_PCA.txt"]
entry1_list = ["linear", "poly", "rbf"]
entry2_list = [1.0, 2.0, 16.0, 64.0, 256.0, 512.0]
outFile = "./graphing/SVCs.csv"
process_files(file_list, entry1_list, entry2_list, outFile)
process_graph(outFile, file_list, entry1_list, 3, 3, "Regularization", "SVCs")

#
file_list = ["./data/GAUS.txt", "./data/GAUS_PCA.txt"]
outFile = "./graphing/GAUS.csv"
entry1_list = [1.0, 2.0, 16.0, 64.0, 256.0, 512.0]
entry2_list = [None]
process_files(file_list, entry1_list, entry2_list, outFile)
entry1_list = [None]
process_graph(outFile, file_list, entry1_list, 2, 1, "Kernel Multiplier", "GAUSs")

#
file_list = ["./data/ANN.txt", "./data/ANN_PCA.txt"]
entry1_list = ["logistic", "tanh", "relu"]
entry2_list = [2, 16, 64, 256, 512]
outFile = "./graphing/ANNs.csv"
process_files(file_list, entry1_list, entry2_list, outFile)
process_graph(outFile, file_list, entry1_list, 3, 3, "Hidden Layers", "ANNs")
