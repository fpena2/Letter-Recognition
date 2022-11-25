import pandas as pd

LABELS = ["name", "entry1", "entry2", "entry3", "accuracy", "trainTime", "testTime"]


def process_file(file_list, entry1_list, entry2_list, outFile):
    outFile = "./graphing/KNNs.csv"
    handle = open(outFile, "w")
    for inFile in file_list:
        df = pd.read_csv(inFile, names=LABELS)
        for entry1 in entry1_list:

            for entry2 in entry2_list:
                buf = df[df["entry1"] == entry1]
                buf = buf[buf["entry2"] == entry2]
                _min, _mean, _max = (
                    buf["accuracy"].min(),
                    buf["accuracy"].mean(),
                    buf["accuracy"].max(),
                )
                # Print to file
                s = f"{inFile},{entry1},{entry2},{_min},{_mean},{_max}"
                print(s, file=handle)

#
file_list = ["./data/KNN.txt", "./data/KNN_PCA.txt"]
entry1_list = ["uniform", "distance"]
entry2_list = [2, 16, 64, 256, 512]
outFile = "./graphing/KNNs.csv"
process_file(file_list, entry1_list, entry2_list, outFile)

#
file_list = ["./data/SVC.txt", "./data/SVC_PCA.txt"]
entry1_list = ["linear", "poly", "rbf"]
entry2_list = [1.0, 2.0, 16.0, 64.0, 256.0, 512.0]
outFile = "./graphing/SVCs.csv"
process_file(file_list, entry1_list, entry2_list, outFile)

#
