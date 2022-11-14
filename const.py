import pandas as pd

result = ["lettr"]

features = [
    "x-box",
    "y-box",
    "width",
    "high",
    "onpix",
    "x-bar",
    "y-bar",
    "x2bar",
    "y2bar",
    "xybar",
    "x2ybr",
    "xy2br",
    "x-ege",
    "xegvy",
    "y-ege",
    "yegvx",
]

labels = result + features

# Read data
df = pd.read_csv("letter-recognition.data", names=labels)

# Pair 1
H = df.loc[df["lettr"] == "H"]
K = df.loc[df["lettr"] == "K"]
# Pair 2
M = df.loc[df["lettr"] == "M"]
Y = df.loc[df["lettr"] == "Y"]
# Pair 3
A = df.loc[df["lettr"] == "A"]
B = df.loc[df["lettr"] == "B"]

LETTER_PAIRS = {
    1: pd.concat([H, K], ignore_index=True),  # 1473 samples
    2: pd.concat([M, Y], ignore_index=True),  # 1578 samples
    3: pd.concat([A, B], ignore_index=True),  # 1555 samples
    4: pd.concat([H, K, M, Y, A, B], ignore_index=True),
}

# rng = np.random.default_rng(seed=42)
# randList = rng.choice(range(len(H)), size=int(0.1 * len(H)), replace=False)
# print(len(randList))
# print(H)

# print(f"len(H):{len(H)}, len(K):{len(K)}")
# print(f"len(M):{len(M)}, len(Y):{len(Y)}")
# print(f"len(A):{len(A)}, len(B):{len(B)}")
