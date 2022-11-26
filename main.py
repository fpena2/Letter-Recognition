from sklearn.model_selection import train_test_split, KFold
from libs.const import LETTER_PAIRS
from libs.models import Model
from sklearn.decomposition import PCA

# ================================
# REMOVE THIS
import warnings

warnings.filterwarnings("ignore")
# ================================

# Configuration
MODEL_NAMES = [
    # "KNN",
    # "SVC",
    "GAUS",
    # "ANN",
]
DIMENSION_REDUCTION = [False, True]

# Split features and response
buffer = LETTER_PAIRS[3]
_X = buffer.loc[:, buffer.columns != "lettr"].to_numpy()
_Y = buffer["lettr"].to_numpy()

# Save 10% of data for testing purposes
_XTrain, _XTest, _YTrain, _YTest = train_test_split(
    _X, _Y, test_size=0.10, random_state=42
)

for do_reduction in DIMENSION_REDUCTION:
    modelNames = []
    # Reduce the number of features
    if do_reduction:
        pca = PCA(n_components=4, random_state=42)
        _XTrain = pca.fit_transform(_XTrain)
        _XTest = pca.transform(_XTest)
        modelNames = [n + "_PCA" for n in MODEL_NAMES]
    else:
        modelNames = MODEL_NAMES

    # Split data (5-fold cross-validation)
    index = 0
    crossValidation = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in crossValidation.split(_XTrain, _YTrain):
        x_train, x_test = _XTrain[train_index], _XTrain[test_index]
        y_train, y_test = _YTrain[train_index], _YTrain[test_index]
        for model in modelNames:
            Model(model, index, x_train, y_train, x_test, y_test)
        index += 1
