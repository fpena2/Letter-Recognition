from sklearn.model_selection import train_test_split, KFold
from libs.const import LETTER_PAIRS
from libs.models import Model
from sklearn.decomposition import PCA
import warnings
# Ignore warnings from the ANN classifier
warnings.filterwarnings("ignore")


# Configuration
PAIR = 4
DIMENSION_REDUCTION = [False, True]
MODEL_NAMES = [
    "KNN",
    "SVC",
    "GAUS",
    "ANN",
]

# Split features and response
buffer = LETTER_PAIRS[PAIR]
_X = buffer.loc[:, buffer.columns != "lettr"].to_numpy()
_Y = buffer["lettr"].to_numpy()

# Save 10% of data for testing purposes
_XTrain, _XTest, _YTrain, _YTest = train_test_split(
    _X, _Y, test_size=0.10, random_state=42
)

# Container to hold the trained models (Model class objects)
models = {}
for do_reduction in DIMENSION_REDUCTION:
    modelNames = []
    if do_reduction:
        # Reduce the number of features using PCA
        pca = PCA(n_components=4, random_state=42)
        _XTrain = pca.fit_transform(_XTrain)
        _XTest = pca.transform(_XTest)
        # Rename the models if PCA is used
        modelNames = [n + "_PCA" for n in MODEL_NAMES]
        if PAIR == 4:
            modelNames = ["Multi-class" + "_PCA"]
    else:
        modelNames = MODEL_NAMES
        if PAIR == 4:
            modelNames = ["Multi-class"]

    # Split data (5-fold cross-validation)
    index = 0
    crossValidation = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in crossValidation.split(_XTrain, _YTrain):
        x_train, x_test = _XTrain[train_index], _XTrain[test_index]
        y_train, y_test = _YTrain[train_index], _YTrain[test_index]

        for model in modelNames:
            # This key is used for identification of the model
            key = (index, PAIR, model)
            if PAIR == 4:
                # multi-class classification
                models[key] = Model(key, x_train, y_train, x_test, y_test)
            else:
                # Binary classification
                models[key] = Model(key, x_train, y_train, x_test, y_test)
        index += 1

    # Test the 10% test dataset against one of the models
    for model in modelNames:
        key = (0, PAIR, model)
        if PAIR == 4:
            models[key].test(_XTest, _YTest)
            models[key].print_results(open(f"./test/{model}.txt", "a"))
        else:
            models[key].test(_XTest, _YTest)
            models[key].print_results(open(f"./test/{model}.txt", "a"))
