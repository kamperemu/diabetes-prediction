from input import load_dataset, preprocess, XY_split, traintest_split
from sklearn import metrics
import pandas as pd

# expieremntal settings variables
# datasets available set1, set2, combined
DATASET = "combined" 
"""
Feature Selection
- 0: All features
- 1: Common features (across datasets)
- 2: Important features (from features_selection.py)
"""
FEATURE_SELECTION = 0

def load_data(dataset):
    if dataset == "set2":
        return load_dataset("dataset_files/" + dataset + "/main.csv")
    else:
        return preprocess(load_dataset("dataset_files/" + dataset + "/main.csv"))

def get_header():
    data = load_data(DATASET)
    return data.columns.values.tolist()

# splits into X and Y
def loadXY(feature_selection=FEATURE_SELECTION, dataset=DATASET):
    # loading and preprocessing the data
    data = load_data(dataset)
    
    # splitting into X and Y
    return XY_split(data, feature_selection, dataset)

# splits into xtrain, xtest, ytrain, ytest
def loadXYtraintest(feature_selection=FEATURE_SELECTION, dataset=DATASET):
    # loading and preprocessing the data

    data = load_data(dataset)

    # convert data to dictionary as it isn't very clean to return multiple values
    x_train, x_test, y_train, y_test = traintest_split(data, feature_selection, dataset)
    new_data = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test
    }
    # splitting into xtrain ytrain xtest ytest
    return new_data

def confusion_matrix(y_test, y_pred):
    confusion_array = metrics.confusion_matrix(y_test,y_pred,labels=[1,0])
    return confusion_array

def classification_report(y_test, y_pred):
    return metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])

def print_common_data_metrics(y_test, y_pred, graphs=False):
    print("DATA METRICS")
    print()
    print("Score:", metrics.accuracy_score(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred))
    print()
    print("Confusion Matrix")
    confusion_array = confusion_matrix(y_test, y_pred)
    print(confusion_array)
    print()
    print("Classification Report")
    print(classification_report(y_test, y_pred))
    if(graphs):
        # visual display for data meterics
        import seaborn as sn
        import matplotlib.pyplot as plt
        import pandas as pd


        df_cm = pd.DataFrame(confusion_array, range(2), range(2))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

        plt.show()
    