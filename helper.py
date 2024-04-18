from input import load_dataset, preprocess, XY_split, traintest_split
from sklearn import metrics

# splits into X and Y
def loadXY():
    # loading and preprocessing the data
    data = preprocess(load_dataset('dataset_files/main.csv'))
    # splitting into X and Y
    return XY_split(data)

# splits into xtrain, xtest, ytrain, ytest
def loadXYtraintest():
    # loading and preprocessing the data
    data = preprocess(load_dataset('dataset_files/main.csv'))
    # convert data to dictionary as it isn't very clean to return multiple values
    x_train, x_test, y_train, y_test = traintest_split(data)
    new_data = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test
    }
    # splitting into xtrain ytrain xtest ytest
    return new_data

def score(y_test, y_pred):
    return metrics.accuracy_score(y_test, y_pred)

def confusion_matrix(y_test, y_pred):
    confusion_array = metrics.confusion_matrix(y_test,y_pred,labels=[1,0])
    return confusion_array

def classification_report(y_test, y_pred):
    return metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])

def print_common_data_metrics(y_test, y_pred):
    print("DATA METRICS")
    print()
    print("Score:", score(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred))
    print()
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification Report")
    print(classification_report(y_test, y_pred))
    