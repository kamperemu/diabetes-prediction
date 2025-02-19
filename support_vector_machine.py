import helper
from sklearn import svm
from sklearn.model_selection import cross_val_score, GridSearchCV

import warnings
warnings.filterwarnings("ignore")

def basicSVM():
    # load the data
    data = helper.loadXYtraintest()

    # creating the support vector machine model
    # variable named svmm as svm already exists as an object of a module
    svmm = svm.SVC(C=1000, gamma=0.0001, kernel='rbf')
    svmm.fit(data['x_train'], data['y_train'])
    data['y_pred'] = svmm.predict(data['x_test'])

    # output of data metrics of the model
    helper.print_common_data_metrics(data['y_test'], data['y_pred'])

    return svmm

def crossvalidateSVM():
    # load the data
    X, Y = helper.loadXY()

    # creating the support vector machine model
    svmm = svm.SVC(C=1000, gamma=0.0001, kernel='rbf')

    # cross validation
    scores = cross_val_score(svmm, X, Y, cv=5)
    print(scores)
    return sum(scores)/len(scores)

def gridsearchSVM():
    # load the data
    data = helper.loadXYtraintest()

    # creating the support vector machine model
    svmm = svm.SVC()

    # grid search
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
    grid = GridSearchCV(svmm, param_grid, refit = True, verbose = 3)
    grid.fit(data['x_train'], data['y_train'])
    print(grid.best_params_) 

if __name__ == "__main__":
    model = basicSVM()
    #crossvalidateSVM()
    #gridsearchSVM()
    while True:
        x = []
        for i in helper.get_header()[:-1]:
            x.append(int(input(f"Input {i}: ")))
        print()
        if model.predict([x]) == 0:
            print("No Diabetes")
        else:
            print("Diabetes")
        print()

"""
# visual display for data meterics
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd


df_cm = pd.DataFrame(confusion_array, range(2), range(2))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
"""
