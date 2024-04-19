import helper
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, GridSearchCV

def basicQDA():
    # load the data
    data = helper.loadXYtraintest()

    # creating the quadratic discriminant analysis model
    qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
    qda.fit(data['x_train'], data['y_train'])
    data['y_pred'] = qda.predict(data['x_test'])

    # output of data metrics of the model
    helper.print_common_data_metrics(data['y_test'], data['y_pred'])
    return qda

def crossvalidateQDA():
    # load the data
    X, Y = helper.loadXY()

    # creating the quadratic discriminant analysis model
    qda = QuadraticDiscriminantAnalysis(reg_param=0.1)

    # cross validation
    scores = cross_val_score(qda, X, Y, cv=5)
    print(scores)

    return sum(scores)/len(scores)

def gridsearchQDA():
    # load the data
    data = helper.loadXYtraintest()

    # creating the support vector machine model
    qda = QuadraticDiscriminantAnalysis()

    # grid search
    param_grid = {'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    grid = GridSearchCV(qda, param_grid, refit = True, verbose = 3)
    grid.fit(data['x_train'], data['y_train'])
    print(grid.best_params_) 

if __name__ == "__main__":
    basicQDA()
    crossvalidateQDA()
    # gridsearchQDA()

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
