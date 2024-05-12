import helper
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV

'''
dataset = 'set1', feature_selection = 0 -> {'var_smoothing': 1e-05}
dataset = 'set2', feature_selection = 0 -> {'var_smoothing': 1e-09}
dataset = 'combined', feature_selection = 0 -> {'var_smoothing': 1e-05}
dataset = 'combined', feature_selection = 2 -> {'var_smoothing': 1e-05}
'''


def basicNB():
    # load the data
    data = helper.loadXYtraintest()

    # creating the naive bayes model
    nb = GaussianNB(var_smoothing=1e-5)
    nb.fit(data['x_train'], data['y_train'])
    data['y_pred'] = nb.predict(data['x_test'])

    # output of data metrics of the model
    helper.print_common_data_metrics(data['y_test'], data['y_pred'])

    return nb

def crossvalidateNB():
    # load the data
    X, Y = helper.loadXY()

    # creating the naive bayes model
    nb = GaussianNB(var_smoothing=1e-5)

    # cross validation
    scores = cross_val_score(nb, X, Y, cv=5)
    print(scores)

    return sum(scores)/len(scores)

def gridsearchNB():
    # load the data
    data = helper.loadXYtraintest()

    # creating the naive bayes model
    nb = GaussianNB()

    # grid search
    param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
    grid = GridSearchCV(nb, param_grid, refit = True, verbose = 3)
    grid.fit(data['x_train'], data['y_train'])
    print(grid.best_params_)

if __name__ == "__main__":
    basicNB()
    #crossvalidateNB()
    #gridsearchNB()

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
