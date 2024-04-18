# input.py file contains input functions that are common to all models.
import helper
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

def basicDT():
    # load the data
    data = helper.loadXYtraintest()

    # creating the decision tree model
    tree = DecisionTreeClassifier(criterion="log_loss", max_depth=20)
    tree.fit(data['x_train'], data['y_train'])
    data['y_pred'] = tree.predict(data['x_test'])

    # output of data metrics of the model
    helper.print_common_data_metrics(data['y_test'], data['y_pred'])


basicDT()

def crossvalidateDT():
    # load the data
    X, Y = helper.loadXY()

    # creating the decision tree model
    tree = DecisionTreeClassifier(criterion="log_loss", max_depth=20)

    # cross validation
    scores = cross_val_score(tree, X, Y, cv=5)
    print(scores)

crossvalidateDT()

def gridsearchDT():
    # load the data
    data = helper.loadXYtraintest()

    # creating the support vector machine model
    tree = DecisionTreeClassifier()

    # grid search
    param_grid = {'criterion': ["gini", "entropy", "log_loss"],
                'max_depth': [None, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300],
                'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    grid = GridSearchCV(tree, param_grid, refit = True, verbose = 3)
    grid.fit(data['x_train'], data['y_train'])
    print(grid.best_params_) 

# gridsearchDT()

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
