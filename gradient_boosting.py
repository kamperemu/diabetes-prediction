import helper
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

import warnings
warnings.filterwarnings("ignore")

'''
dataset = 'set1', feature_selection = 0 -> {'ccp_alpha': 0.0001, 'learning_rate': 0.1, 'loss': 'exponential'}
dataset = 'set1', feature_selection = 1 -> {'ccp_alpha': 0.0001, 'learning_rate': 0.01, 'loss': 'exponential'}
dataset = 'set1', feature_selection = 2 -> {'ccp_alpha': 0.0001, 'learning_rate': 0.01, 'loss': 'exponential'}

'''

def basicGB():
    # load the data
    data = helper.loadXYtraintest()

    # creating the gradient boosting model
    tree = GradientBoostingClassifier(max_depth=5, n_estimators=200, ccp_alpha=0.0001, learning_rate=0.1, loss='log_loss')
    tree.fit(data['x_train'], data['y_train'])
    data['y_pred'] = tree.predict(data['x_test'])

    # output of data metrics of the model
    helper.print_common_data_metrics(data['y_test'], data['y_pred'])

    return tree

def crossvalidateGB():
    # load the data
    X, Y = helper.loadXY()

    # creating the gradient boosting model
    tree = GradientBoostingClassifier(loss = 'exponential')

    # cross validation
    scores = cross_val_score(tree, X, Y, cv=5)
    print(scores)
    
    return sum(scores)/len(scores)

def gridsearchGB():
    # load the data
    data = helper.loadXYtraintest()

    # creating the gradient boosting model
    tree = GradientBoostingClassifier(n_estimators=200)

    # grid search
    param_grid = {'loss': ['log_loss', 'exponential'],
              'learning_rate': [0.1, 0.01, 0.001],
              'ccp_alpha': [ 0.01, 0.001, 0.0001]}
    grid = GridSearchCV(tree, param_grid, refit = True, verbose = 3)
    grid.fit(data['x_train'], data['y_train'])
    print(grid.best_params_)

if __name__ == "__main__":
    model = basicGB()
    #crossvalidateGB()
    #gridsearchGB()
    # user input
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