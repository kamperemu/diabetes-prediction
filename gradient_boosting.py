# input.py file contains input functions that are common to all models.
import helper
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def basicGB():
    # load the data
    data = helper.loadXYtraintest()

    # creating the gradient boosting model
    tree = GradientBoostingClassifier(loss = 'exponential')
    tree.fit(data['x_train'], data['y_train'])
    data['y_pred'] = tree.predict(data['x_test'])

    # output of data metrics of the model
    helper.print_common_data_metrics(data['y_test'], data['y_pred'])


basicGB()

def crossvalidateGB():
    # load the data
    X, Y = helper.loadXY()

    # creating the gradient boosting model
    tree = GradientBoostingClassifier(loss = 'exponential')

    # cross validation
    scores = cross_val_score(tree, X, Y, cv=5)
    print(scores)

crossvalidateGB()

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
