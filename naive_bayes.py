# input.py file contains input functions that are common to all models.
import helper
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def basicNB():
    # load the data
    data = helper.loadXYtraintest()

    # creating the naive bayes model
    nb = GaussianNB()
    nb.fit(data['x_train'], data['y_train'])
    data['y_pred'] = nb.predict(data['x_test'])

    # output of data metrics of the model
    helper.print_common_data_metrics(data['y_test'], data['y_pred'])


basicNB()

def crossvalidateNB():
    # load the data
    X, Y = helper.loadXY()

    # creating the naive bayes model
    nb = GaussianNB()

    # cross validation
    scores = cross_val_score(nb, X, Y, cv=5)
    print(scores)

crossvalidateNB()

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
