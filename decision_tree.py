# input.py file contains input functions that are common to all models.
import helper
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def basicDT():
    # load the data
    data = helper.loadXYtraintest()

    # creating the decision tree model
    tree = DecisionTreeClassifier(criterion="log_loss")
    tree.fit(data['x_train'], data['y_train'])
    data['y_pred'] = tree.predict(data['x_test'])

    # output of data metrics of the model
    helper.print_common_data_metrics(data['y_test'], data['y_pred'])


basicDT()

def crossvalidateDT():
    # load the data
    X, Y = helper.loadXY()

    # creating the decision tree model
    tree = DecisionTreeClassifier(criterion="log_loss")

    # cross validation
    scores = cross_val_score(tree, X, Y, cv=5)
    print(scores)

crossvalidateDT()

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
