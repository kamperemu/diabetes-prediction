import helper
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

def basicQDA():
    # load the data
    data = helper.loadXYtraintest()

    # creating the quadratic discriminant analysis model
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(data['x_train'], data['y_train'])
    data['y_pred'] = qda.predict(data['x_test'])

    # output of data metrics of the model
    helper.print_common_data_metrics(data['y_test'], data['y_pred'])


basicQDA()

def crossvalidateQDA():
    # load the data
    X, Y = helper.loadXY()

    # creating the quadratic discriminant analysis model
    qda = QuadraticDiscriminantAnalysis()

    # cross validation
    scores = cross_val_score(qda, X, Y, cv=5)
    print(scores)

crossvalidateQDA()


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
