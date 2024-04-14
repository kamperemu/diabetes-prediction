# input.py file contains input functions that are common to all models.
from input import load_dataset, preprocess, XY_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics

# loading and preprocessing the data
data = preprocess(load_dataset('dataset_files/main.csv'))
# reading the data for debugging
# print(data.head())
# splitting the data for the machine learning model
x_train, x_test, y_train, y_test = XY_split(data)

# creating the qda model
qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train, y_train)

# different kinds of data metrics for the model
score = qda.score(x_test, y_test)
print("Score:", score)
y_pred = qda.predict(x_test)
confusion_array = metrics.confusion_matrix(y_test,y_pred,labels=[1,0])
print("Confusion Matrix")
print(confusion_array)

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
