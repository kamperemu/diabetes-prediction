# input.py file contains input functions that are common to all models.
from input import load_dataset, preprocess, XY_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# loading and preprocessing the data
data = preprocess(load_dataset('dataset_files/main.csv'))
# reading the data for debugging
# print(data.head())
# splitting the data for the machine learning model
x_train, x_test, y_train, y_test = XY_split(data)

# creating the logistic regresesion model
lr = LogisticRegression(max_iter = 300, class_weight='balanced')
lr.fit(x_train, y_train)

# different kinds of data metrics for the model
score = lr.score(x_test, y_test)
print("Score:", score)
print("Coefficient:", lr.coef_)

y_pred = lr.predict(x_test)
f1_value = metrics.f1_score(y_test, y_pred)
print("F1 Harmonic Mean:", f1_value)

confusion_array = metrics.confusion_matrix(y_test,y_pred,labels=[1,0])
print("\nConfusion Matrix")
print(confusion_array)
print(metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# developing an ROC curve - credit : https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
import matplotlib.pyplot as plt

probabilities = (lr.predict_proba(x_test))[:,1]
fpr_rate, tpr_rate, temp = metrics.roc_curve(y_test, probabilities)
plt.plot(fpr_rate, tpr_rate, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

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
