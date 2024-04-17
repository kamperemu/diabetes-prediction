import helper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def basicLR():
    # load the data
    data = helper.loadXYtraintest()

    # creating the logistic regresesion model
    lr = LogisticRegression(max_iter = 300, class_weight='balanced')
    lr.fit(data['x_train'], data['y_train'])
    data['y_pred'] = lr.predict(data['x_test'])

    # output of data metrics of the model
    helper.print_common_data_metrics(data['y_test'], data['y_pred'])


basicLR()

def crossvalidateLR():
    # load the data
    X, Y = helper.loadXY()

    # creating the logistic regresesion model
    lr = LogisticRegression(max_iter = 300, class_weight='balanced')

    # cross validation
    scores = cross_val_score(lr, X, Y, cv=5)
    print(scores)

crossvalidateLR()


"""
# visual display for data meterics

# developing an ROC curve - credit : https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
import matplotlib.pyplot as plt

probabilities = (lr.predict_proba(x_test))[:,1]
fpr_rate, tpr_rate, temp = metrics.roc_curve(y_test, probabilities)
plt.plot(fpr_rate, tpr_rate, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()



import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd


df_cm = pd.DataFrame(confusion_array, range(2), range(2))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
"""
