from gradient_boosting import basicGB, crossvalidateGB
from support_vector_machine import basicSVM, crossvalidateSVM
from naive_bayes import basicNB, crossvalidateNB
from logistic_regression import basicLR, crossvalidateLR
from sklearn import metrics
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay

import helper

data = helper.loadXYtraintest()

gb = basicGB()
svm = basicSVM()
nb = basicNB()
lr = basicLR()

ax = plt.gca()
RocCurveDisplay.from_estimator(gb, data['x_test'], data['y_test'], ax=ax, name='GB')
RocCurveDisplay.from_estimator(svm, data['x_test'], data['y_test'], ax=ax, name='SVM')
RocCurveDisplay.from_estimator(nb, data['x_test'], data['y_test'], ax=ax, name='NB')
RocCurveDisplay.from_estimator(lr, data['x_test'], data['y_test'], ax=ax, name='LR')
plt.show()

gbPred = gb.predict(data['x_test'])
svmPred = svm.predict(data['x_test'])
nbPred = nb.predict(data['x_test'])
lrPred = lr.predict(data['x_test'])

gbScore = [metrics.accuracy_score(data['y_test'], gbPred), metrics.precision_score(data['y_test'], gbPred), metrics.recall_score(data['y_test'], gbPred), metrics.f1_score(data['y_test'], gbPred), crossvalidateGB()]
svmScore = [metrics.accuracy_score(data['y_test'], svmPred), metrics.precision_score(data['y_test'], svmPred), metrics.recall_score(data['y_test'], svmPred), metrics.f1_score(data['y_test'], svmPred), crossvalidateSVM()]
nbScore = [metrics.accuracy_score(data['y_test'], nbPred), metrics.precision_score(data['y_test'], nbPred), metrics.recall_score(data['y_test'], nbPred), metrics.f1_score(data['y_test'], nbPred), crossvalidateNB()]
lrScore = [metrics.accuracy_score(data['y_test'], lrPred), metrics.precision_score(data['y_test'], lrPred), metrics.recall_score(data['y_test'], lrPred), metrics.f1_score(data['y_test'], lrPred), crossvalidateLR()]

# little help from https://www.geeksforgeeks.org/bar-plot-in-matplotlib/ to plot the bar graph

fig = plt.figure(figsize=(10, 5))
br1 = np.arange(len(gbScore))
br2 = [x + 0.2 for x in br1]
br3 = [x + 0.2 for x in br2]
br4 = [x + 0.2 for x in br3]

plt.bar(br1, gbScore, color ='r', width = 0.2, edgecolor ='grey', label ='GB')
plt.bar(br2, svmScore, color ='g', width = 0.2, edgecolor ='grey', label ='SVM')
plt.bar(br3, nbScore, color ='b', width = 0.2, edgecolor ='grey', label ='NB')
plt.bar(br4, lrScore, color ='y', width = 0.2, edgecolor ='grey', label ='LR')

plt.xlabel('metrics', fontweight ='bold', fontsize = 15)
plt.ylabel('Scores', fontweight ='bold', fontsize = 15)
plt.xticks([r + 0.2 for r in range(len(gbScore))], ['Accuracy', 'Precision', 'Recall', 'F1', 'Cross Validation'])
plt.legend()
plt.show()