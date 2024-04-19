from quadratic_discriminant_analysis import basicQDA, crossvalidateQDA
from support_vector_machine import basicSVM, crossvalidateSVM
from decision_tree import basicDT, crossvalidateDT
from logistic_regression import basicLR, crossvalidateLR
from sklearn import metrics
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay

import helper

data = helper.loadXYtraintest()

qda = basicQDA()
svm = basicSVM()
dt = basicDT()
lr = basicLR()

ax = plt.gca()
RocCurveDisplay.from_estimator(qda, data['x_test'], data['y_test'], ax=ax, name='QDA')
RocCurveDisplay.from_estimator(svm, data['x_test'], data['y_test'], ax=ax, name='SVM')
RocCurveDisplay.from_estimator(dt, data['x_test'], data['y_test'], ax=ax, name='DT')
RocCurveDisplay.from_estimator(lr, data['x_test'], data['y_test'], ax=ax, name='LR')
plt.show()

qdaPred = qda.predict(data['x_test'])
svmPred = svm.predict(data['x_test'])
dtPred = dt.predict(data['x_test'])
lrPred = lr.predict(data['x_test'])

qdaScore = [metrics.accuracy_score(data['y_test'], qdaPred), metrics.precision_score(data['y_test'], qdaPred), metrics.recall_score(data['y_test'], qdaPred), metrics.f1_score(data['y_test'], qdaPred), crossvalidateQDA()]
svmScore = [metrics.accuracy_score(data['y_test'], svmPred), metrics.precision_score(data['y_test'], svmPred), metrics.recall_score(data['y_test'], svmPred), metrics.f1_score(data['y_test'], svmPred), crossvalidateSVM()]
dtScore = [metrics.accuracy_score(data['y_test'], dtPred), metrics.precision_score(data['y_test'], dtPred), metrics.recall_score(data['y_test'], dtPred), metrics.f1_score(data['y_test'], dtPred), crossvalidateDT()]
lrScore = [metrics.accuracy_score(data['y_test'], lrPred), metrics.precision_score(data['y_test'], lrPred), metrics.recall_score(data['y_test'], lrPred), metrics.f1_score(data['y_test'], lrPred), crossvalidateLR()]

# little help from https://www.geeksforgeeks.org/bar-plot-in-matplotlib/ to plot the bar graph

fig = plt.figure(figsize=(10, 5))
br1 = np.arange(len(qdaScore))
br2 = [x + 0.2 for x in br1]
br3 = [x + 0.2 for x in br2]
br4 = [x + 0.2 for x in br3]

plt.bar(br1, qdaScore, color ='r', width = 0.2, edgecolor ='grey', label ='QDA')
plt.bar(br2, svmScore, color ='g', width = 0.2, edgecolor ='grey', label ='SVM')
plt.bar(br3, dtScore, color ='b', width = 0.2, edgecolor ='grey', label ='DT')
plt.bar(br4, lrScore, color ='y', width = 0.2, edgecolor ='grey', label ='LR')

plt.xlabel('metrics', fontweight ='bold', fontsize = 15)
plt.ylabel('Scores', fontweight ='bold', fontsize = 15)
plt.xticks([r + 0.2 for r in range(len(qdaScore))], ['Accuracy', 'Precision', 'Recall', 'F1', 'Cross Validation'])
plt.legend()
plt.show()