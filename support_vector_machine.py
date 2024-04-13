# input.py file contains input functions that are common to all models.
from input import load_dataset, preprocess, XY_split
from sklearn import svm
from sklearn import metrics 

# loading and preprocessing the data
data = preprocess(load_dataset('dataset_files/main.csv'))
# reading the data for debugging
# print(data.head())
# splitting the data for the machine learning model
x_train, x_test, y_train, y_test = XY_split(data)

# creating the logistic regresesion model
# variable named svmm as svm already exists as an object of a module
svmm = svm.SVC()
svmm.fit(x_train, y_train)

# different kinds of data metrics for the model
score = svmm.score(x_test, y_test)
print(score)
