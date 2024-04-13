from input import load_dataset, preprocess, XY_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

data = preprocess(load_dataset('dataset_files/main.csv'))
# print(data.head())

X_train, X_test, y_train, y_test = XY_split(data)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

score = qda.score(X_test, y_test)
print(score)