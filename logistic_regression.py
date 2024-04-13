from input import load_dataset, preprocess, XY_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 

data = preprocess(load_dataset('dataset_files/main.csv'))
# print(data.head())

X_train, X_test, y_train, y_test = XY_split(data)

lr = LogisticRegression(max_iter = 300, class_weight='balanced')
lr.fit(X_train, y_train)

score = lr.score(X_test, y_test)
print(score)
# print(lr.feature_names_in_)
print(lr.coef_)
