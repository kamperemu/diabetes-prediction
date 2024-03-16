from input import load_dataset, preprocess, XY_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 

data = preprocess(load_dataset('dataset_files/main.csv'))
print(data.head())
X_train, X_test, y_train, y_test = XY_split(data)

lr = LogisticRegression()
