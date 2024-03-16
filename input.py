import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_dataset(dataset_file):
    return pd.read_csv(dataset_file)

def preprocess(data):
    data.replace({'gender':'Female'}, 0, inplace=True)
    data.replace({'gender':'Male'}, 1, inplace=True)
    # https://www.geeksforgeeks.org/how-to-convert-categorical-string-data-into-numeric-in-python/
    le = LabelEncoder()
    label = le.fit_transform(data['smoking_history'])
    data['smoking_history'] = label
    return data

def XY_split(data):

    Xcols = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    X = data[Xcols]
    Y = data['diabetes']

    return train_test_split(X, Y, test_size=0.3, random_state=42)
