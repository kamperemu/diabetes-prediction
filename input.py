import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_dataset(dataset_file):
    return pd.read_csv(dataset_file)

def preprocess(data):

    # refered to https://www.geeksforgeeks.org/how-to-convert-categorical-string-data-into-numeric-in-python/
    # we do not keep the encoder object because we do not need them (unless we need to report the effects of smoking history or gender)
    
    # replaces textual data with numerical values for gender
    gle = LabelEncoder()
    glabel = gle.fit_transform(data['gender'])
    data['gender'] = glabel
    
    # replaces textual data with numerical values for smoking history
    sle = LabelEncoder()
    slabel = sle.fit_transform(data['smoking_history'])
    data['smoking_history'] = slabel

    # returns the processed data
    return data

def XY_split(data, feature_selection, dataset):
    
    # choose the data columns to be used as features
    if dataset == "set1":
        Xcols = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        if feature_selection == 1:
            Xcols = ['gender', 'age', 'heart_disease', 'smoking_history', 'bmi']
        elif feature_selection == 2:
            Xcols = ['age', 'bmi', "hypertension"]
    if dataset == "set2":
        Xcols = ['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income']
        if feature_selection == 1:
            Xcols = ["Sex", "Age", "HeartDiseaseorAttack", "Smoker", "BMI"]
        elif feature_selection == 2:
            Xcols = ["PhysHlth", "MentHlth", "BMI", "DiffWalk", "HighBP", "GenHlth", "Age", "HeartDiseaseorAttack"]
    if dataset == "combined":
        Xcols = ['gender', 'age', 'heart_disease', 'smoking_history', 'bmi']
        if feature_selection == 1:
            Xcols = ['gender', 'age', 'heart_disease', 'smoking_history', 'bmi']
        elif feature_selection == 2:
            Xcols = ["bmi", "heart_disease", "age"]

    X = data[Xcols]
    # choose the data column to be used as target
    if dataset == "set2":
        Y = data['Diabetes_binary']
    else:
        Y = data['diabetes']

    # split the data into training and testing sets
    return X, Y


def traintest_split(data, feature_selection, dataset):
    X, Y = XY_split(data, feature_selection, dataset)
    return train_test_split(X, Y, test_size=0.3, random_state=42)
