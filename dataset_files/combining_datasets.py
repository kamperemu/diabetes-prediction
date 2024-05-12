import csv
import pandas as pd


columns_original = ["gender","age","heart_disease","smoking_history","bmi","diabetes"]
df = pd.read_csv('set1/original.csv')
df = df[columns_original]

columns_new = ["Sex", "Age","HeartDiseaseorAttack","Smoker","BMI","Diabetes_binary"]
df_new = pd.read_csv('set2/original.csv')
df_new = df_new[columns_new]
new_names = {"Sex":"gender", "Age":"age", "HeartDiseaseorAttack":"heart_disease", "Smoker":"smoking_history","BMI":"bmi", "Diabetes_binary":"diabetes"}
df_new = df_new.rename(columns=new_names)

df_final = pd.concat([df, df_new], axis = 0)
df_final.to_csv("combined/original.csv", index = False)
            