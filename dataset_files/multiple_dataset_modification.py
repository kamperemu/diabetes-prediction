import csv
import pandas as pd


columns_original = ["gender","age","heart_disease","smoking_history","bmi","diabetes"]
df = pd.read_csv('original.csv')
df = df[columns_original]

columns_new = ["Sex", "Age","HeartDiseaseorAttack","Smoker","BMI","Diabetes_binary"]
df_new = pd.read_csv('original_dataset2.csv')
df_new = df_new[columns_new]
new_names = {"Sex":"gender", "Age":"age", "HeartDiseaseorAttack":"heart_disease", "Smoker":"smoking_history","BMI":"bmi", "Diabetes_binary":"diabetes"}
df_new = df_new.rename(columns=new_names)

df_final = pd.concat([df, df_new], axis = 0)
df_final.to_csv("original_combined.csv", index = False)


with open("original_combined.csv", mode="r") as file:
    with open("main_combined.csv", mode="w") as new_file:
        csv_writer = csv.writer(new_file)
        csv_reader = csv.reader(file)
        for row in csv_reader:
            csv_writer.writerow(row)
            