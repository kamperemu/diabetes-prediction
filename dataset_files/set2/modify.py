import csv
count_0 = 0
with open("original.csv", mode="r") as file:
    with open("main.csv", mode="w") as new_file:
        csv_writer = csv.DictWriter(new_file, fieldnames=['Diabetes_binary', 'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income'])
        csv_reader = csv.DictReader(file)
        csv_writer.writeheader()
        for row in csv_reader:
            if row["Diabetes_binary"] == "0":
                if count_0 != 35346:
                    csv_writer.writerow(row)
                    count_0 += 1
            else:
                csv_writer.writerow(row)
            