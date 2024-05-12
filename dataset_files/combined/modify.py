import csv
count_1 = 0
with open("original.csv", mode="r") as file:
    with open("main.csv", mode="w") as new_file:
        csv_reader = csv.DictReader(file)
        csv_writer = csv.DictWriter(new_file, fieldnames=["gender","age","heart_disease","smoking_history","bmi","diabetes"])
        csv_writer.writeheader()
        for row in csv_reader:
            if row["diabetes"] == "1":
                if count_1 != 43846:
                    if row["gender"] == "0":
                        row["gender"] = "Female"
                    if row["gender"] == "1":
                        row["gender"] = "Male"
                    if row["gender"] == "2":
                        row["gender"] = "Other"
                    if row["smoking_history"] == "0":
                        row["smoking_history"] = "never"
                    if row["smoking_history"] == "1":
                        row["smoking_history"] = "current"
                    csv_writer.writerow(row)
                    count_1 += 1
            else:
                if row["gender"] == "0":
                        row["gender"] = "Female"
                if row["gender"] == "1":
                    row["gender"] = "Male"
                if row["gender"] == "2":
                    row["gender"] = "Other"
                if row["smoking_history"] == "0":
                    row["smoking_history"] = "never"
                if row["smoking_history"] == "1":
                    row["smoking_history"] = "current"
                csv_writer.writerow(row)
            