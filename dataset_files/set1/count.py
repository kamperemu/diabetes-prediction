import csv
count_0 = 0
count_1 = 0
with open("main.csv", mode="r") as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        if row["diabetes"] == "0":
            count_0 += 1
        else:
            count_1 += 1
print(count_0)
print(count_1)