import csv
count_1 = 0
with open("original.csv", mode="r") as file:
    with open("main.csv", mode="w") as new_file:
        csv_writer = csv.writer(new_file)
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] == "1":
                if count_1 != 35346:
                    csv_writer.writerow(row)
                    count_1 += 1
            else:
                csv_writer.writerow(row)
            