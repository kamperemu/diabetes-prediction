import csv
count_0 = 0
with open("original.csv", mode="r") as file:
    with open("main.csv", mode="w") as new_file:
        csv_writer = csv.writer(new_file)
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[-1] == "0":
                if count_0 != 8500:
                    csv_writer.writerow(row)
                    count_0 += 1
            else:
                csv_writer.writerow(row)
            