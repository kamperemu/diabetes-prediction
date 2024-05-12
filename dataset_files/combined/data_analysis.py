import csv
dict_of_data = {"age": {}, "bmi": {}}
with open("main_combined.csv", mode="r") as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        for key in row:
            if dict_of_data.get(key) == None:
                dict_of_data[key] = dict()
            if key == "age":
                ageFloor = int((float(row[key]) // 10) * 10)
                key2 = f"{ageFloor}-{ageFloor + 10}"
                if (key2) in dict_of_data[key]:
                    dict_of_data[key][key2] += 1
                else:
                    dict_of_data[key][key2] = 1
            elif key == "bmi":
                bmiFloor = int((float(row[key]) // 5) * 5)
                key2 = f"{bmiFloor}-{bmiFloor + 5}"
                if (key2) in dict_of_data[key]:
                    dict_of_data[key][key2] += 1
                else:
                    dict_of_data[key][key2] = 1
            else:
                if row[key] in dict_of_data[key]:
                    dict_of_data[key][row[key]] += 1
                else:
                    dict_of_data[key][row[key]] = 1

import matplotlib.pyplot as plt
for key in dict_of_data:
    # help from https://www.geeksforgeeks.org/plot-a-pie-chart-in-python-using-matplotlib/ to plot the pie chart
    labels = dict_of_data[key].keys()
    sizes = dict_of_data[key].values()
    plt.pie(sizes, labels=None)
    plt.title(key)
    plt.legend(title=key, labels = labels, loc="center left", bbox_to_anchor=(1, 0, 1, 1))
    plt.show()
