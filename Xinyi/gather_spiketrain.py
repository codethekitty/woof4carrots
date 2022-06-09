import sys,glob,os
import numpy as np
import csv

animal_class = {}
animal_class_index = {}
id = 0
with open("../train_set spiketrain/exp_animal_list.csv", 'r') as fin:
    r = csv.reader(fin)
    for row in r:
        if row[0] == 'animal':
            continue
        animal_class[row[0]] = row[1]
        if row[1] not in animal_class_index:
            animal_class_index[row[1]] = id
            id += 1
    # print(animal_class)

input_dir = '../train_set spiketrain'
zero_list = [0] * 3080
# max_spike = 0
with open("timestamp.csv", 'w') as f:
    csv_writer = csv.writer(f)
    for filename in os.listdir(input_dir):
        if filename.split(".")[1] != 'npy':
            continue
        data = np.load(os.path.join(input_dir, filename), allow_pickle=True)[0]
        for train in data:
            animal = filename.split('_')[0]
            row = [animal_class_index[animal_class[animal]]]
            for spike in train:
                row.append(float(spike))
            # csv_writer.writerow(row)
            # print(row)
            row.extend(zero_list)
            # print(row)
            for spike in train:
                # if spike > max_spike:
                #     max_spike = spike
                row[round(float(spike) * 10)+2] += 1
            # print(max_spike)
            csv_writer.writerow(row)

