#!/usr/bin/env python3
import os

class_to_id = {'place': 0, 'movie': 1, 'person': 2, 'company': 3, 'drug': 4}
id_to_class = {0: 'place', 1: 'movie', 2: 'person', 3: 'company', 4: 'drug'}

votes = [[0 for i in range(5)] for j in range(2862)]

for filename in os.listdir(os.getcwd() + '/results'):
    with open(os.getcwd() + '/results/' + filename, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            _, class_ = line.strip().split(',')
            votes[i][class_to_id[class_]] += 1


with open('./results/ensemble.csv', 'w') as out:
    out.write('id,type\n')
    for i in range(len(votes)):
        pred = votes[i].index(max(votes[i]))
        out.write(str(i) + ',' + id_to_class[pred] + '\n')

