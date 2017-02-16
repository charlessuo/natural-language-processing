#!/usr/bin/env python3
import os

class_to_id = {'rec.sport.hockey': 15, 'rec.autos': 0, 'sci.crypt': 1, 'sci.electronics': 2, 'sci.space': 3, 'comp.sys.ibm.pc.hardware': 4, 'misc.forsale': 5, 'comp.graphics': 6, 'rec.sport.baseball': 8, 'alt.atheism': 9, 'talk.politics.mideast': 10, 'soc.religion.christian': 12, 'talk.politics.misc': 13, 'comp.os.ms-windows.misc': 14, 'talk.religion.misc': 16, 'talk.politics.guns': 11, 'comp.windows.x': 17, 'sci.med': 7, 'rec.motorcycles': 18, 'comp.sys.mac.hardware': 19}

id_to_class = {0: 'rec.autos', 1: 'sci.crypt', 2: 'sci.electronics', 3: 'sci.space', 4: 'comp.sys.ibm.pc.hardware', 5: 'misc.forsale', 6: 'comp.graphics', 7: 'sci.med', 8: 'rec.sport.baseball', 9: 'alt.atheism', 10: 'talk.politics.mideast', 11: 'talk.politics.guns', 12: 'soc.religion.christian', 13: 'talk.politics.misc', 14: 'comp.os.ms-windows.misc', 15: 'rec.sport.hockey', 16: 'talk.religion.misc', 17: 'comp.windows.x', 18: 'rec.motorcycles', 19: 'comp.sys.mac.hardware'}

votes = [[0 for i in range(20)] for j in range(7532)]

for filename in os.listdir(os.getcwd() + '/results'):
    with open(os.getcwd() + '/results/' + filename, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            _, class_ = line.strip().split(',')
            votes[i][class_to_id[class_]] += 1


with open('./results/ensemble.csv', 'w') as out:
    out.write('id,newsgroup\n')
    for i in range(len(votes)):
        pred = votes[i].index(max(votes[i]))
        out.write(str(i) + ',' + id_to_class[pred] + '\n')

