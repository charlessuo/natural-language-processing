import numpy as np
import os

#ext = '.dev'
ext = '.test'

#filelist = [file_ for file_ in os.listdir('./results') if file_.endswith('.dev')]
filelist = ['bilstm_crf_embed128_pos32_orth32_cell64_layer1_ep12_v1' + ext, 
            'bilstm_crf_embed128_pos32_orth32_cell64_layer1_ep12_v2' + ext, 
            'bilstm_crf_embed128_pos32_orth32_cell64_layer1_ep12_v5' + ext]
d = {'B\n': 0, 'I\n': 1, 'O\n': 2, '\n': 3}
rev_d = {0: 'B\n', 1: 'I\n', 2: 'O\n', 3: '\n'}

rows = []

for file_ in filelist:
    row = []
    with open('./results/' + file_) as f:
        for line in f:
            row.append(d[line])
        rows.append(row)
rows = np.array(rows)

ans = []
for j in range(rows.shape[1]):
    counts = np.bincount(rows[:, j])
    ans.append(rev_d[np.argmax(counts)])

with open('./results/ensemble' + ext, 'w') as fout:
    for item in ans:
        fout.write(item)

