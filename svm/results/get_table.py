#!/usr/bin/env python2

import re
import sys
import matplotlib.pylab as plt

res_file = open('./all_comb.txt', 'r')
file_str = res_file.read()


res = {}
for m in re.finditer(r'\((\d+\.\d+), (\d+\.\d+)\): (\d+\.\d+)',
                     file_str):
    C, tau = float(m.group(1)), float(m.group(2))
    estimator = float(m.group(3))
    if tau not in res: res[tau] = {}
    res[tau][C] = estimator

print len(res)

tau_values = res.keys()
tau_values.sort()
tau = tau_values[0]
C_values = res[tau].keys()
C_values.sort()

data = []
for tau in tau_values:
    row = []
    for C in C_values:
        row.append(res[tau][C])
    data.append(row)

print data


# find min
min_ind = [0, 0]
for i in xrange(len(data)):
    for j in xrange(len(data[i])):
        mi, mj = min_ind
        if data[i][j] < data[mi][mj]:
            min_ind = [i,j]

print data[min_ind[0]][min_ind[1]]

# Take first row into account
min_ind[0] += 1
min_ind = tuple(min_ind)

clust_data = data

colLabels = C_values
rowLabels = tau_values
nrows, ncols = len(clust_data)+1, len(colLabels)
hcell, wcell = 0.2, 0.82
hpad, wpad = 0, 0.1 
fig=plt.figure(figsize=(ncols*wcell+wpad, nrows*hcell+hpad))
ax = fig.add_subplot(111)
ax.axis('off')

colLabels[0] = 'C=' + str(colLabels[0])
rowLabels[0] = 'tau=' + str(rowLabels[0])
table = ax.table(cellText=clust_data,
                     colLabels=colLabels,
                     rowLabels=rowLabels,
                     loc='center',
                     rowLoc='right' )
cell = table.properties()['celld'][min_ind]
# yellow
cell.set_color( (1,1,0.2) )
plt.show()
