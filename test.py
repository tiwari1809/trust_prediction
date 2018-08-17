import scipy.io
import time
import sys
from collections import *
import numpy as np
import math
import copy

outmat = scipy.io.loadmat('predicted_out.mat')
mat = scipy.io.loadmat('epinion_trust_with_timestamp.mat')
# print max(np.array(outmat['trust'].reshape(1,8519*8519)))
dd = 0
predicted = np.zeros([len(outmat['trust']), len(outmat['trust'][0])])
for i in xrange(len(outmat['trust'])):
    # dd = max(dd, max(outmat['trust'][i]))
    for j in xrange(len(outmat['trust'][i])):
        if outmat['trust'][i][j] > 0.15:
            predicted[i][j] = 1
            dd+=1
        # else: predicted[i][j] = 0
print dd

x = 8
N = 0
gg = np.zeros([len(outmat['trust']), len(outmat['trust'][0])])
for i in xrange(len(mat['trust'])):
    if mat['trust'][i][2] == x:
        gg[mat['trust'][i][0]-1][mat['trust'][i][1]-1] = 1
        #vv.append([mat['trust'][i][0]-1, mat['trust'][i][1]-1])
        N+=1
        # print xx
actual = np.array(gg)
print N
print np.sum((actual - predicted)**2)

# print outmat['trust']

# print len(outmat['trust']),len(outmat['trust'][0])
