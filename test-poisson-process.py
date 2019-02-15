# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:58:07 2017

@author: yama
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
# interval between two events is distributed as an exponential
delta_t = scipy.stats.expon.rvs(size=288)
print(delta_t)
#f = open('E:\Projects-SonNV\poissonprocess\poissonprocess.txt','w')
#for i in range(delta_t.size):
#    f.write(delta_t[i])
#f.close()
np.savetxt('E:\Projects-SonNV\poissonprocess\poissonprocess.txt', delta_t, delimiter=',')
t = np.cumsum(delta_t)
plt.hist(t/t.max(), 200)
plt.show() # see how much uniform it is
# perform the ks test (second value returned is the p-value)
scipy.stats.kstest(t/t.max(), 'uniform')