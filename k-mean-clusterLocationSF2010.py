# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:25:44 2017

@author: yama
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
#np.random.seed(11)
#means = [[2, 2], [8, 3], [3, 6]]
#cov = [[1, 0], [0, 1]]
#N = 500
#X0 = np.random.multivariate_normal(means[0], cov, N)
#X1 = np.random.multivariate_normal(means[1], cov, N)
#X2 = np.random.multivariate_normal(means[2], cov, N)
#
#X = np.concatenate((X0, X1, X2), axis = 0)
#K = 195
#print(X)
##
#original_label = np.asarray([0]*N + [1]*N + [2]*N).T
dt = np.dtype([])
day = 0;
X = np.loadtxt('E:\\Projects-SonNV\\SPT\\taxiDemands\\SanFrancisco\\allPickupLocation-day1-day21.txt')
clusters= 250
def kmeans_display(X, label):
    K = np.amax(label) + 1
#    X0 = X[label == 0, :]
#    X1 = X[label == 1, :]
#    X2 = X[label == 2, :]
#    X3 = X[label == 3, :]
#    X4 = X[label == 4, :]
#    X5 = X[label == 5, :]
#    X6 = X[label == 6, :]
#    X7 = X[label == 7, :]
#    X8 = X[label == 8, :]
#    X9 = X[label == 9, :]
#    X10 = X[label == 10, :]
#    X11 = X[label == 11, :]
#    X12 = X[label == 12, :]
#    X13 = X[label == 13, :]
#    X14 = X[label == 14, :]
#    X15 = X[label == 15, :]
#    X16 = X[label == 16, :]
#    X17= X[label == 17, :]
#    X18 = X[label == 18, :]
#    X19 = X[label == 19, :]
    
#    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
#    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
#    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)
#    plt.plot(X3[:, 0], X3[:, 1], 'cp', markersize = 4, alpha = .8)
#    plt.plot(X4[:, 0], X4[:, 1], 'mh', markersize = 4, alpha = .8)
#    plt.plot(X5[:, 0], X5[:, 1], 'y>', markersize = 4, alpha = .8)
#    plt.plot(X6[:, 0], X6[:, 1], 'k<', markersize = 4, alpha = .8)
#    plt.plot(X7[:, 0], X7[:, 1], 'r*', markersize = 4, alpha = .8)
#    plt.plot(X8[:, 0], X8[:, 1], 'rH', markersize = 4, alpha = .8)
#    plt.plot(X9[:, 0], X9[:, 1], 'b+', markersize = 4, alpha = .8)
#    plt.plot(X10[:, 0], X10[:, 1], 'gx', markersize = 4, alpha = .8)
#    plt.plot(X11[:, 0], X11[:, 1], 'cD', markersize = 4, alpha = .8)
#    plt.plot(X12[:, 0], X12[:, 1], 'md', markersize = 4, alpha = .8)
#    plt.plot(X13[:, 0], X13[:, 1], 'ys', markersize = 4, alpha = .8)
#    plt.plot(X14[:, 0], X14[:, 1], 'k^', markersize = 4, alpha = .8)
#    plt.plot(X15[:, 0], X15[:, 1], 'r4', markersize = 4, alpha = .8)
#    plt.plot(X16[:, 0], X16[:, 1], 'g3', markersize = 4, alpha = .8)
#    plt.plot(X17[:, 0], X17[:, 1], 'r2', markersize = 4, alpha = .8)
#    plt.plot(X18[:, 0], X18[:, 1], 'b1', markersize = 4, alpha = .8)
#    plt.plot(X19[:, 0], X19[:, 1], 'y1', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.figure(figsize = (15, 8))
    plt.plot()
    
    fn = 'kmeans_allPoint-day1-day21-' + str(clusters) + '.png'
    plt.show()
    
#def kmeans_init_centers(X, k):
#    # randomly pick k rows of X as initial centers
#    return X[np.random.choice(X.shape[0], k, replace=False)]
#
#def kmeans_assign_labels(X, centers):
#    # calculate pairwise distances btw data and centers
#    D = cdist(X, centers)
#    # return index of the closest center
#    return np.argmin(D, axis = 1)
#
#def kmeans_update_centers(X, labels, K):
#    centers = np.zeros((K, X.shape[1]))
#    for k in range(K):
#        # collect all points assigned to the k-th cluster 
#        Xk = X[labels == k, :]
#        # take average
#        centers[k,:] = np.mean(Xk, axis = 0)
#    return centers
#
#def has_converged(centers, new_centers):
#    # return True if two sets of centers are the same
#    return (set([tuple(a) for a in centers]) == 
#        set([tuple(a) for a in new_centers]))
#def kmeans(X, K):
#    centers = [kmeans_init_centers(X, K)]
#    labels = []
#    it = 0 
#    while True:
#        labels.append(kmeans_assign_labels(X, centers[-1]))
#        new_centers = kmeans_update_centers(X, labels[-1], K)
#        if has_converged(centers[-1], new_centers):
#            break
#        centers.append(new_centers)
#        it += 1
#    return (centers, labels, it)
#kmeans_display(X, original_label)
#(centers, labels, it) = kmeans(X, K)
#print('Centers found by our algorithm:')
#print(centers[-1])
#print(centers)
#print(type(centers))
#print(it)
#
#kmeans_display(X, labels[-1])
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters= clusters, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)
kmeans_display(X, pred_label)
f = open('E:\\Projects-SonNV\\SPT\\taxiDemands\\SanFrancisco\\centerClustering-allPoints-' + str(clusters) + '-day1-day21.txt', 'w')
for x in np.nditer(kmeans.cluster_centers_):
    f.write(str(x) + '\n')
f.write("-1")
f.close()



