# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:12:03 2017

@author: yama
"""

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# #############################################################################
# Generate sample data
f = open("E:\\Projects-SonNV\\SPT\\taxiDemands\\SanFrancisco\\period96-center20-day15-21-Train.txt", "r")
X = np.empty([0,2], dtype = int)

y = np.array([])

for i in range(1920):
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0]), int(my_lines[1])]])
    X = np.concatenate((X, t), axis =0 )
    y = np.concatenate((y, np.array([int(my_lines[2])])), axis = 0)
f.close()
print(X)
print(y)
print(X[0])
print(y[0])
print(type(X[0]))
print(type(y[0]))

f = open("E:\\Projects-SonNV\\SPT\\taxiDemands\\SanFrancisco\\period96-center20-day22-28-Test.txt", "r")
X_test = np.empty([0,2], dtype = int)

y_test = np.empty([0,1], dtype = int)

for i in range(1920):
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0]), int(my_lines[1])]])
    n = np.array([[int(my_lines[2])]])
    X_test = np.concatenate((X_test, t), axis =0 )
    y_test = np.concatenate((y_test, n), axis =0 )
f.close()
# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3, degree = 4)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)


diabetes_y_pred = svr_rbf.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, diabetes_y_pred))
# #############################################################################
# Look at the results

def plot_figs(fig_num, elev, azim, X_train, clf):
    fig = plt.figure(1, figsize=(4, 3))
    ax = Axes3D(fig, elev=elev, azim=azim)
    ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='+')
    surf = ax.plot_surface(X[:, 0], X[:, 1], y, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.scatter(X, y, color='r', label='data') 
    plt.scatter(X_test.T, y_test.T, c = 'y', s = 40, label = 'Test samples')
    plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
    plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    #plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    
lw = 2
elev = 43.5
azim = -110
plot_figs(1, elev, azim, X, svr_rbf)