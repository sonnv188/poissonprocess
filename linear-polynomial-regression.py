# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:06:05 2018

@author: Son
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

f = open("E:\\Projects-SonNV\\SPT\\taxiDemands\\artificalData\\reqs-period288-Train.txt", "r")
X = np.empty([0,1], dtype = int)

y = np.empty([0,1], dtype = int)

for i in range(288):
    my_lines = f.readline().split(" ")
#    if i < 50:
#        continue;
    t = np.array([[int(my_lines[0])]])
    n = np.array([[int(my_lines[1])]])
    X = np.concatenate((X, t), axis =0 )
    y = np.concatenate((y, n), axis =0 )
f.close()
f = open("E:\\Projects-SonNV\\SPT\\taxiDemands\\artificalData\\reqs-period288-Test.txt", "r")
X_test = np.empty([0,1], dtype = int)

y_test = np.empty([0,1], dtype = int)

for i in range(288):
    my_lines = f.readline().split(" ")
#    if i < 50:
#        continue;
    t = np.array([[int(my_lines[0])]])
    n = np.array([[int(my_lines[1])]])
    X_test = np.concatenate((X_test, t), axis =0 )
    y_test = np.concatenate((y_test, n), axis =0 )
f.close()
 

from sklearn.linear_model import LinearRegression
linear_reg1 = LinearRegression()
linear_reg1.fit(X, y)
y_pred_linear = linear_reg1.predict(X_test)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 9)
X_poly = poly_reg.fit_transform(X)
Xtest_poly = poly_reg.fit_transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

y_pred_poly = poly_reg.predict(Xtest_poly)

# The coefficients
print('Linear-Coefficients: \n', linear_reg1.coef_)
# The mean squared error
print("Linear-Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred_linear))
# Explained variance score: 1 is perfect prediction
print('Linear-Variance score: %.2f' % r2_score(y_test, y_pred_linear))

# The coefficients
print('LinearPoly-Coefficients: \n', poly_reg.coef_)
# The mean squared error
print("LinearPoly-Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred_poly))
# Explained variance score: 1 is perfect prediction
print('LinearPoly-Variance score: %.2f' % r2_score(y_test, y_pred_poly))

lw = 2
plt.scatter(X, y, color='r', label='data') 
plt.scatter(X_test.T, y_test.T, c = 'y', s = 40, label = 'Test samples')
#plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X_test, y_pred_linear, color='c', lw=lw, label='Linear model')
plt.plot(X_test, y_pred_poly, color='navy', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Linear and Polynomia Regression')
plt.legend()

fn = 'linear-polyregresion_AFD.png'
plt.xlabel('$x$', fontsize = 20);
plt.ylabel('$y$', fontsize = 20);
plt.savefig(fn, bbox_inches='tight', dpi = 200)
    
plt.show()

