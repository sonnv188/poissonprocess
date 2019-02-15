# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(4)
from sklearn import datasets, linear_model
city = 'CallCenter311'
nbr = 288
f = open("E:\\Projects-SonNV\\SPT\\taxiDemands\\" + city + "\\reqs-period" + str(nbr) +"-Train.txt", "r")
X = np.empty([0,1], dtype = int)

y = np.empty([0,1], dtype = int)

for i in range(nbr):
    if (i < 94 or i > 202):
        f.readline()
        continue
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0])]])
    n = np.array([[int(my_lines[1])]])
    X = np.concatenate((X, t), axis =0 )
    y = np.concatenate((y, n), axis =0 )
f.close()
f = open("E:\\Projects-SonNV\\SPT\\taxiDemands\\" + city + "\\reqs-period" + str(nbr) +"-Test.txt", "r")
X_test = np.empty([0,1], dtype = int)

y_test = np.empty([0,1], dtype = int)

for i in range(nbr):
    if (i < 94 or i > 202):
        f.readline()
        continue
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0])]])
    n = np.array([[int(my_lines[1])]])
    X_test = np.concatenate((X_test, t), axis =0 )
    y_test = np.concatenate((y_test, n), axis =0 )
f.close()


def buildX(X, d):
    res = np.ones((X.shape[0], 1))
    for i in range(1, d+1):
        res = np.concatenate((res, X**i), axis = 1)
    return res 

def myfit(X, y, d):
    print("==============")
    Xbar = buildX(X, d)
    regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    regr.fit(Xbar, y)

    # Make predictions using the testing set
    Xbar_test = buildX(X_test, d)
    diabetes_y_pred = regr.predict(Xbar_test)
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, diabetes_y_pred))

    w = regr.coef_
    # Display result
    #x0 = np.linspace(0, 96, 96, endpoint=True)
    x0 = np.empty([0, 1])
    for i in range(nbr):
        t = np.array([[i]])
        x0 = np.concatenate((x0, t), axis =0 )
    y0 = np.zeros_like(x0)
    for i in range(d+1):
        y0 += w[0][i]*x0**i

    # Draw the fitting line 
    plt.scatter(X.T, y.T, c = 'r', s = 40, label = 'Training samples')     # data 

    plt.scatter(X_test.T, y_test.T, c = 'y', s = 40, label = 'Test samples')     # data 
    
    plt.plot(x0, y0, 'b', linewidth = 2, label = "Trained model")   # the fitting line
    plt.xticks([], [])
    plt.yticks([], [])
    
   
    
    
    str0 = 'Degree = ' + str(d) + ': '
    plt.title(str0)
    #plt.axis([20,80, -10, 200])
    plt.axis([60,230, -10, 100])
    plt.legend(loc="best")
    
    fn = 'lin_'+ city + '_' + str(d) + '.png'
    
    plt.xlabel('$x$', fontsize = 20);
    plt.ylabel('$y$', fontsize = 20);
    
    plt.savefig(fn, bbox_inches='tight', dpi = 200)
    
    plt.show()
    print(w)
myfit(X, y, 1)

