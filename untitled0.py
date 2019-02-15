import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(4)
from sklearn import datasets, linear_model
# #############################################################################
# Generate sample data
city = 'CallCenter311'
nbr = 288

f = open("E:\\Projects-SonNV\\SPT\\taxiDemands\\" + city + "\\reqs-period" + str(nbr) +"-Train.txt", "r")
X = np.empty([0,1], dtype = int)

y = np.array([])
y2 = np.empty([0,1], dtype = int)

for i in range(nbr):
    if (i < 94 or i > 202):
        f.readline()
        continue
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0])]])
    n = np.array([int(my_lines[1])])
    n2 = np.array([[int(my_lines[1])]])
    X = np.concatenate((X, t), axis =0 )
    y = np.concatenate((y, np.array([int(my_lines[1])])), axis = 0)
    y2 = np.concatenate((y2, n2), axis =0 )
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
# #############################################################################
# Fit regression model
def buildX(X, d):
    res = np.ones((X.shape[0], 1))
    for i in range(1, d+1):
        res = np.concatenate((res, X**i), axis = 1)
    return res 
def myfit(X, y, d):
    print("==============")
    Xbar = buildX(X, d)
    regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    regr.fit(Xbar, y2)

    # Make predictions using the testing set
    Xbar_test = buildX(X_test, d)
    diabetes_y_pred = regr.predict(Xbar_test)
    
    # The coefficients
    print('Poly Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Poly Mean squared error: %.2f"
          % mean_squared_error(y_test, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Poly Variance score: %.2f' % r2_score(y_test, diabetes_y_pred))

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
    x02 = 300*x0
    plt.plot(x02, y0, 'b', color='cornflowerblue', lw=lw, label='Polynomial model')   # the fitting line
myfit(X, y, 4)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3, degree = 4)
#svr_poly = SVR(kernel='poly', C=1e3, degree=4)

y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
##y_poly = svr_poly.fit(X, y).predict(X)


diabetes_y_pred_rbf = svr_rbf.predict(X_test)

#The mean squared error
print("Mean squared error RBF: %.2f"
      % mean_squared_error(y_test, diabetes_y_pred_rbf))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, diabetes_y_pred_rbf))


diabetes_y_pred_lin = svr_lin.predict(X_test)
#The mean squared error
print("Mean squared error lin: %.2f"
      % mean_squared_error(y_test, diabetes_y_pred_lin))
# Explained variance score: 1 is perfect prediction
print('Variance score lin: %.2f' % r2_score(y_test, diabetes_y_pred_lin))


# =============================================================================
# diabetes_y_pred_poly = svr_poly.predict(X_test)
# ## The mean squared error
# print("Mean squared error poly: %.2f"
#       % mean_squared_error(y_test, diabetes_y_pred_poly))
# # Explained variance score: 1 is perfect prediction
# print('Variance score poly: %.2f' % r2_score(y_test, diabetes_y_pred_poly))
# =============================================================================

# #############################################################################
# Look at the results
lw = 2
X2 = 300* X
X2_test = 300*X_test

plt.scatter(X2, y, color='r', label='Training samples') 
plt.scatter(X2_test.T, y_test.T, c = 'y', s = 40, label = 'Test samples')

plt.plot(X2, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X2, y_lin, color='springgreen', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.axis([28400, 60900, -10, 60])

plt.title('Support Vector Regression')
plt.legend(fontsize = 'x-small')

fn = 'allreg_' + city + '_288.png'    
plt.xlabel('time (second unit)', fontsize = 11);
plt.ylabel('rate', fontsize = 11);

plt.savefig(fn, bbox_inches='tight', dpi = 200)
plt.show()