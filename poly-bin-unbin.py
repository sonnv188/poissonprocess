from patsy import dmatrix
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import random

city = "SanFrancisco"
dp = np.array([])
dp_lastest = np.array([])
minVar = 10000000.0
nPr = 240
nTime = 72000
f = open("E:/Project/pbts/git_project/SPT/taxiDemands/" + city + "/timePoints-1.txt", "r")
X = np.empty([])
timepoints = f.readline().split(',')
f.close()
timepoints = list(map(int, timepoints))
#read period file
f = open("E:/Project/Projects-SonNV/SPT/taxiDemands/" + city + "/reqs-period288-Train.txt", "r")
X = np.empty([0, 1], dtype=int)

y = np.empty([0, 1], dtype=int)


for i in range(nPr):
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0])]])
    n = np.array([[int(my_lines[1])]])
    X = np.concatenate((X, t), axis=0)
    y = np.concatenate((y, n), axis=0)
f.close()

f = open("E:/Project/Projects-SonNV/SPT/taxiDemands/" + city + "/reqs-period288-Test.txt", "r")
X_test = np.empty([0, 1], dtype=int)

y_test = np.empty([0, 1], dtype=int)

for i in range(nPr):
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0])]])
    n = np.array([[int(my_lines[1])]])
    X_test = np.concatenate((X_test, t), axis=0)
    y_test = np.concatenate((y_test, n), axis=0)
f.close()
print('a')
def estimation():
    global dp
    global X
    global y
    global X_test
    global y_test
    dp = np.sort(dp)
    # Generating cubic spline with 4 knots
    transformed_x2 = dmatrix("bs(train, knots=dp,degree =3, include_intercept=False)", {"train": X},
                             return_type='dataframe')

    # Fitting Generalised linear model on transformed dataset
    fit2 = sm.GLM(y, transformed_x2).fit()

    # Predictions on both splines
    pred2 = fit2.predict(dmatrix("bs(xp, knots=dp,degree =3, include_intercept=False)", {"xp": X_test},
                                 return_type='dataframe'))

    # Calculating RMSE values
    rms2 = sqrt(mean_squared_error(y_test, pred2))

    return rms2

#chia bin
def dividedPoints(l, u, c):
    global minErr
    global minVar
    global dp
    global dp_lastest
    global X
    global y
    global X_test
    global y_test
    if u - l < 10:
        return
    p = -1
    while p == -1 or p - l <= 4 or u - p <= 4:
        p = random.randint(l, u)

    #tp1 = np.array([])
    #tp2 = np.array([])
    #for t in timepoints:
        #if t >= l*300 and t < p*300:
            #tp1 = np.append(tp1, t)
        #elif t >= p*300 and t < u*300:
            #tp2 = np.append(tp2, t)
    #if tp1.size < 10 or tp2.size < 10:
        #return
    #tp1= tp1/((p-l)*300)
    #tp2= tp2/((u-p)*300)
    #d1, pval1 = kstest(tp1, 'uniform', args=(0, 1))
    #d2, pval2 = kstest(tp2, 'uniform', args=(0, 1))
    #print(pval1)
    #print(pval2)
    #if pval1 >= 0.05 and pval2 >= 0.05:
    s = p
    dp = np.append(dp, s)
    var = estimation()
    if var < minVar:
        minVar = var
        dp_lastest = dp.copy()
    dividedPoints(l, p, 0)
    dividedPoints(p, u, 0)

# #############################################################################
for i in range(2):
    print(i)
    dividedPoints(0, nPr, 0)
    var = estimation()
    if var < minVar:
        minVar = var
        dp_lastest = dp.copy()
    dp = np.array([])


f=open("SF-binning-unbining-optAlgo-poly.txt", "a+")
dp_lastest = np.sort(dp_lastest)
print(dp_lastest.tostring())
print('avg var_rbf error: %.2f' % minVar)
f.write(" ".join(map(str, dp_lastest)))
f.write('\n avg var_rbf error: %.2f' % minVar)
f.write('\n Mean - Var: \n')

# Generating cubic spline with 4 knots
transformed_x2 = dmatrix("bs(train, knots=dp_lastest,degree =3, include_intercept=False)", {"train": X},
                         return_type='dataframe')
# Fitting Generalised linear model on transformed dataset
fit2 = sm.GLM(y, transformed_x2).fit()
# Predictions on both splines
pred2 = fit2.predict(dmatrix("bs(xp, knots=dp_lastest,degree =3, include_intercept=False)", {"xp": X_test},
                             return_type='dataframe'))
# Calculating RMSE values
rms2 = sqrt(mean_squared_error(y_test, pred2))
y_pred = fit2.predict(dmatrix("bs(xp, knots=dp_lastest,degree =3, include_intercept=False)", {"xp": X},
                             return_type='dataframe'))
svr = SVR(kernel='linear', C=1e3)
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
print('c')
X_test_poly = poly.fit_transform(X_test)
print('e')
svr.fit(X_poly, y)
print('f')

diabetes_y_pred = svr.predict(X_test_poly)
y_pred = svr.predict(X_poly)
print("Mean squared error RBF: %.2f" % mean_squared_error(y_test, diabetes_y_pred))
print('Variance score: %.2f' % r2_score(y_test, diabetes_y_pred))

f.write("\n Mean squared error RBF: %.2f" % mean_squared_error(y_test, diabetes_y_pred))
f.write('\n Variance score: %.2f' % r2_score(y_test, diabetes_y_pred))
f.close()

X_s = X * 300
X_test_s = X_test * 300

plt.scatter(X_s, y, color='r', marker='o', s = 10, label='data')
plt.scatter(Xi_test_s.T, yi_test.T, c='silver', marker='^', s = 10, label='Test samples')
s = dp_lastest.size
plt.plot(X_s, y_pred, color='r', label='Polynomial regression with %s knots'%s)
plt.plot(X_s, y_pred, color='navy', marker=5, lw=lw, label='Polynomial regression')

plt.legend()
plt.xlabel('time')
plt.ylabel('#arrivalRequests')
plt.show()
plt.xlabel('time (s)');
plt.ylabel('#reqs');
plt.xlim(0, nTime)
plt.title('SF-binning-optAlgo')
plt.legend()

fn = 'SF-binning-unbinning-optAlgo-poly.png'
plt.savefig(fn, bbox_inches='tight', dpi=600)
plt.show()