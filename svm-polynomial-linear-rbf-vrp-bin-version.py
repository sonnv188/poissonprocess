import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# #############################################################################
# read first bin
f = open("E:/Project/Projects-SonNV/poissonprocess/data/SanFrancisco/reqs-period288-Train.txt", "r")
X1 = np.empty([0,1], dtype = int)

y1 = np.array([])

for i in range(50):
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0])]])
    n = np.array([int(my_lines[1])])
    X1 = np.concatenate((X1, t), axis =0 )
    y1 = np.concatenate((y1, np.array([int(my_lines[1])])), axis = 0)
f.close()

f = open("E:/Project/Projects-SonNV/poissonprocess/data/SanFrancisco/reqs-period288-Test.txt", "r")
X1_test = np.empty([0,1], dtype = int)

y1_test = np.empty([0,1], dtype = int)

for i in range(50):
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0])]])
    n = np.array([[int(my_lines[1])]])
    X1_test = np.concatenate((X1_test, t), axis =0 )
    y1_test = np.concatenate((y1_test, n), axis =0 )
f.close()
###############################################################################################
#read second bin
f = open("E:/Project/Projects-SonNV/poissonprocess/data/SanFrancisco/reqs-period288-Train.txt", "r")
X2 = np.empty([0,1], dtype = int)

y2 = np.array([])

for i in range(100):
    my_lines = f.readline().split(" ")
    if i < 50:
        continue
    t = np.array([[int(my_lines[0])]])
    n = np.array([int(my_lines[1])])
    X2 = np.concatenate((X2, t), axis =0 )
    y2 = np.concatenate((y2, np.array([int(my_lines[1])])), axis = 0)
f.close()

f = open("E:/Project/Projects-SonNV/poissonprocess/data/SanFrancisco/reqs-period288-Test.txt", "r")
X2_test = np.empty([0,1], dtype = int)

y2_test = np.empty([0,1], dtype = int)

for i in range(100):
    my_lines = f.readline().split(" ")
    if i < 50:
        continue
    t = np.array([[int(my_lines[0])]])
    n = np.array([[int(my_lines[1])]])
    X2_test = np.concatenate((X2_test, t), axis =0 )
    y2_test = np.concatenate((y2_test, n), axis =0 )
f.close()
#################################################################################################
#read third bin
f = open("E:/Project/Projects-SonNV/poissonprocess/data/SanFrancisco/reqs-period288-Train.txt", "r")
X3 = np.empty([0,1], dtype = int)

y3 = np.array([])

for i in range(150):
    my_lines = f.readline().split(" ")
    if i < 100:
        continue
    t = np.array([[int(my_lines[0])]])
    n = np.array([int(my_lines[1])])
    X3 = np.concatenate((X3, t), axis =0 )
    y3 = np.concatenate((y3, np.array([int(my_lines[1])])), axis = 0)
f.close()

f = open("E:/Project/Projects-SonNV/poissonprocess/data/SanFrancisco/reqs-period288-Test.txt", "r")
X3_test = np.empty([0,1], dtype = int)

y3_test = np.empty([0,1], dtype = int)

for i in range(150):
    my_lines = f.readline().split(" ")
    if i < 100:
        continue
    t = np.array([[int(my_lines[0])]])
    n = np.array([[int(my_lines[1])]])
    X3_test = np.concatenate((X3_test, t), axis =0 )
    y3_test = np.concatenate((y3_test, n), axis =0 )
f.close()

# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=3)
###############################################################################3
#fit first bin
svr_rbf.fit(X1, y1)
svr_lin.fit(X1, y1)
svr_poly.fit(X1, y1)
y1_rbf = svr_rbf.predict(X1)
y1_lin = svr_lin.predict(X1)
y1_poly = svr_poly.predict(X1)

diabetes_y1_pred_rbf = svr_rbf.predict(X1_test)
#The mean squared error
print("Mean squared error RBF 1 : %.2f"
      % mean_squared_error(y1_test, diabetes_y1_pred_rbf))
# Explained variance score: 1 is perfect prediction
print('Variance score RBF 1: %.2f' % r2_score(y1_test, diabetes_y1_pred_rbf))

diabetes_y1_pred_lin = svr_lin.predict(X1_test)
#The mean squared error
print("Mean squared error lin 1: %.2f"
      % mean_squared_error(y1_test, diabetes_y1_pred_lin))
# Explained variance score: 1 is perfect prediction
print('Variance score lin 1: %.2f' % r2_score(y1_test, diabetes_y1_pred_lin))

diabetes_y1_pred_poly = svr_poly.predict(X1_test)
print("Mean squared error poly 1: %.2f"
      % mean_squared_error(y1_test, diabetes_y1_pred_poly))
print('Variance score poly 1: %.2f' % r2_score(y1_test, diabetes_y1_pred_poly))

################################################################33
#fit second bin
svr_rbf.fit(X2, y2)
svr_lin.fit(X2, y2)
svr_poly.fit(X2, y2)
y2_rbf = svr_rbf.predict(X2)
y2_lin = svr_lin.predict(X2)
y2_poly = svr_poly.predict(X2)

diabetes_y2_pred_rbf = svr_rbf.predict(X2_test)
print("Mean squared error RBF 2 : %.2f"
      % mean_squared_error(y2_test, diabetes_y2_pred_rbf))
# Explained variance score: 1 is perfect prediction
print('Variance score RBF 2: %.2f' % r2_score(y2_test, diabetes_y2_pred_rbf))

diabetes_y2_pred_lin = svr_lin.predict(X2_test)
#The mean squared error
print("Mean squared error lin 2: %.2f"
      % mean_squared_error(y2_test, diabetes_y2_pred_lin))
# Explained variance score: 1 is perfect prediction
print('Variance score lin 2: %.2f' % r2_score(y2_test, diabetes_y2_pred_lin))

diabetes_y2_pred_poly = svr_poly.predict(X2_test)
print("Mean squared error poly 2: %.2f"
      % mean_squared_error(y2_test, diabetes_y2_pred_poly))
print('Variance score poly 2: %.2f' % r2_score(y2_test, diabetes_y2_pred_poly))

##############################################################################
#fit third bin
svr_rbf.fit(X3, y3)
svr_lin.fit(X3, y3)
svr_poly.fit(X3, y3)
y3_rbf = svr_rbf.predict(X3)
y3_lin = svr_lin.predict(X3)
y3_poly = svr_poly.predict(X3)

diabetes_y3_pred_rbf = svr_rbf.predict(X3_test)
print("Mean squared error RBF 3 : %.2f"
      % mean_squared_error(y3_test, diabetes_y3_pred_rbf))
print('Variance score RBF 3: %.2f' % r2_score(y3_test, diabetes_y3_pred_rbf))

diabetes_y3_pred_lin = svr_lin.predict(X3_test)
#The mean squared error
print("Mean squared error lin 3: %.2f"
      % mean_squared_error(y3_test, diabetes_y3_pred_lin))
# Explained variance score: 1 is perfect prediction
print('Variance score lin 3: %.2f' % r2_score(y3_test, diabetes_y3_pred_lin))

diabetes_y3_pred_poly = svr_poly.predict(X3_test)
print("Mean squared error poly 3: %.2f"
      % mean_squared_error(y3_test, diabetes_y3_pred_poly))
print('Variance score poly 3: %.2f' % r2_score(y3_test, diabetes_y3_pred_poly))

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X1, y1, color='r', marker = 'o')
plt.scatter(X2, y2, color='r', marker = 'o') 
plt.scatter(X3, y3, color='r', marker = 'o',label='data')

plt.scatter(X1_test.T, y1_test.T, c = 'y', marker = '^', s = 40)
plt.scatter(X2_test.T, y2_test.T, c = 'y', marker = '^', s = 40)
plt.scatter(X3_test.T, y3_test.T, c = 'y', marker = '^', s = 40, label = 'Test samples')

plt.plot(X1, y1_rbf, color='navy', marker=5, lw=lw)
plt.plot(X2, y2_rbf, color='navy', marker=5, lw=lw)
plt.plot(X3, y3_rbf, color='navy', marker=5, lw=lw, label='RBF model')

plt.plot(X1, y1_lin, color='c', marker='*', lw=lw)
plt.plot(X2, y2_lin, color='c', marker='*', lw=lw)
plt.plot(X3, y3_lin, color='c', marker='*', lw=lw, label='Linear model')

plt.plot(X1, y1_poly, color='cornflowerblue', marker='x', lw=lw)
plt.plot(X2, y2_poly, color='cornflowerblue', marker='x', lw=lw)
plt.plot(X3, y3_poly, color='cornflowerblue', marker='x', lw=lw, label='Polynomial model')

plt.xlabel('time')
plt.ylabel('#Reqs')
plt.title('Support Vector Regression')
plt.legend()

fn = 'allreg_SF_288.png'    
plt.xlabel('$x$', fontsize = 20);
plt.ylabel('$y$', fontsize = 20);
    
plt.savefig(fn, bbox_inches='tight', dpi = 200)
plt.show()