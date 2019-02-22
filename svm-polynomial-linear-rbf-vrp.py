import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# #############################################################################
# Generate sample data
f = open("poissonprocess/data/SanFrancisco/reqs-period288-Train.txt", "r")
X = np.empty([0,1], dtype = int)

y = np.array([])

for i in range(250):
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0])]])
    n = np.array([int(my_lines[1])])
    X = np.concatenate((X, t), axis =0 )
    y = np.concatenate((y, np.array([int(my_lines[1])])), axis = 0)
f.close()

f = open("poissonprocess/data/SanFrancisco/reqs-period288-Test.txt", "r")
X_test = np.empty([0,1], dtype = int)

y_test = np.empty([0,1], dtype = int)

for i in range(250):
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0])]])
    n = np.array([[int(my_lines[1])]])
    X_test = np.concatenate((X_test, t), axis =0 )
    y_test = np.concatenate((y_test, n), axis =0 )
f.close()
# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=4)

svr_rbf.fit(X, y)
svr_lin.fit(X, y)
svr_poly.fit(X, y)
y_rbf = svr_rbf.predict(X)
y_lin = svr_lin.predict(X)
y_poly = svr_poly.predict(X)

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


diabetes_y_pred_poly = svr_poly.predict(X_test)
## The mean squared error
print("Mean squared error poly: %.2f"
      % mean_squared_error(y_test, diabetes_y_pred_poly))
# Explained variance score: 1 is perfect prediction
print('Variance score poly: %.2f' % r2_score(y_test, diabetes_y_pred_poly))

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X, y, color='r', marker = 'o',label='data') 
plt.scatter(X_test.T, y_test.T, c = 'y', marker = '^', s = 40, label = 'Test samples')

plt.plot(X, y_rbf, color='navy', marker=5, lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', marker='*', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', marker='x', lw=lw, label='Polynomial model')

plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()

fn = 'allreg_SF_288.png'    
plt.xlabel('$x$', fontsize = 20);
plt.ylabel('$y$', fontsize = 20);
    
plt.savefig(fn, bbox_inches='tight', dpi = 200)
plt.show()