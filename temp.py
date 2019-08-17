import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# #############################################################################
# Generate sample data
f = open("E:/Project/Projects-SonNV/poissonprocess/data/SanFrancisco/reqs-period288-Train.txt", "r")
X = np.empty([0, 1], dtype=int)

y = np.array([])

for i in range(250):
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0])]])
    n = np.array([int(my_lines[1])])
    X = np.concatenate((X, t), axis=0)
    y = np.concatenate((y, np.array([int(my_lines[1])])), axis=0)
f.close()

f = open("E:/Project/Projects-SonNV/poissonprocess/data/SanFrancisco/reqs-period288-Test.txt", "r")
X_test = np.empty([0, 1], dtype=int)

y_test = np.empty([0, 1], dtype=int)

for i in range(250):
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0])]])
    n = np.array([[int(my_lines[1])]])
    X_test = np.concatenate((X_test, t), axis=0)
    y_test = np.concatenate((y_test, n), axis=0)
f.close()
# #############################################################################
# Fit regression model
svr_lin = SVR(kernel='linear', C=1e3)
# svr_poly = SVR(kernel='poly', C=1e3, gamma='auto', degree=3, epsilon=.1, coef0=1)
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
X_test_poly = poly.fit_transform(X_test)


svr_lin.fit(X_poly, y)
y_lin = svr_lin.predict(X_poly)

diabetes_y_pred_lin = svr_lin.predict(X_test_poly)
# The mean squared error
print("Mean squared error lin: %.2f"
      % mean_squared_error(y_test, diabetes_y_pred_lin))
# Explained variance score: 1 is perfect prediction
print('Variance score lin: %.2f' % r2_score(y_test, diabetes_y_pred_lin))

lw = 2
plt.scatter(X, y, color='r', marker='o', label='data')
plt.scatter(X_test.T, y_test.T, c='y', marker='^', s=40, label='Test samples')

plt.plot(X, y_lin, color='c', marker='*', lw=lw, label='Linear model')

plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()

fn = 'temp.png'
plt.xlabel('$x$', fontsize=20);
plt.ylabel('$y$', fontsize=20);

plt.savefig(fn, bbox_inches='tight', dpi=200)
plt.show()