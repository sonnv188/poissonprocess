import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)


# generate points used to plot
x_plot = np.linspace(0, 240, 240)

# generate points and keep a subset of them
x = np.linspace(0, 240, 240)
rng = np.random.RandomState(0)
rng.shuffle(x)
f1 = open("E:\\Projects-SonNV\\SPT\\taxiDemands\\SanFrancisco\\reqs-period288-Train.txt", "r")
x = np.array([], dtype = int)

y = np.array([], dtype = int)

for i in range(288):
    my_lines = f1.readline().split(" ")
    t = int(my_lines[0])
    n = int(my_lines[1])
    x = np.append(x, np.array(t))
    y = np.append(y, np.array(n))
f1.close()

f2 = open("E:\\Projects-SonNV\\SPT\\taxiDemands\\SanFrancisco\\reqs-period288-Test.txt", "r")
x_test = np.array([], dtype = int)

y_test = np.array([], dtype = int)

for i in range(288):
    my_lines = f2.readline().split(" ")
    t = int(my_lines[0])
    n = int(my_lines[1])
    x_test = np.append(x_test, np.array(t))
    y_test = np.append(y_test, np.array(n))
f2.close()


# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

colors = ['teal', 'yellowgreen', 'gold']
lw = 2
plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw,
         label="ground truth")
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

for count, degree in enumerate([3, 4, 5]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree)
# The coefficients
    X_test = x_test[:, np.newaxis]
    diabetes_y_pred = model.predict(X_test)
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, diabetes_y_pred))
plt.legend(loc='lower left')

plt.show()