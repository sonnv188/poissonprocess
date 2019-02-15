'''
======================
Triangular 3D surfaces
======================

Plot a 3D surface with a triangular mesh.
'''
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

clusters = 20
n_radii = 8
n_angles = 36

# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

# Repeat all angles for each radius.
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

f = open('E:\\Projects-SonNV\\SPT\\taxiDemands\\SanFrancisco\\period96-center' + str(clusters) + '-day15-21-Train.txt', "r")
X = np.empty([0,2], dtype = int)

y = np.array([])

for i in range(clusters*96):
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0]), int(my_lines[1])]])
    X = np.concatenate((X, t), axis =0 )
    y = np.concatenate((y, np.array([int(my_lines[2])])), axis = 0)
f.close()
print(X)
print(y)
print(X.shape)
print(y.shape)
print(type(X[0]))
print(type(y[0]))

f = open('E:\\Projects-SonNV\\SPT\\taxiDemands\\SanFrancisco\\period96-center' + str(clusters) + '-day22-28-Test.txt', "r")
X_test = np.empty([0,2], dtype = int)

y_test = np.array([])

for i in range(clusters*96):
    my_lines = f.readline().split(" ")
    t = np.array([[int(my_lines[0]), int(my_lines[1])]])
    X_test = np.concatenate((X, t), axis =0 )
    y_test = np.concatenate((y, np.array([int(my_lines[2])])), axis = 0)
f.close()

#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#y_rbf = svr_rbf.fit(X, y).predict(X)
svr_lin = SVR(kernel='linear', C=1e3, degree = 4)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)
#y_pred = svr_rbf.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_poly))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_poly))

#fig1 = plt.figure(figsize = (20, 10))
#ax1 = fig1.gca(projection='3d')
#
#ax1.plot_trisurf(X[:, 0], X[:, 1], y, linewidth=0.2, antialiased=True)

fig2 = plt.figure(figsize = (20, 10))
ax2 = fig2.gca(projection='3d')

ax2.plot_trisurf(X[:, 0], X[:, 1], y_poly, linewidth=0.2, antialiased=True)
ax2.plot_trisurf(X_test[:, 0], X_test[:, 1], y_test, linewidth=0.2, antialiased=True)

fig3 = plt.figure(figsize = (20, 10))
ax3 = fig3.gca(projection='3d')

ax3.plot_trisurf(X[:, 0], X[:, 1], y_lin, linewidth=0.2, antialiased=True)

#fig3 = plt.figure(figsize = (20, 10))
#ax3 = fig3.gca(projection='3d')
#
#ax3.plot_trisurf(X[:, 0], X[:, 1], y_poly, linewidth=0.2, antialiased=True)

#fig4 = plt.figure(figsize = (20, 10))
#ax4 = fig4.gca(projection='3d')
#
#ax4.plot_trisurf(X_test[:, 0], X_test[:, 1], y_pred, linewidth=0.2, antialiased=True)
#
#fig6 = plt.figure(figsize = (20, 10))
#ax6 = fig6.gca(projection = '3d')
#ax6.plot_trisurf(X_test[:, 0], X_test[:, 1], y_test, linewidth = 0.2, antialiased = True)
#
#fig5 = plt.figure(figsize = (20, 10))
#ax5 = fig5.gca(projection = '3d')
#ax5.plot_trisurf(X[:, 0], X[:, 1], y, linewidth = 0.2, antialiased = True)
#ax5.plot_trisurf(X[:, 0], X[:, 1], y_rbf, linewidth = 0.2, antialiased = True)
#ax5.plot_trisurf(X_test[:, 0], X_test[:, 1], y_pred, linewidth = 0.2, antialiased = True)

plt.show()
plt.savefig('Train-Data-3d-plot.pdf')

