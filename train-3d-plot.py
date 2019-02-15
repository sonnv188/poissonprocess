from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#==============================================================================
# f = open("E:\Project\pbts\git_project\SPT\period-center-matrix-Train.txt", "r")
# X = np.empty([0,2], dtype = int)
# 
# y = np.array([])
# 
# for i in range(1920):
#     my_lines = f.readline().split(" ")
#     t = np.array([[int(my_lines[0]), int(my_lines[1])]])
#     X = np.concatenate((X, t), axis =0 )
#     y = np.concatenate((y, np.array([int(my_lines[2])])), axis = 0)
# f.close()
# print(X)
# print(y)
# print(X.shape)
# print(y.shape)
# print(type(X[0]))
# print(type(y[0]))
# 
# f = open("E:\Project\pbts\git_project\SPT\period-center-matrix-Test.txt", "r")
# X_test = np.empty([0,2], dtype = int)
# 
# y_test = np.empty([0,1], dtype = int)
# 
# for i in range(79):
#     my_lines = f.readline().split(" ")
#     t = np.array([[int(my_lines[0]), int(my_lines[1])]])
#     n = np.array([[int(my_lines[2])]])
#     X_test = np.concatenate((X_test, t), axis =0 )
#     y_test = np.concatenate((y_test, n), axis =0 )
# f.close()
# 
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# 
# ax.plot_surface(X[:, 0], X[:, 1], y, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
#==============================================================================
plt.show()