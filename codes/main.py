# Author Taher Ahmadi

import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import random
import pandas as pd
import read_data
from mpl_toolkits.mplot3d import Axes3D
import warnings
from read_data import read
from methods import naive_bayes_with_some_features

def warn(*args, **kwargs):
    pass


warnings.warn = warn


# Loading dataset
cleveland = read('./data_set/processed.cleveland.data.txt')
hungarian = read('./data_set/processed.hungarian.data.txt')
switzerland = read('./data_set/processed.switzerland.data.txt')
va = read('./data_set/processed.va.data.txt')
print('Data set Loaded!')

# Merge datasets
frames = [cleveland, hungarian, switzerland, va]
all_city_data = pd.concat(frames)

# Splitting label and features
all_city_data, all_city_label = read_data.split_label(all_city_data, 13)
all_city_label = all_city_label.reshape(len(all_city_label), 1)
all_city_data = all_city_data.reset_index(drop=True)

# Filling missing values with each columns mean for column [0, 3, 4, 7, 9] and mode for the rest
all_city_data = all_city_data.replace('?', -10)
all_city_data = all_city_data.astype(np.float)
all_city_data = all_city_data.replace(-10, np.NaN)

means = all_city_data.mean()
mean_indices = [0, 3, 4, 7, 9]
mode_indices = [1, 2, 5, 6, 8, 10, 11, 12]
for i in mean_indices:
    all_city_data[i] = all_city_data[i].fillna(means[i])
for i in mode_indices:
    all_city_data[i] = all_city_data[i].fillna(all_city_data[i].mode()[0])

# Decreasing label classes from 5 to 2(0 or 1)
for i in range(0, len(all_city_label)):
    if all_city_label[i] != 0:
        all_city_label[i] = 1

# a) Scatter data before discretizing
fig1 = plt.figure('a')
gs = gridspec.GridSpec(4, 4)
counter = 0
for i in range(0, 4):
    for j in range(0, 3):
        counter += 1
        ax_temp = fig1.add_subplot(gs[i, j])
        ax_temp.scatter(all_city_data.get(counter - 1), all_city_label,
                        s=10, color='r', alpha=.4)
        ax_temp.title.set_text(('Feature ' + str(counter)))
plt.show()

# Discretizing values of continuous columns [0, 3, 4, 7, 9]
all_city_data[0] = list(pd.cut(all_city_data[0].values, 5,
                          labels=[0, 1, 2, 3, 4]))
all_city_data[3] = list(pd.cut(all_city_data[3].values, 3,
                          labels=[0, 1, 2]))
all_city_data[4] = list(pd.cut(all_city_data[4].values, 5,
                          labels=[0, 1, 2, 3, 4]))
all_city_data[7] = list(pd.cut(all_city_data[7].values, 5,
                          labels=[0, 1, 2, 3, 4]))
all_city_data[9] = list(pd.cut(all_city_data[9].values, 3,
                          labels=[0, 1, 2]))

# a) Scatter discreteized plot of each feature
fig1 = plt.figure('a')
gs = gridspec.GridSpec(4, 4)
counter = 0
for i in range(0, 4):
    for j in range(0, 3):
        counter += 1
        ax_temp = fig1.add_subplot(gs[i, j])
        ax_temp.scatter(all_city_data.get(counter - 1), all_city_label,
                        s=10, color='r', alpha=.4)
        ax_temp.title.set_text(('Feature ' + str(counter)))
plt.show()


# # Bar plot each feature vs label after filling missing values
# fig = plt.figure()
#
# gs = gridspec.GridSpec(3, 3)
# counter = 0
# # Discrete values
# for k in range(0, 3):
#     for j in range(0, 3):
#         if counter == 8:
#             break
#         ax_temp = fig.add_subplot(gs[k, j], projection='3d')
#
#         x = all_city_data[mode_indices[counter]].values.reshape(len(all_city_data[mode_indices[counter]]), 1)
#         y = all_city_label
#         d = {}
#         for i in range(0, len(x)):
#             if (x[i][0], y[i][0]) in d.keys():
#                 d[(x[i][0], y[i][0])] += 1
#             else:
#                 d[(x[i][0], y[i][0])] = 0
#         x = []
#         y = []
#         z = []
#         for i in d.items():
#             x.append(i[0][0])
#             y.append(i[0][1])
#             z.append(i[1])
#         ax_temp.bar(x, z, zs=y, zdir='y', alpha=0.6, color='r' * 4)
#         ax_temp.set_xlabel('X')
#         ax_temp.set_ylabel('Y')
#         ax_temp.set_zlabel('Z')
#         ax_temp.title.set_text(('Feature ' + str(mode_indices[counter])))
#         counter += 1
# plt.show()
#
# # Continuous values
# fig = plt.figure()
# gs = gridspec.GridSpec(2, 3)
# counter = 0
# for k in range(0, 2):
#     for j in range(0, 3):
#         if counter == 5:
#             break
#         # print(counter)
#         ax_temp = fig.add_subplot(gs[k, j], projection='3d')
#
#         x = all_city_data[mean_indices[counter]].values.reshape(len(all_city_data[mean_indices[counter]]), 1)
#         y = all_city_label
#         d = {}
#         for i in range(0, len(x)):
#             if (x[i][0], y[i][0]) in d.keys():
#                 d[(x[i][0], y[i][0])] += 1
#             else:
#                 d[(x[i][0], y[i][0])] = 0
#         x = []
#         y = []
#         z = []
#         for i in d.items():
#             x.append(i[0][0])
#             y.append(i[0][1])
#             z.append(i[1])
#         ax_temp.bar(x, z, zs=y, zdir='y', alpha=0.6, color='r' * 4)
#         ax_temp.set_xlabel('X')
#         ax_temp.set_ylabel('Y')
#         ax_temp.set_zlabel('Z')
#         ax_temp.title.set_text(('Feature ' + str(mean_indices[counter])))
#         counter += 1
# plt.show()

# Learning naive bayes model from various subsets of data
naive_bayes_with_some_features(all_city_data, all_city_label, feature_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
naive_bayes_with_some_features(all_city_data, all_city_label, feature_list=[0, 1, 2])
naive_bayes_with_some_features(all_city_data, all_city_label, feature_list=[0, 1, 2, 4])
naive_bayes_with_some_features(all_city_data, all_city_label, feature_list=[0, 1, 2, 3, 4, 5])

# Implementing PGM model on data
# pgm_model = BayesianModel()
# pgm_model.add_nodes_from(['0', '1', '2'])
# pgm_model.add_edges_from([('0', '1'), ('1', '2')])
# zero_cpd = TabularCPD('zero', 2, [[0.2], [0.8]])
# one_cpd = TabularCPD('one', 3, [[0.5], [0.3], [0.2]])
# print(zero_cpd)
# print(one_cpd)
