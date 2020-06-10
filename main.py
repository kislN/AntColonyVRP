import numpy as np
from os.path import join
import os
from algorithms.ant_colony import VPR
from tools.data_loader import get_data
from tools.measures import get_all_results

files_A = os.listdir('./data/benchmark/A')
files_A.sort()
files_B = os.listdir('./data/benchmark/B')
files_B.sort()

cases = []
for file in files_A:
    case = get_data('A/' + file)
    if case:
        cases.append(case)
for file in files_B:
    case = get_data('B/' + file)
    if case:
        cases.append(case)

# np.random.seed(41)
# print(len(cases))
# i = 12
# # for i in range(len(cases[1:2])):
# print(cases[i]['name'])
# test = VPR(cases[i]['n_trucks'], cases[i]['dimension'], cases[i]['capacity'], cases[i]['demands'], cases[i]['adj_matrix'])
# test.compute(100, 50, 1.2, 0.1, 100, 50)
# print()

i = 7
j = 9
df = get_all_results(cases[i: j], 'results' + str(i) + '_' + str(j - 1) + '.csv')