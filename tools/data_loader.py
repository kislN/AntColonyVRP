import numpy as np
from os.path import join


def get_adj_matrix(coord):
    adj_matrix = np.zeros((len(coord), len(coord)))
    for i in range(len(coord)):
        for j in range(i + 1, len(coord)):
            adj_matrix[i, j] = adj_matrix[j, i] = np.sqrt((coord[i][0] - coord[j][0])**2 +
                                                          (coord[i][1] - coord[j][1])**2)
            if adj_matrix[i, j] == 0:
                adj_matrix[i, j] = adj_matrix[j, i] = 0.01
    return adj_matrix

def get_data(file_name):
    case_dict = {}
    path = './data/benchmark'

    with open(join(path, file_name), 'r') as file:
        data = file.read()
        data = data.split('\n')
        case_dict['name'] = data[0].split(' ')[2]
        info = data[1].split(' ')
        case_dict['n_trucks'] = int(info[8][:-1])
        case_dict['opt'] = float(info[11][:-1])
        case_dict['dimension'] = int(data[3].split(' ')[2])
        case_dict['capacity'] = int(data[5].split(' ')[2])

        coord = []
        for i in range(7, 7 + case_dict['dimension']):
            point = data[i].split(' ')
            coord.append((int(point[2]), int(point[3])))
        case_dict['coordinates'] = np.array(coord)

        dem = []
        for i in range(8 + case_dict['dimension'], 8 + 2 * case_dict['dimension']):
            point = data[i].split(' ')
            dem.append(int(point[1]))
        case_dict['demands'] = np.array(dem)
        case_dict['sum_demand'] = np.array(dem).sum()

        case_dict['adj_matrix'] = get_adj_matrix(coord)

        # if case_dict['n_trucks'] * case_dict['capacity'] < case_dict['sum_demand']:
        #     print(case_dict['name'], '- ERROR_CAPACITY!')
        #     return None
        #
        # if len(set(coord)) < case_dict['dimension']:
        #     print(case_dict['name'], '- ERROR_COORDINATES!')
        #     for i in range(len(coord)):
        #         for j in range(i + 1, len(coord)):
        #             if coord[i] == coord[j]:
        #                 print(i + 1, '-', j + 1)
        #     return None

        print(case_dict['name'], ' is done!')

    return case_dict

