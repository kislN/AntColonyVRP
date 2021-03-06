import numpy as np
from os.path import join
import pandas as pd
import time
from tqdm import tqdm
from algorithms.ant_colony import VPR


def get_result(case, iterations=3, **params):
    case_class = VPR(case['n_trucks'], case['dimension'], case['capacity'], case['demands'], case['adj_matrix'])
    delta_time = 0
    cost = 1e+5
    solution = None
    for _ in range(iterations):
        ts = time.time()
        case_class.compute(**params)
        te = time.time()
        delta_time += te - ts
        if case_class.final_cost < cost:
            cost = case_class.final_cost
            solution = case_class.final_sol
    delta_time = round(delta_time / iterations, 4)
    return (delta_time, cost, solution)

def get_all_results(cases):
    iters = 1
    for case in tqdm(cases):
        df = pd.DataFrame(columns=['case', 'epochs', 'n_ants', 'alpha', 'beta', 'rho', 'init_pher', 'mean_time',
                                   'found_cost', 'opt_cost'])
        best_cost = 1e+5
        best_sol = None
        for epochs in [100]:
            for n_ants in [50, 100]:
                for alpha in [1.5]:
                    for beta in [0.1, 0.3]:
                        for rho in [0.95]:
                            for init_pher in [10, 1000]:
                                result = get_result(case, iterations=iters, epochs=epochs, n_ants=n_ants, alpha=alpha,
                                                    beta=beta, rho=rho, init_pheromone=init_pher)
                                df = df.append(pd.Series([case['name'], epochs, n_ants, alpha, beta, rho, init_pher,
                                                          result[0], round(result[1], 4), case['opt']],
                                                         index=df.columns), ignore_index=True)

                                if best_cost > result[1]:
                                    best_cost = result[1]
                                    best_sol = result[2]
        write_result(case, best_sol, best_cost)
        df.to_csv('./data/comparisons/result_' + case['name'] + '.csv')


def write_result(case, best_sol, best_cost):
    path = './data'
    file_name = case['name'] + '.sol'

    sol_str = ''
    for i in range(case['n_trucks']):
        sol_str += 'Route #' + str(i + 1) + ':'
        for j in best_sol[i]:
            sol_str += ' ' + str(j)
        sol_str += '\n'
    sol_str += 'cost ' + str(best_cost)

    with open(join(path, 'solutions', file_name), 'w') as file:
        file.write(sol_str)