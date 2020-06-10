import numpy as np
np.random.seed(42)
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time


class VPR:

    def __init__(self, n_trucks, dimension, capacity, demands, adj_matrix):
        self.n_trucks = n_trucks
        self.dimension = dimension
        self.capacity = capacity
        self.demands = demands
        self.adj_matrix = adj_matrix
        self.adj_matrix_sum = adj_matrix.sum()

        self.init_pheramone_value = None
        self.pheramone_map = None
        self.alpha = None
        self.beta = None
        self.epoches = None
        self.tabu = None
        self.capacity_left = None
        self.p = None
        self.best_ant_sol = None
        self.raw_prob_matrix = None
        self.k = None
        self.potential_vertexes = None

    def get_probality(self, raw_prob_list):
        prob_list = raw_prob_list/raw_prob_list.sum()
        return prob_list

    def get_next_vertex(self, pos):
        potential = deepcopy(self.tabu)
        potential_sum = self.tabu_sum
        while potential_sum < self.dimension:
            raw_prob_list = deepcopy(self.raw_prob_matrix[pos]) * potential
            next_vertex = np.random.choice(np.arange(0, self.dimension), p=self.get_probality(raw_prob_list))
            if self.demands[next_vertex] <= self.capacity_left:
                return next_vertex
            potential[next_vertex] = 0
            potential_sum += 1
        return 0

    def local_update(self, i, j):
        self.pheramone_map[i, j] += self.p * self.init_pheramone_value / self.adj_matrix[i, j]
        self.pheramone_map[j, i] = self.pheramone_map[i, j]
        self.raw_prob_matrix[i, j] = self.raw_prob_matrix[j, i] = (self.pheramone_map[i, j]**self.alpha) * \
                                                                  (self.adj_matrix[i, j]**self.beta)

    def global_update(self, best_solution, best_cost):
        for one_path in best_solution:
            for i in range(len(one_path)-1):
                self.pheramone_map[one_path[i], one_path[i + 1]] += self.p * self.capacity / best_cost
                self.pheramone_map[one_path[i + 1], one_path[i]] = self.pheramone_map[one_path[i], one_path[i + 1]]
                self.raw_prob_matrix[one_path[i], one_path[i + 1]] = \
                    self.raw_prob_matrix[one_path[i + 1], one_path[i]] = \
                    (self.pheramone_map[one_path[i], one_path[i + 1]] ** self.alpha) * \
                    (self.adj_matrix[one_path[i], one_path[i + 1]] ** self.beta)

    def get_cost(self, solution):
        current_cost = 0
        for i in range(len(solution) - 1):
            current_cost += self.adj_matrix[solution[i], solution[i + 1]]
        return current_cost

    def compute(self, epoches=5, k=10, alpha=0.9, beta=0.1, p=100, init_pheramone=1):
        self.epoches = epoches
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.init_pheramone_value = init_pheramone

        self.pheramone_map = np.full(shape=(self.dimension, self.dimension), fill_value=self.init_pheramone_value)
        np.fill_diagonal(self.pheramone_map, 0)
        self.raw_prob_matrix = (self.pheramone_map ** self.alpha) * (self.adj_matrix ** self.beta)

        self.tabu_sum = None

        show_epoch = []
        iam_the_best_of_the_best_cost = self.adj_matrix_sum
        iam_the_best_of_the_best_sol = None
        for epoch in range(self.epoches):
            time_s = time()
            best_solution = None
            best_cost = self.adj_matrix_sum
            for ant in range(self.k):
                current_state = 0
                solutions = []
                one_path_solution = [0]
                self.capacity_left = self.capacity
                self.tabu = np.ones(self.dimension)
                self.tabu[0] = 0
                self.tabu_sum = 1
                while self.tabu_sum < self.dimension:
                    next_state = self.get_next_vertex(current_state)    # TODO: optimise
                    if next_state == 0:
                        one_path_solution.append(0)
                        solutions.append(one_path_solution)
                        one_path_solution = [0]
                        current_state = 0
                        self.capacity_left = self.capacity
                        continue
                    one_path_solution.append(next_state)
                    self.capacity_left -= self.demands[next_state]
                    self.local_update(current_state, next_state)
                    current_state = next_state
                    self.tabu[current_state] = 0
                    self.tabu_sum += 1

                one_path_solution.append(0)
                solutions.append(one_path_solution)

                cost = sum([self.get_cost(sol) for sol in solutions])  # TODO: optimise

                assert all(np.unique(np.hstack(solutions)) == np.arange(self.dimension))
                # assert len(solutions) <= self.n_trucks

                if len(solutions) > self.n_trucks:
                    pass
                    # print('fuck')
                else:
                    if cost < best_cost:
                        best_cost = cost
                        best_solution = solutions

            if best_solution is None:
                print('global fuck')
            else:
                self.global_update(best_solution, best_cost)

                show_epoch.append(best_cost)
                if iam_the_best_of_the_best_cost > best_cost:
                    iam_the_best_of_the_best_cost = best_cost
                    iam_the_best_of_the_best_sol = best_solution

                print(f'Epoch: {epoch} | time: {round(time() - time_s, 4)}| best cost: {best_cost}')

        print(self.alpha, self.beta, self.p, self.k)
        plt.plot(np.arange(len(show_epoch)), np.array(show_epoch))
        plt.show()
        print(iam_the_best_of_the_best_sol)
        print(f'Cost: {iam_the_best_of_the_best_cost}')


