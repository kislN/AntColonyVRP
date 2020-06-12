import numpy as np
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
        self.final_cost = self.adj_matrix_sum
        self.final_sol = None

        self.epochs = None
        self.n_ants = None
        self.alpha = None
        self.beta = None
        self.rho = None
        self.init_pheromone_value = None
        self.pheromone_map = None
        self.raw_prob_matrix = None
        self.tabu = None
        self.tabu_sum = None
        self.capacity_left = None

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
        self.pheromone_map[i, j] += self.rho * self.init_pheromone_value / self.adj_matrix[i, j]
        # self.pheromone_map[i, j] = (1 - self.p) * self.pheromone_map[i, j] + self.p * self.init_pheromone_value
        self.pheromone_map[j, i] = self.pheromone_map[i, j]
        self.raw_prob_matrix[i, j] = self.raw_prob_matrix[j, i] = (self.pheromone_map[i, j] ** self.alpha) * \
                                                                  ((1 / self.adj_matrix[i, j]) ** self.beta)


    def global_update(self, best_solution, best_cost):
        for one_path in best_solution:
            for i in range(len(one_path)-1):
                self.pheromone_map[one_path[i], one_path[i + 1]] += self.rho * self.capacity / best_cost
                # self.pheromone_map[one_path[i], one_path[i + 1]] = (1 - self.p) * self.pheromone_map[one_path[i], one_path[i + 1]] \
                #                                                    + self.p * self.capacity / best_cost
                self.pheromone_map[one_path[i + 1], one_path[i]] = self.pheromone_map[one_path[i], one_path[i + 1]]
                self.raw_prob_matrix[one_path[i], one_path[i + 1]] = \
                    self.raw_prob_matrix[one_path[i + 1], one_path[i]] = \
                    (self.pheromone_map[one_path[i], one_path[i + 1]] ** self.alpha) * \
                    ((1 / self.adj_matrix[one_path[i], one_path[i + 1]]) ** self.beta)

    def get_cost(self, solution):
        current_cost = 0
        for i in range(len(solution) - 1):
            current_cost += self.adj_matrix[solution[i], solution[i + 1]]
        return current_cost

    def plot_function(self):
        box = {'facecolor': 'white',
               'edgecolor': 'black',
               'boxstyle': 'round'}
        plt.text(self.epochs - 20, self.show_epoch[5],
                 f'cost={round(self.final_cost, 2)}, \n'
                 f'epochs={self.epochs}, \n'
                 f'n_ants={self.n_ants}, \n'
                 f'alpha={self.alpha}, \n'
                 f'beta={self.beta}, \n'
                 f'p={self.rho}, \n'
                 f'init_ph={self.init_pheromone_value}',
                 bbox=box, color='black', fontsize=10)
        plt.plot(np.arange(self.epochs), self.show_cost, 'r')
        plt.plot(np.arange(self.epochs), self.show_epoch, 'k')
        plt.grid()
        plt.title(f'A-n{self.dimension}-k{self.n_trucks}')
        plt.xlabel('epoch')
        plt.ylabel('cost')
        plt.show()

    def compute(self, epochs=100, n_ants=50, alpha=1.5, beta=0.3, rho=0.95, init_pheromone=1000):
        self.epochs = epochs
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.init_pheromone_value = init_pheromone
        self.pheromone_map = np.full(shape=(self.dimension, self.dimension), fill_value=self.init_pheromone_value)
        np.fill_diagonal(self.pheromone_map, 0)

        np.fill_diagonal(self.adj_matrix, 0.1)
        self.raw_prob_matrix = (self.pheromone_map ** self.alpha) * ((1 / self.adj_matrix) ** self.beta)

        np.fill_diagonal(self.adj_matrix, 0)

        self.show_epoch = []
        self.show_cost = []
        for epoch in range(self.epochs):
            time_s = time()
            best_solution = None
            best_cost = self.adj_matrix_sum
            for ant in range(self.n_ants):
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
                self.show_epoch.append(best_cost)
                if self.final_cost > best_cost:
                    self.final_cost = best_cost
                    self.final_sol = best_solution
                    self.show_cost.append(self.final_cost)
                else:
                    self.show_cost.append(self.show_cost[-1])
                # print(f'Epoch: {epoch} | time: {round(time() - time_s, 4)}| best cost: {best_cost}')



        # if self.final_sol is None:
        #     print('WORLD WIDE FUCK')
        # print(self.final_sol)
        # print(f'Cost: {self.final_cost}')


