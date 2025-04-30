from nbconvert.filters import prevent_list_blocks
from search import *
from random import randint
from assignment2aux import *
import numpy as np
import random

def read_tiles_from_file(filename):
    lines = [line.rstrip('\n') for line in open(filename, 'r').readlines()]
    character_to_tile = {' ': (), 'i': (0,), 'L': (0, 1), 'I': (0, 2), 'T': (0, 1, 2)}
    return tuple(tuple(character_to_tile[character] for character in line) for line in lines)

class KNetWalk(Problem):
    def __init__(self, tiles):
        if type(tiles) is str:
            self.tiles = read_tiles_from_file(tiles)
        else:
            self.tiles = tiles
        self.height = len(self.tiles)
        self.width = len(self.tiles[0])
        self.max_fitness = sum(sum(len(tile) for tile in row) for row in self.tiles)
        super().__init__(self.generate_random_state())

    def generate_random_state(self):
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [randint(0, 3) for _ in range(height) for _ in range(width)]

    def actions(self, state):
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [(i, j, k) for i in range(height) for j in range(width) for k in [0, 1, 2, 3] if state[i * width + j] != k]

    def result(self, state, action):
        pos = action[0] * len(self.tiles[0]) + action[1]
        return state[:pos] + [action[2]] + state[pos + 1:]

    def goal_test(self, state):
        return self.value(state) == self.max_fitness

    def value(self, state):
        # Task 1
        # Return an integer fitness value of a given state.
        fitness = 0 # value to store the fitness of the state

        # iterate through the tiles, and check for connections between tiles
        for i in range(self.height):
            for j in range(self.width):
                # check if the tile exists
                if self.tiles[i][j] == ():
                    continue

                # check if there is an upwards connection
                # as long as the tile is not in the first row
                if i != 0:
                    if self.connection_up(state, i, j):
                        fitness = fitness + 1

                # check if there is a downwards connection
                # as long as the tile is not in the last row
                if i != self.height - 1:
                    if self.connection_down(state, i, j):
                        fitness = fitness + 1

                # check if there is a left connection
                # as long as the tile is not in the first column
                if j != 0:
                    if self.connection_left(state, i, j):
                        fitness = fitness + 1

                # check if there is a right connection
                # as long as the tile is not in the last column
                if j != self.width - 1:
                    if self.connection_right(state, i, j):
                        fitness = fitness + 1
        return fitness
    
    def connection_up(self, state, i, j):
        # check if my tile has a connection going up
        my_tile = self.tiles[i][j]
        my_orientation = state[i * self.width + j]
        my_oriented_tile = tuple((con + my_orientation) % 4 for con in my_tile)

        # check if the tile above has a connection going down
        above_tile = self.tiles[i - 1][j]
        above_orientation = state[(i - 1) * self.width + j]
        above_oriented_tile = tuple((con + above_orientation) % 4 for con in above_tile)    

        # check if the connection is valid
        if 1 in my_oriented_tile and 3 in above_oriented_tile:
            return True
        return False
    

    def connection_down(self, state, i, j):
        # check if my tile has a connection going down
        my_tile = self.tiles[i][j]
        my_orientation = state[i * self.width + j]
        my_oriented_tile = tuple((con + my_orientation) % 4 for con in my_tile)

        # check if the tile below has a connection going up 
        below_tile = self.tiles[i + 1][j]
        below_orientation = state[(i + 1) * self.width + j]
        below_oriented_tile = tuple((con + below_orientation) % 4 for con in below_tile)

        # check if the connection is valid
        if 3 in my_oriented_tile and 1 in below_oriented_tile:
            return True
        return False

    def connection_left(self, state, i, j):    
        # check if my tile has a connection going left
        my_tile = self.tiles[i][j]
        my_orientation = state[i * self.width + j]
        my_oriented_tile = tuple((con + my_orientation) % 4 for con in my_tile)

        # check if the tile to the left has a connection going right
        left_tile = self.tiles[i][j - 1]
        left_orientation = state[i * self.width + j - 1]
        left_oriented_tile = tuple((con + left_orientation) % 4 for con in left_tile)

        # check if the connection is valid
        if 2 in my_oriented_tile and 0 in left_oriented_tile:
            return True
        return False
    

    def connection_right(self, state, i, j):    
        # check if my tile has a connection going right
        my_tile = self.tiles[i][j]
        my_orientation = state[i * self.width + j]
        my_oriented_tile = tuple((con + my_orientation) % 4 for con in my_tile)

        # check if the tile to the right has a connection going left
        right_tile = self.tiles[i][j + 1]
        right_orientation = state[i * self.width + j + 1]
        right_oriented_tile = tuple((con + right_orientation) % 4 for con in right_tile)

        # check if the connection is valid
        if 0 in my_oriented_tile and 2 in right_oriented_tile:
            return True
        return False

# Task 2
# Configure an exponential schedule for simulated annealing.
sa_schedule = exp_schedule(k=150, lam=0.4, limit=100)

# Task 3
# Configure parameters for the genetic algorithm.
pop_size = 20
num_gen = 500
mutation_prob = 0.333

def local_beam_search(problem, population):
    # Task 4
    # Implement local beam search.
    # Return a goal state if found in the population.
    # Return the fittest state in the population if the next population contains no fitter state.
    
    beam_width = len(population)
    current_population = population

    while True:
        # sort the current population by fitness
        fitness = sorted(current_population, key=problem.value, reverse=True)

        # check if it is a goal state
        if problem.goal_test(fitness[0]):
            return fitness[0]

        # generate successors
        successors = [
            problem.result(state, action)
            for state in fitness[:beam_width]
            for action in problem.actions(state)
        ]

        # sort the successors by fitness
        next_population = sorted(successors, key=problem.value, reverse=True)[:beam_width]

        # check if the next population is better than the current
        if problem.value(fitness[0]) >= problem.value(next_population[0]):
            return fitness[0]

        current_population = next_population

def stochastic_beam_search(problem, population, limit=1000):
    # Task 5
    # Implement stochastic beam search.
    # Return a goal state if found in the population.
    # Return the fittest state in the population if the generation limit is reached.
    # Replace the line below with your code.
    
    # number of iterations
    iterations = 0

    # continue to repeat until limit is reach
    while iterations <= limit:
        
        # see if goal state has been achieved
        if problem.goal_test(population[0]):
            return population[0]
        
        # sort the population by fitness
        curr_population = sorted(population, key=problem.value, reverse=True)

        # generate successors
        successors = [
            problem.result(state, action)
            for state in curr_population
            for action in problem.actions(state)
        ]

        # sort the successors by fitness
        next_population = sorted(successors, key=problem.value, reverse=True)[:len(population)]

        # check if the next population is better than the current
        if problem.value(curr_population[0]) >= problem.value(next_population[0]):
            return curr_population[0]
        
        # select a random sample of the next population
        next_population = random.sample(next_population, len(population))
        # check if the next population is better than the current
        if problem.value(curr_population[0]) >= problem.value(next_population[0]):
            return next_population[0]
            
        # set the next population to the current population
        population = next_population

        # increment the number of iterations
        iterations += 1

if __name__ == '__main__':

    network = KNetWalk('assignment2config.txt')
    visualise(network.tiles, network.initial)


    # Task 1 test code
    run = 1
    method = 'hill climbing'
    while True:
        network = KNetWalk('assignment2config.txt')
        state = hill_climbing(network)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)

    # Task 2 test code
    run = 1
    method = 'simulated annealing'
    while True:
        network = KNetWalk('assignment2config.txt')
        state = simulated_annealing(network, schedule=sa_schedule)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)

    # Task 3 test code
    run = 1
    method = 'genetic algorithm'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = genetic_algorithm([network.generate_random_state() for _ in range(pop_size)], network.value, [0, 1, 2, 3], network.max_fitness, num_gen, mutation_prob)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    
    
    # Task 4 test code
    run = 1
    method = 'local beam search'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = local_beam_search(network, [network.generate_random_state() for _ in range(100)])
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)

    # Task 5 test code
    run = 1
    method = 'stochastic beam search'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = stochastic_beam_search(network, [network.generate_random_state() for _ in range(100)])
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
