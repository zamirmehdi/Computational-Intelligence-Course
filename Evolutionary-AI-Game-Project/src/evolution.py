import copy
import csv
import random
from statistics import mean

from player import Player
import numpy as np
from config import CONFIG


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):

        # TODO
        # child: an object of class `Player`
        self.change_matrix(child.nn.b1, 'b')
        self.change_matrix(child.nn.b2, 'b')
        self.change_matrix(child.nn.w1, 'w')
        self.change_matrix(child.nn.w2, 'w')

    def change_matrix(self, matrix, mode):
        threshold = 0.1
        if mode == 'b':
            threshold = 0.2

        mutation_prob = np.random.uniform(low=-0.5, high=0.5)
        if mutation_prob >= threshold:
            noise = np.random.normal(loc=0, scale=0.3, size=(len(matrix), 1))
            # noise = np.random.randn(len(matrix), 1)
            matrix += noise

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:

            # TODO
            # num_players example: 150
            # prev_players: an array of Player objects
            fitnesses = []
            for pervert in prev_players:
                fitnesses.append(pervert.fitness ^ 4)
            chosen = random.choices(prev_players, weights=fitnesses, cum_weights=None, k=num_players)
            babies = []
            for parent in chosen:
                new_player = copy.deepcopy(parent)
                self.mutate(new_player)
                babies.append(new_player)
            # TODO (additional): a selection method other than fitness proportionate
            # TODO (additional): implementing crossover
            # new_players = prev_players
            # return new_players
            return babies

    # def generate_new_population(self, num_players, prev_players=None):
    #     # in first generation, we create random players
    #     if prev_players is None:
    #         return [Player(self.mode) for _ in range(num_players)]
    #     else:
    #         # TODO
    #         # num_players example: 150
    #         # prev_players: an array of `Player` objects
    #
    #         next_generation = []
    #         parents = prev_players
    #
    #         fitnesses = []
    #         for player in prev_players:
    #             fitnesses.append(player.fitness ^ 4)
    #         parents = random.choices(prev_players, weights=fitnesses, cum_weights=None, k=num_players)
    #
    #         for player in parents:
    #             new_player = copy.deepcopy(player)
    #             self.mutate(new_player)
    #             next_generation.append(new_player)
    #
    #         # TODO (additional): a selection method other than `fitness proportionate`
    #         # TODO (additional): implementing crossover
    #         '''
    #         for i in range(num_players):
    #             parent1 = random.choice(prev_players)
    #             parent2 = random.choice(prev_players)
    #             new_player = self.crossover(parent1, parent2)
    #             self.mutate(new_player)
    #             next_generation.append(new_player)
    #         '''
    #
    #         # next_generation.sort(key=lambda x: x.fitness, reverse=True)
    #         # next_generation = next_generation[: num_players]
    #
    #         return next_generation
    #
    #         # new_players = prev_players
    #         # return new_players

    def generate_new_population(self, num_players, prev_players=None):
        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]
        else:
            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects

            next_generation = []
            parents = prev_players

            fitnesses = []
            for player in prev_players:
                fitnesses.append(player.fitness ^ 2)
            parents = random.choices(prev_players, weights=fitnesses, cum_weights=None, k=num_players)

            for player in parents:
                new_player = copy.deepcopy(player)
                self.mutate(new_player)
                next_generation.append(new_player)

            # TODO (additional): a selection method other than `fitness proportionate`
            # TODO (additional): implementing crossover
            '''
            for i in range(num_players):
                parent1 = random.choice(prev_players)
                parent2 = random.choice(prev_players)
                new_player = self.crossover(parent1, parent2)
                self.mutate(new_player)
                next_generation.append(new_player)
            '''

            next_generation.sort(key=lambda x: x.fitness, reverse=True)
            next_generation = next_generation[: num_players]

            return next_generation

            # new_players = prev_players
            # return new_players

    def crossover(self, parent1, parent2):

        new_player = copy.deepcopy(parent1)
        threshold = 0.5

        crossover_prob = np.random.uniform(low=-0.5, high=0.5)
        if crossover_prob < threshold:
            new_player.nn.w1 = parent2.nn.w1

        crossover_prob = np.random.uniform(low=-0.5, high=0.5)
        if crossover_prob < threshold:
            new_player.nn.w2 = parent2.nn.w2

        crossover_prob = np.random.uniform(low=-0.5, high=0.5)
        if crossover_prob < threshold:
            new_player.nn.b1 = parent2.nn.b1

        crossover_prob = np.random.uniform(low=-0.5, high=0.5)
        if crossover_prob < threshold:
            new_player.nn.b2 = parent2.nn.b2

        return new_player

    def next_population_selection(self, players, num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects

        survivors = []
        fitnesses = {}

        temp_pop = players

        # TODO (additional): a selection method other than `top-k`
        '''
        q_tournament = 280
        if q_tournament > len(players):
            q_tournament = 150
        temp_pop = random.sample(population=players, k=q_tournament)
        '''

        temp_pop.sort(key=lambda x: x.fitness, reverse=True)
        survivors = temp_pop[: num_players]

        '''
        for player in temp_pop:

            if len(survivors) < num_players:
                survivors.append(player)
                fitnesses[player] = player.fitness
            elif player.fitness > min(fitnesses.values()):
                survivors.append(player)
                fitnesses[player] = player.fitness

                min_fitness = min(fitnesses.values())
                position = list(fitnesses.values()).index(min_fitness)
                kick = list(fitnesses.keys())[position]

                survivors.remove(kick)
                fitnesses.pop(kick)
        '''

        # TODO (additional): plotting
        fitness_list = []

        for player in survivors:
            fitness_list.append(player.fitness)

        if len(players) < 200:
            csv_mode = 'w'
        else:
            csv_mode = 'a'

        with open('data.csv', mode=csv_mode) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # csv_writer.writerow([max(fitness_list), min(fitness_list), sum(fitness_list) / len(fitness_list)])
            csv_writer.writerow([max(fitness_list), min(fitness_list), sum(fitness_list) / len(players)])

        return survivors
        # return players[: num_players]
