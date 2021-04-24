# -*- coding: utf-8 -*-
import os
import sys
from time import sleep
import pygad.gann
import pygad.nn

from snake_env import SnakeEnv


class GANNInstance(object):
    pass


ginst = GANNInstance()


class GenAction():
    def __init__(self, idx):
        self.idx = idx
        self.layer = ginst.gann.population_networks[idx]

    def __call__(self, obs):
        return pygad.nn.predict(last_layer=self.layer,
                                data_inputs=obs.reshape(1, -1))[0]


def show_gen(idx):
    global ginst

    genaction = GenAction(idx)
    env = SnakeEnv(use_simple=True)
    obs = env.reset()

    c = 0
    while True:
        c += 1
        action = genaction(obs)
        _, obs, done, _ = env(action)
        extra = '代数: {}'.format(ginst.n_gen)
        if c % 3 == 0:

            if ginst.n_gen:
                env.render(extra)
            else:
                env.render()

        # sleep(1 / ginst.n_gen)
        print(done, env.status.direction, env.life)
        if done:
            break

    sleep(1.5)


def fitness(solution, idx):
    global ginst

    genaction = GenAction(idx)

    env = ginst.env
    obs = env.reset()
    while True:
        action = genaction(obs)
        _, obs, done, _ = env(action)
        if done:
            break

    if env.status.score + env.n_moves / 10. > 500:
        show_gen(idx)

    return env.status.score + env.n_moves / 10.


def callback_generation(ga):
    population_matrices = pygad.gann.population_as_matrices(
        population_networks=ginst.gann.population_networks,
        population_vectors=ga.population)
    ginst.gann.update_population_trained_weights(
        population_trained_weights=population_matrices)

    print("Generation = {generation}".format(
        generation=ga.generations_completed))
    _, fitness_score, idx = ga.best_solution()
    print("Accuracy   = {fitness}".format(fitness=fitness_score))
    ga.save('ga-generations/gen-{}'.format(ga.generations_completed))
    ga.save('ga-generations/gen-latest')

    show_gen(idx)


class GAPolicy(object):
    def __init__(self,
                 n_solutions=2000,
                 n_input=24,
                 n_output=4,
                 hiddens=[18, 18],
                 n_gen=0):
        self.n_solutions = n_solutions
        self.n_input = n_input
        self.n_output = n_output
        self.hiddens = hiddens

        self.gann = pygad.gann.GANN(num_solutions=n_solutions,
                                    num_neurons_input=n_input,
                                    num_neurons_output=n_output,
                                    num_neurons_hidden_layers=hiddens,
                                    hidden_activations=["relu", "relu"],
                                    output_activation="softmax")
        global ginst
        ginst.gann = self.gann
        ginst.gene = 0
        ginst.n_gen = n_gen
        if not hasattr(ginst, 'env') or ginst.env is None:
            ginst.env = SnakeEnv(need_render=True, use_simple=True)
            ginst.env.reset()
            ginst.env.render()
            input()
        ginst.env.reset()
        ginst.env.render()

        self.init_generation(n_gen=n_gen)
        self.n_gen = n_gen

    def init_generation(self, n_gen):
        population_vectors = pygad.gann.population_as_vectors(
            population_networks=self.gann.population_networks)

        initial_population = population_vectors.copy()
        num_parents_mating = 4

        num_generations = 3000

        mutation_percent_genes = 5

        parent_selection_type = "sss"

        crossover_type = "single_point"

        mutation_type = "random"

        keep_parents = 1

        init_range_low = -2
        init_range_high = 5

        self.ga = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=fitness,
                           mutation_percent_genes=mutation_percent_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           keep_parents=keep_parents,
                           on_generation=callback_generation,
                           save_best_solutions=True)
        fname = 'ga-generations/gen-{n_gen}'.format(n_gen=n_gen)
        if os.path.exists(fname + '.pkl'):
            self.ga = pygad.load(fname)
            pm = pygad.gann.population_as_matrices(
                population_networks=self.gann.population_networks,
                population_vectors=self.ga.population)
            self.gann.update_population_trained_weights(
                population_trained_weights=pm)
            print('best score = {}, idx = {}'.format(
                self.ga.last_generation_fitness.max(),
                self.ga.last_generation_fitness.argmax()))

    def run(self):
        self.ga.run()

    def play_best(self):
        show_gen(self.ga.last_generation_fitness.argmax())


def play():
    if len(sys.argv) > 1:
        gens = [int(sys.argv[1])]
    else:
        gens = list(range(1, 10))
        gens.extend(range(10, 100, 10))
        gens.extend(range(100, 401, 100))
    for n_gen in gens:
        pa = GAPolicy(n_solutions=2000,
                      n_input=24,
                      n_output=4,
                      hiddens=[18, 18],
                      n_gen=n_gen)
        pa.play_best()


if __name__ == '__main__':
    play()
