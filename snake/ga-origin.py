# -*- coding: utf-8 -*-

# from time import sleep
import pygad.gann
import pygad.nn

from snake_env import GRID_WIDTH_NUM, SnakeEnv


class GANNInstance(object):
    pass


ginst = GANNInstance()


class GenAction():
    def __init__(self, idx):
        self.idx = idx
        try:
            self.layer = ginst.gann.population_networks[idx]
        except Exception:
            pass

    def __call__(self, obs):
        return pygad.nn.predict(last_layer=self.layer,
                                data_inputs=obs.reshape(1, -1))[0]


def show_gen(idx):
    global ginst

    genaction = GenAction(idx)
    env = SnakeEnv(use_simple=True)
    obs = env.reset()

    while True:
        action = genaction(obs)
        _, obs, done, _ = env(action)
        env.render()
        if done:
            break

    env.close()


def fitness(solution, idx):
    global ginst
    if idx is None:
        return 0

    genaction = GenAction(idx)
    env = SnakeEnv(need_render=False,
                   use_simple=True,
                   set_life=int(GRID_WIDTH_NUM / 3))
    obs = env.reset()
    while True:
        action = genaction(obs)
        _, obs, done, _ = env(action)
        if done:
            break

    bl = len(env.status.snake_body)
    if bl < 10:
        fscore = (env.n_moves**2) * (2**bl)
    else:
        fscore = env.n_moves**2
        fscore *= 1024
        fscore *= (bl - 9)

    if fscore > ginst.max_score:
        ginst.max_score = fscore
        ginst.max_idx = idx
        ginst.params = env.n_moves, bl
        print('find new best: ', fscore, env.n_moves, bl)

    return fscore


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

    if idx is not None:
        print('playing best: ', ginst.max_score, ginst.max_idx, ginst.params)
        show_gen(idx)


class GAPolicy(object):
    def __init__(self, n_solutions, n_input, n_output, hiddens):
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
        ginst.max_score = 0
        ginst.max_idx = 0
        self.init_generation()

    def init_generation(self):
        population_vectors = pygad.gann.population_as_vectors(
            population_networks=self.gann.population_networks)

        initial_population = population_vectors.copy()
        num_parents_mating = 150

        num_generations = 3000

        mutation_percent_genes = [0.5, 0.05]

        parent_selection_type = "sss"

        crossover_type = "single_point"

        mutation_type = "adaptive"

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

    def run(self):
        self.ga.run()


def play():
    pa = GAPolicy(n_solutions=2000, n_input=24, n_output=4, hiddens=[18, 18])
    pa.run()


if __name__ == '__main__':
    play()
