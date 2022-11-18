import random
import numpy as np
import scipy.special
from utils import *


class NNet:
    def __init__(self, num_input, num_hidden, num_output):
        self.num_input = num_input
        self.num_hidden_1 = num_hidden
        self.num_hidden_2 = num_hidden
        self.num_output = num_output

        self.weight_input_hidden = np.random.uniform(
            -100, 100, size=(self.num_hidden_1, self.num_input))
        self.weight_hidden_output = np.random.uniform(
            -100, 100, size=(self.num_output, self.num_hidden_2))
        self.activation_function = lambda x: scipy.special.expit(x)

        self.fitness = 0

    def activate(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        x = np.dot(self.weight_input_hidden, inputs)
        x = self.activation_function(x)
        x = np.dot(self.weight_hidden_output, x)
        final_outputs = self.activation_function(x)
        return final_outputs

    def get_max_index(self, inputs_list):
        outputs = self.activate(inputs_list)
        return outputs.argmax()

    def modify_weights(self):
        NNet.modify_array(self.weight_input_hidden)
        NNet.modify_array(self.weight_hidden_output)

    def create_mixed_weights(self, net1, net2):
        self.weight_input_hidden = NNet.get_mix_from_arrays(
            net1.weight_input_hidden, net2.weight_input_hidden)
        self.weight_hidden_output = NNet.get_mix_from_arrays(
            net1.weight_hidden_output, net2.weight_hidden_output)

    def modify_array(a):
        for x in np.nditer(a, op_flags=['readwrite']):
            if random.random() < MUTATION_WEIGHT_MODIFY_CHANCE:
                x[...] = np.random.random_sample() - 0.5

    def get_mix_from_arrays(ar1, ar2):
        total_entries = ar1.size
        num_rows = ar1.shape[0]
        num_cols = ar1.shape[1]

        num_to_take = total_entries - \
            int(total_entries * MUTATION_ARRAY_MIX_PERC)
        idx = np.random.choice(np.arange(total_entries),
                               num_to_take, replace=False)

        res = np.random.rand(num_rows, num_cols)

        for row in range(0, num_rows):
            for col in range(0, num_cols):
                index = row * num_cols + col
                if index in idx:
                    res[row][col] = ar1[row][col]
                else:
                    res[row][col] = ar2[row][col]

        return res

    def save_weights(self):
        np.savetxt(f'./genetic-data/input_hidden.txt',
                   self.weight_input_hidden, fmt='%5.4g', delimiter=',')
        np.savetxt(f'./genetic-data/hidden_output.txt',
                   self.weight_hidden_output, fmt='%5.4g', delimiter=',')

    def load_weights(self, path):
        self.weight_input_hidden = np.loadtxt(
            f'{path}/input_hidden.txt', dtype=float, delimiter=',')
        self.weight_hidden_output = np.loadtxt(
            f'{path}/hidden_output.txt', dtype=float, delimiter=',')

        print("===== Loaded Weights =====")

    def create_offspring(n1, n2):
        new_nnet = NNet(NNET_INPUTS, NNET_HIDDEN, NNET_OUTPUTS)
        new_nnet.create_mixed_weights(
            n1, n2)
        return new_nnet

    def evolve_population(models):
        models.sort(key=lambda x: x.fitness, reverse=True)

        total_fitness = 0
        for model in models:
            total_fitness += model.fitness

        print("Avg. fitness:", total_fitness/GEN_SIZE)

        cut_off = int(len(models) * MUTATION_CUTOFF)
        good_models = models[0:cut_off]
        bad_models = models[cut_off:]
        num_bad_to_take = int(len(models) * MUTATION_BAD_TO_KEEP)

        for model in bad_models:
            model.modify_weights()

        new_models = []

        idx_bad_to_take = np.random.choice(
            np.arange(len(bad_models)), num_bad_to_take, replace=False)

        for index in idx_bad_to_take:
            new_models.append(bad_models[index])

        new_models.extend(good_models)

        while len(new_models) < len(models):
            idx_to_breed = np.random.choice(
                np.arange(len(good_models)), 2, replace=False)
            if idx_to_breed[0] != idx_to_breed[1]:
                new_model = NNet.create_offspring(
                    good_models[idx_to_breed[0]], good_models[idx_to_breed[1]])
                if random.random() < MUTATION_MODIFY_CHANCE_LIMIT:
                    new_model.modify_weights()
                new_models.append(new_model)

        return new_models
