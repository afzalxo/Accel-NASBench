#  Code adopted from AutoML-Freiburg/automl
#  Link to original code: https://github.com/automl/nas_benchmarks/blob/master/experiment_scripts/run_regularized_evolution.py
#  Copyright 2023 AutoML-Freiburg

import sys
import argparse
import collections
import os
import random
import csv
from ConfigSpace.read_and_write import json as cs_json
from copy import deepcopy

import ConfigSpace
import numpy as np

sys.path.append("/home/aahmadaa/NASBenchFPGA/surrogate_benchmarks")
import accelnb as anb


class Model(object):
    """A class representing a model.

    It holds two attributes: `arch` (the simulated architecture) and `accuracy`
    (the simulated accuracy / fitness). See Appendix C for an introduction to
    this toy problem.

    In the real case of neural networks, `arch` would instead hold the
    architecture of the normal and reduction cells of a neural network and
    accuracy would be instead the result of training the neural net and
    evaluating it on the validation set.

    We do not include test accuracies here as they are not used by the algorithm
    in any way. In the case of real neural networks, the test accuracy is only
    used for the purpose of reporting / plotting final results.

    In the context of evolutionary algorithms, a model is often referred to as
    an "individual".

    Attributes:  (as in the original code)
      arch: the architecture as an int representing a bit-string of length `DIM`.
          As a result, the integers are required to be less than `2**DIM`. They
          can be visualized as strings of 0s and 1s by calling `print(model)`,
          where `model` is an instance of this class.
      accuracy:  the simulated validation accuracy. This is the sum of the
          bits in the bit-string, divided by DIM to produce a value in the
          interval [0.0, 1.0]. After that, a small amount of Gaussian noise is
          added with mean 0.0 and standard deviation `NOISE_STDEV`. The resulting
          number is clipped to within [0.0, 1.0] to produce the final validation
          accuracy of the model. A given model will have a fixed validation
          accuracy but two models that have the same architecture will generally
          have different validation accuracies due to this noise. In the context
          of evolutionary algorithms, this is often known as the "fitness".
    """

    def __init__(self):
        self.arch = None
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        # print(self.arch)
        #return '{0:b}'.format(self.arch)
        return str(self.arch)


def eval_arch(config):
    acc, _ = b.query(config)
    # returns negative error (similar to maximizing accuracy)
    return acc


def random_architecture():
    config = cs.sample_configuration()
    return config


def mutate_arch(parent_arch):
    # pick random parameter
    dim = np.random.randint(len(cs.get_hyperparameters()))
    hyper = cs.get_hyperparameters()[dim]

    if type(hyper) == ConfigSpace.OrdinalHyperparameter:
        choices = list(hyper.sequence)
    else:
        choices = list(hyper.choices)
    # drop current values from potential choices
    choices.remove(parent_arch[hyper.name])

    # flip parameter
    idx = np.random.randint(len(choices))

    child_arch = deepcopy(parent_arch)
    child_arch[hyper.name] = choices[idx]
    return child_arch


def regularized_evolution(cycles, population_size, sample_size):
    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture()
        model.accuracy = eval_arch(model.arch)
        population.append(model)
        history.append(model)

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    best_acc_so_far = [max(history, key=lambda i: i.accuracy).accuracy[0]]
    acc_traject = [(len(history)/cycles * 100, best_acc_so_far[0])]
    cur_cycle = len(history)
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.accuracy)
        if best_acc_so_far[-1] < parent.accuracy[0]:
            best_acc_so_far.append(parent.accuracy[0])
        acc_traject.append(((cur_cycle+1)/cycles * 100, best_acc_so_far[-1]))

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch)
        child.accuracy = eval_arch(child.arch)
        population.append(child)
        history.append(child)

        # Remove the oldest model.
        population.popleft()
        cur_cycle += 1

    return history, parent, acc_traject


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--n_iters', default=600, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="./", type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--pop_size', default=20, type=int, nargs='?', help='population size')
parser.add_argument('--sample_size', default=10, type=int, nargs='?', help='sample_size')
parser.add_argument('--seed', default=1, type=int)


args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

b = anb.ANBEnsemble("xgb", seed=None).load_ensemble()

output_path = os.path.join(args.output_path, "logs")
os.makedirs(os.path.join(output_path), exist_ok=True)
ss_configspace_path = "../configspace/configspace.json"
with open(ss_configspace_path, "r") as fh:
    json_string = fh.read()

cs = cs_json.read(json_string)
history, final, acc_trajectory = regularized_evolution(
    cycles=args.n_iters, population_size=args.pop_size, sample_size=args.sample_size)

print(acc_trajectory)
with open(os.path.join(output_path, f"acc_trajectory_rea_{args.seed}.csv"), 'w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    for elem in acc_trajectory:
        csv_writer.writerow(elem)
