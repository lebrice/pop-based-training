import multiprocessing
import pprint
import random
from dataclasses import dataclass
from itertools import combinations
from typing import *
import os

from model import Candidate, HyperParameters, Population
from operators import CrossoverOperator, MutationOperator


def evaluate(candidate: Candidate) -> Candidate:
    """
    TODO: Given the Model and HyperParameters:
    1. construct the model from the HParams
    2. Load the state-dict (if possible) from the given `Model`;
    3. Train the model on some task/dataset;
    4. Evaluate the model on some validation set;
    5. Return a new Candidate / Update - the candidate with the new fitness value; 

    """
    candidate.fitness = random.random()
    return candidate

def epbt(max_generations: int, initial_population: Population):
    population = initial_population
    pop_size = len(population)
    elite_count = pop_size // 2
    
    crossover = CrossoverOperator()
    mutate = MutationOperator()

    with multiprocessing.Pool(1) as pool:
        for i in range(max_generations):
            # Sort the population by decreasing fitness (best first).
            population.sort(reverse=True)
            # Save the best K candidates (so they don't get mutated later).
            elites = population[:elite_count]
            # Select the fittest descendants with tournament selection.
            descendants = population.select_fittest(len(population) - elite_count)
            # Apply the Mutation and Crossover operator.
            mutate(descendants, inplace=True)
            crossover(descendants, inplace=True)

            candidates = elites + descendants
            
            print(f"Generation {i}: ")
            pprint.pprint(population)
            # Evaluate the population.
            population = Population(pool.map(evaluate, candidates))
    
    return population


from model.hyperparameters import field
from priors import LogUniformPrior, UniformPrior

@dataclass
class Bob(HyperParameters):
    learning_rate: float = field(default=1e-3, min=1e-10, max=1, prior=LogUniformPrior(1e-10, 1))
    n_layers: int = field(10, prior=UniformPrior(1,20))
    optimizer: str = "ADAM"
    momentum: float = 0.9


if __name__ == "__main__":
    pop: List[Candidate] = Population([
        Candidate(model=None, hparams=Bob(learning_rate=1e-2), fitness=0.1),
        Candidate(model=None, hparams=Bob(learning_rate=1e-5), fitness=0.2),
        Candidate(model=None, hparams=Bob(learning_rate=1e-7), fitness=0.5),
        Candidate(model=None, hparams=Bob(learning_rate=1e-8), fitness=0.3),
        Candidate(model=None, hparams=Bob(learning_rate=1e-2), fitness=0.4),
    ])
    pop = epbt(5, pop)
    print(pop)
    exit()
