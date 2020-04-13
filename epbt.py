import multiprocessing
import os
import pprint
import random
import time
from dataclasses import dataclass
from itertools import combinations
from typing import *

from hyperparameters import HyperParameters, hparam
from model import Candidate, Population
from operators import CrossoverOperator, MutationOperator
from priors import LogUniformPrior, UniformPrior


def epbt(n_generations: int,
         initial_population: Population,
         evaluation_function: Callable[[Candidate], Candidate],
         n_processes: int=None,
         multiprocessing=multiprocessing) -> Generator[List[Candidate], None, None]:
    population = initial_population
    pop_size = len(population)
    elite_count = pop_size // 2
    
    crossover = CrossoverOperator()
    mutate = MutationOperator()

    with multiprocessing.Pool(n_processes) as pool:
        for i in range(n_generations):
            # Sort the population by decreasing fitness (best first).
            population.sort(reverse=True)
            # Save the best K candidates (so they don't get mutated later).
            elites = population[:elite_count]
            # Select the fittest descendants with tournament selection.
            descendants = population.select_fittest(len(population) - elite_count)
            # Apply the Mutation and Crossover operators.
            mutate(descendants, inplace=True)
            crossover(descendants, inplace=True)

            candidates = elites + descendants
            
            from pathlib import Path
            from functools import wraps
            from contextlib import redirect_stdout

            # Evaluate the population.
            population = Population()
            start_time = time.time()
            for evaluated_candidate in pool.imap_unordered(evaluation_function, candidates):
                population.append(evaluated_candidate)
                print(f"Evaluated candidate {evaluated_candidate.hparams} in {time.time() - start_time}s. Fitness = {evaluated_candidate.fitness}")
            yield population

