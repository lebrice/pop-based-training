import random
from candidate import Candidate
from typing import List, TypeVar
from utils import T

P = TypeVar("P", bound=List[Candidate])


class TournamentSelectionOperator:
    def __init__(self, n_winners: int, fighters_per_round: int=2):
        """Creates a TournamentSelection Operator.
        
        Args:
            n_winners (int): Number of winners to return when applied.
            fighters_per_round (int, optional): Number of fighters per round.
             Defaults to 2.
        """
        self.n = n_winners
        self.t = fighters_per_round

    def __call__(self, pop: P, n: int=None, t: int=None) -> P:
        """Performs tournament selection, as described in the EPBT paper. """
        winners = type(pop)()
        n = n if n is not None else self.n
        t = t if t is not None else self.t
        while len(winners) < n:
            fighters: List[Candidate] = random.sample(pop, t)
            fighters.sort()
            best_fighter = fighters.pop()
            winners.append(best_fighter)
        return winners