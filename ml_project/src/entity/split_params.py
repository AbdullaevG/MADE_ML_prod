"""Splitting data  params"""

from dataclasses import dataclass, field

RANDOM_STATE = 42
TEST_SIZE = 0.3

@dataclass
class SplittingParams:
    """ Structure contain parameters for splitting data """
    test_size: float = field(default=TEST_SIZE)
    random_state: int = field(default=RANDOM_STATE)
