"""Train params"""

from dataclasses import dataclass, field
from typing import List, Optional
RANDOM_STATE = 42
N_ESTIMATORS = 200
MEAN_SAMPLES_LEAF = 5
CRITERION = "gini"

@dataclass()
class ModelType:
    """ Structure for train model parameters """
    model_type: str = field(default='RandomForestClassifier')
    n_estimators: int = field(default=N_ESTIMATORS)
    min_samples_leaf: int = field(default=MEAN_SAMPLES_LEAF)
    random_state: int = field(default=RANDOM_STATE)


