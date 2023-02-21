import numpy as np
import pandas as pd
import logging
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from typing import Dict

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def evaluate_model(predict: np.ndarray, target: pd.Series) -> Dict[str, float]:
    """ Evaluate model """
    logger.info('Start calculating metrics...')
    acc = accuracy_score(target, predict)
    f1 = f1_score(target, predict, average='macro')
    roc = roc_auc_score(target, predict)
    logger.info('Finished calculating metrics.')
    return {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc_score': roc,
    }