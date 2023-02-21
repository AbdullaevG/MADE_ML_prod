import numpy as np
import logging
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from typing import Union
SklearnClassificationModel = Union[RandomForestClassifier, GradientBoostingClassifier]

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict(model: SklearnClassificationModel, df: pd.DataFrame) -> np.ndarray:
    """Get prediction for model"""
    logger.info('Start predict for model...')
    predict = model.predict(df)
    logger.info('Finished predict')
    return predict