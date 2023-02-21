import sys
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from typing import Union

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


SklearnClassificationModel = Union[RandomForestClassifier, GradientBoostingClassifier]


def train_model(df: pd.DataFrame, target: pd.Series, train_params) -> SklearnClassificationModel:
    """Get trained model"""
    logger.info('Start loading %s model...', train_params.model_type)

    if train_params.model_type == 'RandomForestClassifier':
        model = RandomForestClassifier()

    elif train_params.model_type == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier()

    else:
        logger.exception('Model is incorrect')
        raise NotImplementedError()

    logger.info('Finished loading model')
    logger.info('Start model fitting...')
    model.fit(df, target)
    logger.info('Finished model fitting')
    return model