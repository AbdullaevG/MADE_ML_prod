# -*- coding: utf-8 -*-
"""Subpackage for load data"""
import logging
from typing import Tuple, Optional
import sys
import gdown
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
import os

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def download_data(downloading_params):
    """Download data from gdrive as zip file """
    url = downloading_params.file_link
    output_folder = downloading_params.output_folder
    file_name = downloading_params.name
    path = output_folder + file_name

    if not os.path.exists(path):
        logger.info("Loading dataset... ")
        try:
            gdown.download(url=url, output=output_folder + file_name, fuzzy=True)
            logger.info("Dataset was downloaded")
        except ConnectionError:
            logger.info("ConnectionError: you should have link to the internet!")

    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)


def read_raw_data(path: str) -> pd.DataFrame:
    """Read data from csv file"""
    logger.info('Loading dataset from %s...', path)
    df = pd.read_csv(path)
    logger.info('Loading from %s finished', path)
    logger.info('Data shape %s', df.shape)
    return df


def prepare_data(df: pd.DataFrame, params) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Prepare data"""
    logger.info('Preparing dataset...')

    logger.info('Remove outliers...')
    outliers_field = params.outliers
    row = df[df[outliers_field] == 0].index
    df = df.drop(df.index[row])

    logger.info('Replace Nulls...')
    nulls_field = params.nulls
    median_values = df[nulls_field].median()
    row = df[df[nulls_field] == 0].index
    df.loc[row, nulls_field] = median_values

    target_name = params.target
    if target_name in df.columns:
        logger.info('Extract and drop target column...')
        target = df[target_name]
        df = df.drop([target_name], axis=1)
        return df, target
    else:
        return df, None


def split_data(df: pd.DataFrame,
               target: pd.Series,
               params) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data to train and test"""
    logger.info('Splitting data into train and test...')
    (X_train, X_test, y_train, y_test) = train_test_split(df,
                                                          target,
                                                          test_size=params.test_size,
                                                          random_state=params.random_state,
                                                          stratify=target)

    return X_train, X_test, y_train, y_test