import logging
import os
import sys
import click
import json
import numpy as  np
import pandas as pd

from entity.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from data.make_dataset import download_data, read_raw_data, prepare_data, split_data
from preprocessing.build_transformer import build_transformer, fit_transformer, get_features
from models.train_model import train_model
from models.predict import predict
from models.evalute_model import evaluate_model
from models.save_load_model import save_model, load_model, save_transformer, load_transformer

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    """ train pipeline"""
    training_pipeline_params = read_training_pipeline_params(config_path)

    logger.info('Start train pipeline with %s...', training_pipeline_params.model_type.model_type)
    download_data(training_pipeline_params.downloading_params)
    logger.info("Data files are unziped!!!")

    df: pd.DataFrame = read_raw_data(training_pipeline_params.input_data_path)
    df, target = prepare_data(df, training_pipeline_params.clean_features)
    df_train, df_test, train_target, test_target = split_data(df,
                                                              target,
                                                              training_pipeline_params.splitting_params)

    logger.info("Build transformer...")
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer = fit_transformer(transformer, df_train)
    save_transformer(transformer, training_pipeline_params.save_transformer_path)

    logger.info("Get features for train data...")
    train_features = get_features(transformer, df_train)
    model = train_model(train_features, train_target, training_pipeline_params.model_type)
    save_model(model, training_pipeline_params.save_model_path)

    test_features = get_features(transformer, df_test)
    test_predict = predict(model, test_features)
    metrics = evaluate_model(test_predict, test_target)
    logger.info('Metrics: %s', metrics)

    with open(training_pipeline_params.metric_path, "w", encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file)
    logger.info("Metrics saved")


@click.command(name='train_pipeline')
@click.argument('config_path', default='configs/train_config_random_forest.yaml')
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()