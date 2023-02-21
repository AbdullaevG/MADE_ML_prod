""" Predicting pipeline for ml model """
import logging
import sys

import click
import pandas as pd


from entity.predict_pipeline_params import PredictPipelineParams, read_predict_pipeline_params
from data.make_dataset import read_raw_data, prepare_data
from preprocessing.build_transformer import  get_features
from models.predict import predict
from models.save_load_model import load_model, load_transformer


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(config_path: str):
    """ Predict pipeline """
    predict_pipeline_params = read_predict_pipeline_params(config_path)
    # print(predict_pipeline_params)
    logger.info('Start predict pipeline...')
    df: pd.DataFrame = read_raw_data(predict_pipeline_params.input_data_path)
    df, target = prepare_data(df, predict_pipeline_params.clean_features)
    transformer = load_transformer(predict_pipeline_params.transformer_path)
    features: pd.DataFrame = get_features(transformer, df)
    model = load_model(predict_pipeline_params.model_path)
    logger.info('Making predict...')
    y_predict = predict(model, features)
    logger.info('Writing to file...')
    pd.Series(y_predict, index=df.index, name='Predict').to_csv(predict_pipeline_params.predict_path)


@click.command(name='predict_pipeline')
@click.argument('config_path')
def predict_pipeline_command(config_path: str):
    """ Make start for terminal """
    predict_pipeline(config_path)


if __name__ == '__main__':
    predict_pipeline_command()