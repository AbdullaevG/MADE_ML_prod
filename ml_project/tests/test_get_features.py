import numpy as np
from src.data.make_dataset import read_raw_data, prepare_data, split_data
from src.preprocessing.build_transformer import build_transformer, fit_transformer, get_features
from src.entity.train_pipeline_params import TrainingPipelineParams


def test_make_features(params: TrainingPipelineParams):
    df = read_raw_data(params.input_data_path)
    df, target = prepare_data(df, params.clean_features)
    df_train, df_test, train_target, test_target = split_data(df, target, params.splitting_params)
    preprocessor = build_transformer(params.feature_params)
    transformed_data = get_features(fit_transformer(preprocessor, df_train), df_train)
    assert transformed_data.shape[1] == 26
    assert np.ndarray == type(transformed_data), (
        f'Wrong data type'
        f'Result is {type(transformed_data)}'
        f'Expected type is {np.ndarray}'
    )