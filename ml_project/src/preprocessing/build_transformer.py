import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline


def build_transformer(params):
    """Get transformer"""
    num_transformer = Pipeline(steps=[('min_max', MinMaxScaler())])
    cat_transformer = Pipeline(steps=[('one_hot', OneHotEncoder(handle_unknown='ignore'))])

    transformer = ColumnTransformer(
        transformers=[
            ('numerical', num_transformer, params.num_features),
            ('categorical', cat_transformer, params.cat_features)])

    transformer = Pipeline(steps=[('preprocessor', transformer)])
    return transformer


def fit_transformer(transformer: ColumnTransformer, df_train: pd.DataFrame):
    """Fitting transformer with train data"""
    transformer.fit(df_train)
    return transformer


def get_features(transformer: ColumnTransformer, df: pd.DataFrame):
    """ Get transformed data """
    return transformer.transform(df)