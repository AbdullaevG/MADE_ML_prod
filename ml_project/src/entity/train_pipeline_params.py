from typing import Optional
from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from .downloading_params import DownloadingParams
from .split_params import SplittingParams
from .model_type import ModelType
from .feature_params import FeatureParams
from .custom_transformer import CustomTransformer
from .clean_features import OutliersNulls

RANDOM_STATE = 42
TEST_SIZE = 0.3



@dataclass
class TrainingPipelineParams:
    """Structure for pipeline parameters"""
    input_data_path: str
    metric_path: str
    save_model_path: str
    save_transformer_path: str
    downloading_params: DownloadingParams
    model_type: ModelType
    clean_features: OutliersNulls
    custom_transformer: CustomTransformer
    feature_params: FeatureParams
    splitting_params: SplittingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))

