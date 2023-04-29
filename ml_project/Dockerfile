FROM python:3.6

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip -r /requirements.txt

COPY configs/ /configs
COPY ml_project_total/ /
RUN mkdir -p /models

RUN python src/train_pipeline.py configs/train_config_random_forest.yaml
