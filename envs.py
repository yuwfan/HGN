import os
import logging
import sys
from os.path import join

# Add submodule path into import paths
# is there a better way to handle the sub module path append problem?
PROJECT_FOLDER = os.path.dirname(__file__)
sys.path.append(join(PROJECT_FOLDER, 'transformers'))

# Define the dataset folder and model folder based on environment
HOME_DATA_FOLDER = '/ssd/HGN/data'
DATASET_FOLDER = join(HOME_DATA_FOLDER, 'dataset')
MODEL_FOLDER = join(HOME_DATA_FOLDER, 'models')
KNOWLEDGE_FOLDER = join(HOME_DATA_FOLDER, 'knowledge')
OUTPUT_FOLDER = join(HOME_DATA_FOLDER, 'outputs')

os.environ['PYTORCH_PRETRAINED_BERT_CACHE']  = join(HOME_DATA_FOLDER, 'models', 'pretrained_cache')
