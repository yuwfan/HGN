from os import environ as os_environ
from sys.path import append as sys_path_append
from os.path import join as os_path_join, dirname as os_path_dirname

# Add submodule path into import paths
# is there a better way to handle the sub module path append problem?
PROJECT_FOLDER = os_path_dirname(__file__)
sys_path_append(os_path_join(PROJECT_FOLDER, 'transformers'))

# Define the dataset folder and model folder based on environment
HOME_DATA_FOLDER = '/home/zhanwen/hgn/data'
DATASET_FOLDER = os_path_join(HOME_DATA_FOLDER, 'dataset')
MODEL_FOLDER = os_path_join(HOME_DATA_FOLDER, 'models')
KNOWLEDGE_FOLDER = os_path_join(HOME_DATA_FOLDER, 'knowledge')
OUTPUT_FOLDER = os_path_join(HOME_DATA_FOLDER, 'outputs')

os_environ['PYTORCH_PRETRAINED_BERT_CACHE']  = os_path_join(HOME_DATA_FOLDER, 'models', 'pretrained_cache')
