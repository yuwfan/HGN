import numpy as np
import logging
import sys

from os.path import join

from csr_mhqa.argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from csr_mhqa.data_processing import Example, InputFeatures, DataHelper
from csr_mhqa.utils import *

from models.HGN import *
from model_envs import MODEL_CLASSES, ALL_MODELS

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#########################################################################
# Initialize arguments
##########################################################################
parser = default_train_parser()

logger.info("IN CMD MODE")
args_config_provided = parser.parse_args(sys.argv[1:])
if args_config_provided.config_file is not None:
    argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
else:
    argv = sys.argv[1:]
args = parser.parse_args(argv)
args = complete_default_train_parser(args)

logger.info('-' * 100)
logger.info('Input Argument Information')
logger.info('-' * 100)
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))


#########################################################################
# Read Data
##########################################################################
helper = DataHelper(gz=True, config=args)

# Set datasets
dev_example_dict = helper.dev_example_dict
dev_feature_dict = helper.dev_feature_dict
dev_dataloader = helper.dev_loader

#########################################################################
# Initialize Model
##########################################################################
config_class, model_encoder, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.encoder_name_or_path)

encoder_path = join(args.exp_name, 'encoder.pkl')
model_path = join(args.exp_name, 'model.pkl')
logger.info("Loading encoder from: {}".format(encoder_path))
logger.info("Loading model from: {}".format(model_path))

encoder, _ = load_encoder_model(args.encoder_name_or_path, args.model_type)
model = HierarchicalGraphNetwork(config=args)

if encoder_path is not None:
    encoder.load_state_dict(torch.load(encoder_path))
if model_path is not None:
    model.load_state_dict(torch.load(model_path))

encoder.to(args.device)
model.to(args.device)

encoder.eval()
model.eval()

#########################################################################
# Evaluation
##########################################################################
output_pred_file = join(args.exp_name, 'pred.json')
output_eval_file = join(args.exp_name, 'eval.txt')

metrics, threshold = eval_model(args, encoder, model,
                                dev_dataloader, dev_example_dict, dev_feature_dict,
                                output_pred_file, output_eval_file, args.dev_gold_file)
print("Best threshold: {}".format(threshold))
for key, val in metrics.items():
    print("{} = {}".format(key, val))
