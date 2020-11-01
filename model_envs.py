import envs

from transformers import (BertConfig, BertTokenizer, BertModel,
                          RobertaConfig, RobertaTokenizer, RobertaModel,
                          AlbertConfig, AlbertTokenizer, AlbertModel)
from transformers import (BertModel, XLNetModel, RobertaModel)

############################################################
# Model Related Global Varialbes
############################################################

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, AlbertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer),
}
