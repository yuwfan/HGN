import envs

from transformers import (BertConfig, BertTokenizer, BertModel,
                          RobertaConfig, RobertaTokenizer, RobertaModel,
                          AlbertConfig, AlbertTokenizer, AlbertModel)
from transformers import (BertModel, XLNetModel, RobertaModel)

from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

############################################################
# Model Related Global Varialbes
############################################################

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, AlbertConfig)), ())

ALL_MODELS = sum((tuple(conf.keys()) for conf in (BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer),
}
