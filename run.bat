@echo off

REM DEFINE data related (please make changes according to your configurations)
REM DATA ROOT folder where you put data files
set DATA_ROOT=./data/

REM define the processes you want to run, e.g. "download,preprocess,train" or "preprocess" only
set PROCS=${1:-"download"}

REM define precached BERT MODEL path
set ROBERTA_LARGE=%DATA_ROOT%\models\pretrained\roberta-large

REM Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR_TMP:$DIR_TMP/transformers
export PYTORCH_PRETRAINED_BERT_CACHE=$DATA_ROOT/models/pretrained_cache

mkdir -p %DATA_ROOT%\models\pretrained_cache