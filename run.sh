#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files
DATA_ROOT=./data/


PROCS=${1:-"download"} # define the processes you want to run, e.g. "download,preprocess,train" or "preprocess" only

# define precached BERT MODEL path
ROBERTA_LARGE=$DATA_ROOT/models/pretrained/roberta-large

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR_TMP:$DIR_TMP/transformers
export PYTORCH_PRETRAINED_BERT_CACHE=$DATA_ROOT/models/pretrained_cache

mkdir -p $DATA_ROOT/models/pretrained_cache

# 0. Build Database from Wikipedia
download() {
    [[ -d $DATA_ROOT ]] || mkdir -p $DATA_ROOT/dataset/data_raw; mkdir -p $DATA_ROOT/knowledge

    wget -P $DATA_ROOT/dataset/data_raw/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
    wget -P $DATA_ROOT/dataset/data_raw/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    wget -P $DATA_ROOT/dataset/data_raw http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
    wget -P $DATA_ROOT/dataset/data_raw/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json
    if [[ ! -f $DATA_ROOT/knowledge/enwiki_ner.db ]]; then
        wget -P $DATA_ROOT/knowledge/ https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
        tar -xjvf $DATA_ROOT/knowledge/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 -C $DATA_ROOT/knowledge
        # Required: DrQA and Spacy
        python scripts/0_build_db.py $DATA_ROOT/knowledge/enwiki-20171001-pages-meta-current-withlinks-abstracts $DATA_ROOT/knowledge/enwiki_ner.db
    fi
}

preprocess() {
    INPUTS=("hotpot_dev_distractor_v1.json;dev_distractor" "hotpot_train_v1.1.json;train")
    for input in ${INPUTS[*]}; do
        INPUT_FILE=$(echo $input | cut -d ";" -f 1)
        DATA_TYPE=$(echo $input | cut -d ";" -f 2)

        echo "Processing input_file: ${INPUT_FILE}"

        INPUT_FILE=$DATA_ROOT/dataset/data_raw/$INPUT_FILE
        OUTPUT_PROCESSED=$DATA_ROOT/dataset/data_processed/$DATA_TYPE
        OUTPUT_FEAT=$DATA_ROOT/dataset/data_feat/$DATA_TYPE

        [[ -d $OUTPUT_PROCESSED ]] || mkdir -p $OUTPUT_PROCESSED
        [[ -d $OUTPUT_FEAT ]] || mkdir -p $OUTPUT_FEAT

        echo "1. Extract Wiki Link & NER from DB"
        # Input: INPUT_FILE, enwiki_ner.db
        # Output: doc_link_ner.json
        python scripts/1_extract_db.py $INPUT_FILE $DATA_ROOT/knowledge/enwiki_ner.db $OUTPUT_PROCESSED/doc_link_ner.json

        echo "2. Extract NER for Question and Context"
        # Input: doc_link_ner.json
        # Output: ner.json
        python scripts/2_extract_ner.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json

        echo "3. Paragraph ranking"
        # Output: para_ranking.json
        python scripts/3_prepare_para_sel.py $INPUT_FILE $OUTPUT_PROCESSED/hotpot_ss_$DATA_TYPE.csv

        # switch to RoBERTa for final leaderboard
        python scripts/3_paragraph_ranking.py --data_dir $OUTPUT_PROCESSED --eval_ckpt $DATA_ROOT/models/finetuned/PS/pytorch_model.bin --raw_data $INPUT_FILE --input_data $OUTPUT_PROCESSED/hotpot_ss_$DATA_TYPE.csv --model_name_or_path roberta-large --model_type roberta --max_seq_length 256 --per_gpu_eval_batch_size 128 --fp16

        echo "4. MultiHop Paragraph Selection"
        # Input: $INPUT_FILE, doc_link_ner.json,  ner.json, para_ranking.json
        # Output: multihop_para.json
        python scripts/4_multihop_ps.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json $OUTPUT_PROCESSED/para_ranking.json $OUTPUT_PROCESSED/multihop_para.json

        echo "5. Dump features"
        python scripts/5_dump_features.py --para_path $OUTPUT_PROCESSED/multihop_para.json --raw_data $INPUT_FILE --model_name_or_path roberta-large --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type roberta --tokenizer_name roberta-large --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json
        python scripts/5_dump_features.py --para_path $OUTPUT_PROCESSED/multihop_para.json --raw_data $INPUT_FILE --model_name_or_path albert-xxlarge-v2 --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type albert --tokenizer_name albert-xxlarge-v2 --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json

        echo "6. Test dumped features"
        #python scripts/6_test_features.py --raw_data $INPUT_FILE --input_dir $OUTPUT_FEAT --output_dir $OUTPUT_FEAT --model_type roberta --model_name_or_path roberta-large
    done

}

for proc in "download" "preprocess"
do
    if [[ ${PROCS:-"download"} =~ $proc ]]; then
        $proc
    fi
done
