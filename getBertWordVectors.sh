#!/bin/bash

input_file=./bert_sentences.txt
output_file=./bertWordVectors.jsonl

BERT_BASE_DIR="$PRE_TRAINED_HOME/bert/uncased_L-12_H-768_A-12"
bert_master=./bert_master

pipenv run python $bert_master/extract_features.py \
  --input_file=$input_file \
  --output_file=$output_file \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12 \
  --max_seq_length=128 \
  --batch_size=8

