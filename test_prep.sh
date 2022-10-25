
################
# add --prep_test flag to process only the test set
# to all files, except preprocess.py for now!!
###############


export SINGLE_WR_WRITE="/disk1/sajad/datasets/sci/mup/single_tokenized/"
export BERT_DIR="/disk1/sajad/datasets/sci/mup/bert_data/"
export JSON_PATH="/disk1/sajad/datasets/sci/mup/bert_data/"
export HF_WR_DIR="/disk1/sajad/datasets/sci/mup/hf_format/"
export HF_FILENAME="test.parquet"

#python tokenize_ds.py --WR_DIR $SINGLE_WR_WRITE

#python data_processor/ds_text_preprocessing.py --INP_DIR $SINGLE_WR_WRITE


#mkdir -p $JSON_PATH
#mkdir -p $BERT_DIR
#python preprocess.py  -emb_path "/disk1/sajad/w2v_embeds/w2v_mup_reduced.emb" \
#                      -create_json True \
#                      -raw_path $SINGLE_WR_WRITE \
#                      -jsons_path $JSON_PATH \
#                      -save_path $BERT_DIR

#python data_processor/format_dataset_hf_with_section_scoring.py \
#                      --prep_test \
#                      --INP_DIR $BERT_DIR \
#                      --WR_DIR $HF_WR_DIR  \
#                      --filename $HF_FILENAME


################## Now inference #####################

#export PRETRAINED_CHECKPOINT="/disk0/$USER/.cache/sci-trained-models/mup-led-arxiv-6144-AllSents-PrepConc-fixed/checkpoint-29330/"
export PRETRAINED_CHECKPOINT="/disk0/$USER/.cache/sci-trained-models/mup-led-arxiv/checkpoint-66400/"

bash eval.sh $PRETRAINED_CHECKPOINT $HF_WR_DIR $BERT_DIR