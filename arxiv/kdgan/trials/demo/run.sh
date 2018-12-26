DATADIR=/Users/xiaojiew1/Projects/data/yfcc100m/fbinput
DATASET=yfcc10k

TRAIN_FBINPUT=$DATADIR/${DATASET}.train
VALID_FBINPUT=$DATADIR/${DATASET}.valid
LABEL_FILE=$DATADIR/${DATASET}.label
VOCAB_FILE=$TRAIN_FBINPUT.vocab

# python main.py \
#     --facebook_infile=${TRAIN_FBINPUT} \
#     --label_file=$LABEL_FILE \
#     --vocab_file=$VOCAB_FILE \
#     --ngrams=2,3,4

# python main.py \
#     --facebook_infile=${VALID_FBINPUT} \
#     --label_file=$LABEL_FILE \
#     --vocab_file=$VOCAB_FILE \
#     --ngrams=2,3,4

# exit

TRAIN_TFRECORD=$TRAIN_FBINPUT.tfrecord
VALID_TFRECORD=$VALID_FBINPUT.tfrecord
LOGS_DIR=$DATADIR/logs

# VOCAB_SIZE=`cat $VOCAB_FILE | wc -l | sed -e "s/[ \t]//g"`
# echo $VOCAB_SIZE

python main.py \
    --train_tfrecord=$TRAIN_TFRECORD \
    --valid_tfrecord=$VALID_TFRECORD \
    --label_file=$LABEL_FILE \
    --vocab_file=$VOCAB_FILE \
    --logs_dir=$LOGS_DIR \
    --num_epochs=1000