kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
dataset_dir=$HOME/Projects/data/mnist
num_epoch=200

train() {
  train_size=$1
  batch_size=$2

  gen_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_gen
  echo ${gen_model_ckpt}
  python pretrain_gen.py \
    --gen_model_ckpt=${gen_model_ckpt} \
    --dataset_dir=${dataset_dir} \
    --gen_model_name=mlp \
    --optimizer=adam \
    --train_size=${train_size} \
    --batch_size=${batch_size} \
    --num_epoch=${num_epoch}

  tch_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_tch
  echo ${tch_model_ckpt}
  python pretrain_tch.py \
    --tch_model_ckpt=${tch_model_ckpt} \
    --dataset_dir=${dataset_dir} \
    --tch_model_name=lenet \
    --optimizer=adam \
    --train_size=${train_size} \
    --batch_size=${batch_size} \
    --num_epoch=${num_epoch}

  dis_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_dis
  echo ${dis_model_ckpt}
  python pretrain_dis.py \
    --dis_model_ckpt=${dis_model_ckpt} \
    --dataset_dir=${dataset_dir} \
    --dis_model_name=lenet \
    --optimizer=adam \
    --train_size=${train_size} \
    --batch_size=${batch_size} \
    --num_epoch=${num_epoch}
}


train_size=100
batch_size=10
train ${train_size} ${batch_size}

train_size=1000
batch_size=50
train ${train_size} ${batch_size}

train_size=10000
batch_size=100
train ${train_size} ${batch_size}



train_size=5000
batch_size=50
train ${train_size} ${batch_size}

train_size=10000
batch_size=100
train ${train_size} ${batch_size}

train_size=50000
batch_size=100
train ${train_size} ${batch_size}