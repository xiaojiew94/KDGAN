kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
train_size=50000
batch_size=128

dataset_dir=$HOME/Projects/data/cifar
std_model_ckpt=${checkpoint_dir}/mdlcompr_cifar${train_size}_std
tch_model_ckpt=${checkpoint_dir}/mdlcompr_cifar${train_size}_tch

python pretrain_tch.py \
  --tch_model_ckpt=${tch_model_ckpt} \
  --dataset_dir=${dataset_dir} \
  --image_size=32 \
  --train_size=${train_size} \
  --batch_size=${batch_size} \
  --learning_rate_decay_factor=0.95 \
  --num_epochs_per_decay=10.0 \
  --num_epoch=20
#cifar=50000 final=0.6536
exit

python pretrain_std.py \
  --std_model_ckpt=${std_model_ckpt} \
  --dataset_dir=${dataset_dir} \
  --train_size=${train_size} \
  --batch_size=${batch_size} \
  --learning_rate_decay_factor=0.96 \
  --num_epochs_per_decay=10.0 \
  --num_epoch=200
#cifar=50000 final=0.8420
exit