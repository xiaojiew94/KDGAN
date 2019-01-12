kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
train_size=10000
batch_size=100

# scp xiaojie@10.100.228.149:$checkpoint_dir/mdlcompr_mnist* $checkpoint_dir

python train_gan.py \
  --dis_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_dis \
  --gen_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_gen \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --gen_model_name=mlp \
  --optimizer=adam \
  --train_size=${train_size} \
  --batch_size=${batch_size} \
  --num_epoch=20 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_negative=20 \
  --num_positive=5
#mnist=10000 bstacc=0.9885 et=1848s
exit

python pretrain_gen.py \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=10000 bstacc=0.9699 et=38s
exit

python pretrain_tch.py \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=10000 bstacc=0.9899 et=116s
exit

python pretrain_dis.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=10000 bstacc=0.9896 et=115s
exit

python train_kd.py \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=100 \
  --kd_model=mimic
#mnist=10000 mimic@86=98.65 iniacc=0.9713 et=118s
exit

python train_kd.py \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=100 \
  --kd_model=distn \
  --kd_soft_pct=0.7 \
  --temperature=3.0
#mnist=10000 distn@91=98.79 iniacc=0.9713 et=125s
exit

python train_kd.py \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=100 \
  --kd_model=noisy \
  --noisy_ratio=0.1 \
  --noisy_sigma=0.1
#mnist=10000 noisy@79=98.58 iniacc=0.9713 et=121s
exit

python train_kdgan.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_tch_epoch=10 \
  --kdgan_model=ow \
  --num_negative=20 \
  --num_positive=5 \
  --kd_model=mimic \
  --noisy_ratio=0.1 \
  --noisy_sigma=0.1
#mnist=10000 kdgan_ow=0.9913 tot=14761s
exit

################################################################
#
# backup
#
################################################################

python train_kdgan.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_tch_epoch=10 \
  --kdgan_model=tw \
  --num_negative=20 \
  --num_positive=5 \
  --kd_model=mimic \
  --kd_soft_pct=0.0 \
  --temperature=3.0
exit











