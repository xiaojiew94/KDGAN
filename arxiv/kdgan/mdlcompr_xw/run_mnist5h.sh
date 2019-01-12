kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=$kdgan_dir/checkpoints
train_size=500
batch_size=10

# scp xiaojie@10.100.228.149:$checkpoint_dir/mdlcompr_mnist* $checkpoint_dir

python pretrain_gen.py \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=500 bstacc=0.7666 et=17s
exit

python pretrain_tch.py \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=500 bstacc=0.8932 et=41s
exit

python pretrain_dis.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=500 bstacc=0.8799 et=41s
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
  --num_epoch=200 \
  --kd_model=mimic
#mnist=500 mimic@186=88.66 iniacc=76.66 et=39s
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
  --num_epoch=200 \
  --kd_model=distn \
  --kd_soft_pct=0.7 \
  --temperature=3.0
#mnist=500 distn@193=89.59 iniacc=76.66 et=36s
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
  --num_epoch=200 \
  --kd_model=noisy \
  --noisy_ratio=0.1 \
  --noisy_sigma=0.1
#mnist=500 noisy@195=88.42 iniacc=76.66 et=33s
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
  --num_epoch=100 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_tch_epoch=10 \
  --kdgan_model=ow \
  --num_negative=20 \
  --num_positive=5 \
  --kd_model=mimic \
  --noisy_ratio=0.1 \
  --noisy_sigma=0.1
#mnist=1000 kdgan_ow=0.9356 et=1459s
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
  --kd_soft_pct=0.3 \
  --temperature=3.0
#mnist=10000 kdgan_ow=0.9786 et=10419s
exit

python train_gan.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --gen_model_name=mlp \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_gen_epoch=2 \
  --num_negative=20 \
  --num_positive=5
exit










