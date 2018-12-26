kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
train_size=100
batch_size=10

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
  --num_epoch=10 \
  --num_dis_epoch=5 \
  --num_gen_epoch=5 \
  --num_negative=20 \
  --num_positive=5
#mnist=100 bstacc=0.6490 et=7s
exit

python pretrain_gen.py \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=100 bstacc=0.5494 et=5s
exit

python pretrain_tch.py \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=100 bstacc=0.6602 et=21s
exit

python pretrain_dis.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=100 bstacc=0.6739 et=21s
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
  --num_epoch=400 \
  --kd_model=mimic
#mnist=100 mimic@339=65.35 iniacc=54.94 et=18s
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
#mnist=100 distn@130=68.34 iniacc=54.94 et=9s
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
  --num_epoch=400 \
  --kd_model=noisy \
  --noisy_ratio=0.1 \
  --noisy_sigma=0.1
#mnist=100 noisy@375=65.53 iniacc=54.94 et=16s
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
#mnist=100 kdgan_ow@96=77.26 et=316s
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










