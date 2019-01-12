kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
train_size=50
batch_size=5

cz_server=xiaojie@10.100.228.149 # cz
xw_server=xiaojie@10.100.228.181 # xw

# scp ${cz_server}:${checkpoint_dir}/mdlcompr_mnist${train_size}* ${checkpoint_dir}

# mac
# scp ${cz_server}:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/mdlcompr_mnist${train_size}_gen.data-00000-of-00001 ${checkpoint_dir}
# scp ${cz_server}:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/mdlcompr_mnist${train_size}_gen.index ${checkpoint_dir}
# scp ${cz_server}:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/mdlcompr_mnist${train_size}_gen.meta ${checkpoint_dir}
# scp ${cz_server}:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/mdlcompr_mnist${train_size}_dis.data-00000-of-00001 ${checkpoint_dir}
# scp ${cz_server}:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/mdlcompr_mnist${train_size}_dis.index ${checkpoint_dir}
# scp ${cz_server}:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/mdlcompr_mnist${train_size}_dis.meta ${checkpoint_dir}

# scp ${checkpoint_dir}/mdlcompr_mnist${train_size}_gen.data-00000-of-00001 ${cz_server}:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints
# scp ${checkpoint_dir}/mdlcompr_mnist${train_size}_gen.index ${cz_server}:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints
# scp ${checkpoint_dir}/mdlcompr_mnist${train_size}_gen.meta ${cz_server}:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints
# scp ${checkpoint_dir}/mdlcompr_mnist${train_size}_dis.data-00000-of-00001 ${cz_server}:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints
# scp ${checkpoint_dir}/mdlcompr_mnist${train_size}_dis.index ${cz_server}:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints
# scp ${checkpoint_dir}/mdlcompr_mnist${train_size}_dis.meta ${cz_server}:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints

python train_gan.py \
  --dis_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_dis \
  --gen_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_gen \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --gen_model_name=mlp \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --dis_learning_rate=1e-3 \
  --gen_learning_rate=1e-3 \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_negative=20 \
  --num_positive=5
exit

python pretrain_gen.py \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=50 bstacc=0.5209 et=5s
exit

python pretrain_tch.py \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=50 bstacc=0.6343 et=21s
exit

python pretrain_dis.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=50 bstacc=0.6294 et=20s
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
#mnist=50 mimic@325=62.74 iniacc=52.09 et=15s
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
#mnist=50 distn@190=63.92 iniacc=52.09 et=7s
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
#mnist=50 noisy@232=62.18 iniacc=52.09 et=15s
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
  --dis_learning_rate=1e-3 \
  --gen_learning_rate=5e-4 \
  --tch_learning_rate=5e-4 \
  --num_epoch=100 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_tch_epoch=10 \
  --kdgan_model=odgan \
  --num_negative=20 \
  --num_positive=5 \
  --kd_model=mimic \
  --noisy_ratio=0.1 \
  --noisy_sigma=0.1
#mnist=50 kdgan_ow@87=72.32 et=268s
exit

################################################################
#
# backup
#
################################################################

num_epoch=200
pickle_dir=${kdgan_dir}/pickles
gen_model_p=${pickle_dir}/mdlcompr_mnist${train_size}_gen@${num_epoch}.p
tch_model_p=${pickle_dir}/mdlcompr_mnist${train_size}_tch@${num_epoch}.p
gan_model_p=${pickle_dir}/mdlcompr_mnist${train_size}_gan@${num_epoch}.p
kdgan_model_p=${pickle_dir}/mdlcompr_mnist${train_size}_kdgan@${num_epoch}.p

python pretrain_gen.py \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200 \
  --all_learning_curve_p=${gen_model_p} \
  --collect_cr_data=True
exit

python pretrain_tch.py \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200 \
  --all_learning_curve_p=${tch_model_p} \
  --collect_cr_data=True
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
  --dis_learning_rate=1e-3 \
  --gen_learning_rate=5e-4 \
  --tch_learning_rate=5e-4 \
  --num_epoch=${num_epoch} \
  --num_dis_epoch=40 \
  --num_gen_epoch=20 \
  --num_tch_epoch=20 \
  --kdgan_model=odgan \
  --num_negative=20 \
  --num_positive=5 \
  --kd_model=mimic \
  --noisy_ratio=0.1 \
  --noisy_sigma=0.1 \
  --all_learning_curve_p=${kdgan_model_p} \
  --collect_cr_data=True
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
  --dis_learning_rate=5e-3 \
  --gen_learning_rate=1e-3 \
  --num_epoch=${num_epoch} \
  --num_dis_epoch=10 \
  --num_gen_epoch=5 \
  --num_negative=20 \
  --num_positive=5 \
  --all_learning_curve_p=${gan_model_p} \
  --collect_cr_data=True
exit









