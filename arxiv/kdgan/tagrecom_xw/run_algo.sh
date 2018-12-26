kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
pretrained_dir=${checkpoint_dir}/pretrained

variant=basic
dataset=yfcc10k
image_model=vgg_16
dis_model_ckpt=${checkpoint_dir}/dis_$variant.ckpt
gen_model_ckpt=${checkpoint_dir}/gen_$variant.ckpt
tch_model_ckpt=${checkpoint_dir}/tch_$variant.ckpt

python pretrain_gen.py \
  --gen_model_ckpt=${gen_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=exp \
  --gen_learning_rate=0.05 \
  --num_epoch=200
exit

python pretrain_dis.py \
  --dis_model_ckpt=${dis_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=exp \
  --dis_learning_rate=0.05 \
  --num_epoch=200
exit

python pretrain_tch.py \
  --tch_model_ckpt=${tch_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=exp \
  --tch_learning_rate=0.05 \
  --epk_train=0.95 \
  --epk_valid=0.05 \
  --num_epoch=200
exit

python train_kd.py \
  --gen_model_ckpt=${gen_model_ckpt} \
  --tch_model_ckpt=${tch_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=fix \
  --gen_learning_rate=0.05 \
  --kd_model=distn \
  --kd_soft_pct=0.1 \
  --temperature=3.0 \
  --num_epoch=200
  # kd_soft_pct=0.0 best@120=0.2880
exit

python train_kd.py \
  --gen_model_ckpt=${gen_model_ckpt} \
  --tch_model_ckpt=${tch_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=fix \
  --gen_learning_rate=0.1 \
  --kd_model=mimic \
  --kd_soft_pct=0.1 \
  --num_epoch=200
exit

python train_gan.py \
  --dis_model_ckpt=${dis_model_ckpt} \
  --gen_model_ckpt=${gen_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --image_weight_decay=0.0 \
  --optimizer=sgd \
  --learning_rate_decay_type=exp \
  --dis_learning_rate=0.05 \
  --gen_learning_rate=0.01 \
  --num_epochs_per_decay=20.0 \
  --learning_rate_decay_factor=0.95 \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10
  # best@3=0.2953 et=13887s
exit

python train_kdgan.py \
  --dis_model_ckpt=${dis_model_ckpt} \
  --gen_model_ckpt=${gen_model_ckpt} \
  --tch_model_ckpt=${tch_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=exp \
  --dis_learning_rate=0.05 \
  --gen_learning_rate=0.01 \
  --tch_learning_rate=0.01 \
  --kd_model=distn \
  --kd_soft_pct=0.1 \
  --temperature=3.0 \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_tch_epoch=10
exit



