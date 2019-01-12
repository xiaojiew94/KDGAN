kdgan_dir=$HOME/Projects/kdgan/kdgan
checkpoint_dir=$kdgan_dir/checkpoints
pretrained_dir=$checkpoint_dir/pretrained

python pretrain_dis.py \
  --dataset=yfcc10k \
  --dis_model_ckpt=$checkpoint_dir/dis_vgg_16.ckpt \
  --feature_size=4096 \
  --model_name=vgg_16 \
  --num_epoch=200
# 369s best hit=0.7747
exit

python pretrain_tch.py \
  --dataset=yfcc10k \
  --tch_model=tch \
  --model_name=vgg_16 \
  --tch_weight_decay=0.0 \
  --num_epoch=200
# 224s best hit=0.9373
exit

python pretrain_gen.py \
  --dataset=yfcc10k \
  --feature_size=4096 \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --model_name=vgg_16 \
  --num_epoch=200
# 366s best hit=0.7787
exit

python train_gan.py \
  --dis_model_ckpt=$checkpoint_dir/dis_vgg_16.ckpt \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --model_name=vgg_16 \
  --feature_size=4096 \
  --tch_weight_decay=0.0 \
  --gen_weight_decay=0.0 \
  --num_epoch=100 \
  --num_dis_epoch=50 \
  --num_gen_epoch=10
exit

python train_kd.py \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --tch_model_ckpt=$checkpoint_dir/tch.ckpt \
  --model_name=vgg_16 \
  --feature_size=4096 \
  --beta=0.00001 \
  --temperature=10.0 \
  --num_epoch=200
exit