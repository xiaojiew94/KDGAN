kdgan_dir=/home/xiaojie/Projects/kdgan/kdgan
pretrained_dir=$kdgan_dir/checkpoints/pretrained

python yfcc20k.py \
    --model_name=vgg_16 \
    --preprocessing_name=vgg_16 \
    --checkpoint_path=$pretrained_dir/vgg_16.ckpt \
    --end_point=vgg_16/fc7 \
    --num_epoch=500
# 4096
exit