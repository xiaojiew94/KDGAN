kdgan_dir=/home/xiaojie/Projects/kdgan/kdgan
pretrained_dir=$kdgan_dir/checkpoints/pretrained

# python pretrain_gen.py \
#     --model_name=vgg_16 \
#     --preprocessing_name=vgg_16 \
#     --checkpoint_path=$pretrained_dir/vgg_16.ckpt \
#     --checkpoint_exclude_scopes=vgg_16/fc8 \
#     --trainable_scopes=vgg_16/fc8 \
#     --num_epoch=1000

# python pretrain_tch.py \
#     --num_epochs=1000

# exit

python main.py \
    --model_name=vgg_16 \
    --preprocessing_name=vgg_16 \
    --checkpoint_path=$pretrained_dir/vgg_16.ckpt \
    --checkpoint_exclude_scopes=vgg_16/fc8 \
    --trainable_scopes=vgg_16/fc8 \
    --num_epoch=1000