kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
pretrained_ckpt=${checkpoint_dir}/vgg_16.ckpt
overwrite=True # False
baseline=True

python yfcc_small_rnd.py \
    --model_name=vgg_16 \
    --preprocessing_name=vgg_16 \
    --end_point=vgg_16/fc7 \
    --pretrained_ckpt=${pretrained_ckpt} \
    --overwrite=${overwrite} \
    --baseline=${baseline}
exit


# tar -zcvf archive-name.tar.gz directory-name
# scp xiaojie@10.100.228.181:/home/xiaojie/Projects/data/yfcc100m/yfcc10k/* .
if [ ! -f $checkpoint_path ]; then
    [ -d $pretrained_dir ] || mkdir -p $pretrained_dir
    temp_path=$pretrained_dir/vgg_16_2016_08_28.tar.gz
    curl http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz \
        -o $temp_path
    tar -xzvf $temp_path -C $pretrained_dir
    rm $temp_path
fi


python yfcc10k.py \
    --model_name=vgg_16 \
    --preprocessing_name=vgg_16 \
    --checkpoint_path=$checkpoint_path \
    --end_point=vgg_16/fc7 \
    --num_epoch=500
# 4096
exit


python yfcc10k.py \
    --model_name=inception_resnet_v2 \
    --preprocessing_name=inception_resnet_v2 \
    --checkpoint_path=$pretrained_dir/inception_resnet_v2_2016_08_30.ckpt \
    --end_point=global_pool \
    --num_epoch=500
# 1536
exit


python yfcc10k_fast.py \
    --model_name=vgg_19 \
    --preprocessing_name=vgg_19 \
    --checkpoint_path=$pretrained_dir/vgg_19.ckpt \
    --end_point=vgg_19/fc7
# 4096
exit

python yfcc10k_fast.py \
    --model_name=resnet_v2_152 \
    --preprocessing_name=resnet_v2_152 \
    --checkpoint_path=$pretrained_dir/resnet_v2_152.ckpt \
    --end_point=global_pool
#2048
exit