cz_server=xiaojie@10.100.228.149 # cz
xw_server=xiaojie@10.100.228.181 # xw

# scp xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/gen_vgg_16.eval .

################################################################
#
# download tagrecom data
#
################################################################

yfcc_dir=Projects/data/yfcc100m
survey_data=survey_data.zip
src_yfcc_dir=/home/xiaojie/$yfcc_dir
dst_yfcc_dir=$HOME/$yfcc_dir
# [ -f $dst_yfcc_dir/$survey_data ] || scp $server:$src_yfcc_dir/$survey_data $dst_yfcc_dir

dataset=yfcc10k

src_data_dir=$server:/home/xiaojie/$yfcc_dir/$dataset
dst_data_dir=$HOME/$yfcc_dir/$dataset
echo $src_data_dir
echo $dst_data_dir

[ -d $dst_data_dir ] || mkdir -p $dst_data_dir
# scp $src_data_dir/$dataset.label $dst_data_dir
# scp $src_data_dir/$dataset.vocab $dst_data_dir
# scp $src_data_dir/$dataset.train $dst_data_dir
# scp $src_data_dir/$dataset.valid $dst_data_dir

src_precomputed_dir=$src_data_dir/Precomputed
dst_precomputed_dir=$dst_data_dir/Precomputed
[ -d ${dst_precomputed_dir} ] || mkdir -p ${dst_precomputed_dir}

scp ${src_precomputed_dir}/*.npy ${dst_precomputed_dir}
exit

model_name=vgg_16
scp ${src_precomputed_dir}/${dataset}_${model_name}_000.valid.tfrecord ${dst_precomputed_dir}
for i in $(seq -w 000 49)
do
  filename=${dataset}_${model_name}_${i}.train.tfrecord
  if [ -f ${dst_precomputed_dir}/$filename ]
  then
    continue
  fi
  scp ${src_precomputed_dir}/$filename ${dst_precomputed_dir}
done


################################################################
#
# download mdlcompr data
#
################################################################

exit

download_mdlcompr()
{
  model_name=$1
  echo "downloading $model_name"
  src_home_dir=$server:/home/xiaojie
  dst_home_dir=$HOME
  relative_dir=Projects/kdgan/kdgan/checkpoints/$model_name
  src_mdlcompr_mnist_dir=$src_home_dir/$relative_dir
  dst_mdlcompr_mnist_dir=$dst_home_dir/$relative_dir
  rm -rf ${dst_mdlcompr_mnist_dir}
  mkdir ${dst_mdlcompr_mnist_dir}
  scp $src_mdlcompr_mnist_dir/* $dst_mdlcompr_mnist_dir
}

model_name=mdlcompr_mnist_dis
download_mdlcompr ${model_name}
model_name=mdlcompr_mnist_gen
download_mdlcompr ${model_name}
model_name=mdlcompr_mnist_tch
download_mdlcompr ${model_name}
scp ${src_data_dir}/yfcc10k.vocab ${dst_precomputed_dir}