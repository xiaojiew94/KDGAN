kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=$kdgan_dir/checkpoints
pickle_dir=${kdgan_dir}/pickles

cz_server=xiaojie@10.100.228.149 # cz
cy_server=xiaojie@10.100.228.151 # xy

# scp ${cz_server}:${checkpoint_dir}/mdlcompr_mnist* ${checkpoint_dir}
# scp ${checkpoint_dir}/mdlcompr_mnist* ${cy_server}:${checkpoint_dir}

tune() {
  train_size=$1
  batch_size=$2
  for alpha in 0.0 0.2 0.4 0.6 0.8 1.0 # 0.1 0.3 0.5 0.7 0.9
  do
    for beta in 0.0 0.2 0.4 0.6 0.8 1.0
    do
      for gamma in 0.0 0.2 0.4 0.6 0.8 1.0
      do
        dis_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_dis
        gen_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_gen
        tch_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_tch

        epk_learning_curve_p=${pickle_dir}/mdlcompr_mnist${train_size}_kdgan_${alpha}_${beta}_${gamma}.p
        echo ${epk_learning_curve_p}
        python train_kdgan.py \
          --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
          --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
          --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
          --epk_learning_curve_p=${epk_learning_curve_p} \
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
          --num_epoch=60 \
          --num_dis_epoch=20 \
          --num_gen_epoch=10 \
          --num_tch_epoch=10 \
          --kdgan_model=odgan \
          --num_negative=20 \
          --num_positive=5 \
          --kd_model=mimic \
          --noisy_ratio=0.1 \
          --noisy_sigma=0.1 \
          --alpha=$alpha \
          --beta=$beta \
          --gamma=$gamma
      done
    done
  done
}

train_size=10000
batch_size=100
tune ${train_size} ${batch_size}

