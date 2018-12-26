kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
pickle_dir=${kdgan_dir}/pickles

cz_server=xiaojie@10.100.228.149 # cz
cy_server=xiaojie@10.100.228.151 # xy

# scp ${cz_server}:${checkpoint_dir}/mdlcompr_mnist* ${checkpoint_dir}
# scp ${checkpoint_dir}/mdlcompr_mnist* ${cy_server}:${checkpoint_dir}

tune() {
  train_size=$1
  batch_size=$2
  intelltch_weights=$3
  distilled_weights=$4
  for intelltch_weight in ${intelltch_weights[*]}
  do
    for distilled_weight in ${distilled_weights[*]}
    do
    dis_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_dis
    gen_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_gen
    tch_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_tch
    epk_learning_curve_p=${pickle_dir}/mdlcompr_mnist${train_size}_kdgan_${intelltch_weight}_${distilled_weight}.p
    if [ -e ${epk_learning_curve_p} ]
    then
      continue
    fi
    echo ${epk_learning_curve_p}
    python train_kdgan.py \
      --dis_model_ckpt=${dis_model_ckpt} \
      --gen_model_ckpt=${gen_model_ckpt} \
      --tch_model_ckpt=${tch_model_ckpt} \
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
      --num_negative=20 \
      --num_positive=5 \
      --kd_model=mimic \
      --noisy_ratio=0.1 \
      --noisy_sigma=0.1 \
      --intelltch_weight=${intelltch_weight} \
      --distilled_weight=${distilled_weight}
    done
  done
}

train_size=10000
batch_size=100
intelltch_weights=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
distilled_weights=(0.125 0.250 0.500 1.000 2.000 4.000 8.000)
# tune ${train_size} ${batch_size} ${intelltch_weights} ${distilled_weights}

train_size=5000
batch_size=50
intelltch_weights=(1.0 0.9 0.8 0.7 0.6 0.5)
distilled_weights=(8.000 4.000 2.000 1.000 0.500 0.250 0.125)
# tune ${train_size} ${batch_size} ${intelltch_weights} ${distilled_weights}

gamma() {
  train_size=$1
  batch_size=$2
  intelltch_weight=$3
  distilled_weight=$4
  for intellstd_weight in 1e-0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7
  do

    dis_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_dis
    gen_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_gen
    tch_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_tch
    epk_learning_curve_p=${pickle_dir}/mdlcompr_mnist${train_size}_kdgan_${intelltch_weight}_${distilled_weight}_${intellstd_weight}.p
    echo ${epk_learning_curve_p}
    if [ -e ${epk_learning_curve_p} ]
    then
      continue
    fi
    python train_kdgan.py \
      --dis_model_ckpt=${dis_model_ckpt} \
      --gen_model_ckpt=${gen_model_ckpt} \
      --tch_model_ckpt=${tch_model_ckpt} \
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
      --num_negative=20 \
      --num_positive=5 \
      --kd_model=mimic \
      --noisy_ratio=0.1 \
      --noisy_sigma=0.1 \
      --intelltch_weight=${intelltch_weight} \
      --distilled_weight=${distilled_weight} \
      --intellstd_weight=${intellstd_weight}
  done
}

train_size=50
batch_size=5
intelltch_weight=0.3
distilled_weight=4.000
gamma ${train_size} ${batch_size} ${intelltch_weight} ${distilled_weight}

train_size=10000
batch_size=100
intelltch_weight=0.8
distilled_weight=2.000
gamma ${train_size} ${batch_size} ${intelltch_weight} ${distilled_weight}
