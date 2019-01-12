kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
dataset_dir=$HOME/Projects/data/mnist
num_epoch=100

# scp xiaojie@10.100.228.181:/home/xiaojie/Projects/kdgan_xw/kdgan/mdlcompr_rebuttal/run_pretrain.log .

train() {
  train_size=$1
  batch_size=$2
  intelltch_weight=$3
  distilled_weight=$4
  for intellstd_weight in 0e-0 1e-0 1e-1 1e-2 1e-3 # 1e-4 1e-5 1e-6 1e-7
  do
    dis_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_dis
    gen_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_gen
    tch_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_tch
    tch_file=results/intellstd/mnist${train_size}_${intellstd_weight}.acc
    
    python train_kdgan.py \
      --dis_model_ckpt=${dis_model_ckpt} \
      --gen_model_ckpt=${gen_model_ckpt} \
      --tch_model_ckpt=${tch_model_ckpt} \
      --dataset_dir=${dataset_dir} \
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
      --intellstd_weight=${intellstd_weight} \
      --evaluate_tch=True \
      --tch_file=${tch_file}
  done
}

train_size=100
batch_size=10
intelltch_weight=0.9
distilled_weight=2.000
train ${train_size} ${batch_size} ${intelltch_weight} ${distilled_weight}

train_size=1000
batch_size=50
intelltch_weight=0.9
distilled_weight=0.250
train ${train_size} ${batch_size} ${intelltch_weight} ${distilled_weight}

train_size=10000
batch_size=100
intelltch_weight=0.8
distilled_weight=2.000
train ${train_size} ${batch_size} ${intelltch_weight} ${distilled_weight}




