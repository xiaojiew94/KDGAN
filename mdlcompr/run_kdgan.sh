hidden_size=800

kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
dataset_dir=$HOME/Projects/data/mnist
num_epoch=100

# train_size=5000
# batch_size=50
train_size=10000
batch_size=100
# train_size=50000
# batch_size=100

gen_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_gen${hidden_size}
tch_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_tch
dis_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_dis

intelltch_weight=0.8
distilled_weight=2.0
intellstd_weight=0.0001

# scp xiaojie@10.100.228.158:/home/xiaojie/Projects/kdgan_xw/kdgan/mdlcompr_rebuttal/run_kdgan8h_5k.log .

python train_kdgan.py \
  --dis_model_ckpt=${dis_model_ckpt} \
  --gen_model_ckpt=${gen_model_ckpt} \
  --tch_model_ckpt=${tch_model_ckpt} \
  --dataset_dir=${dataset_dir} \
  --dis_model_name=lenet \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=${train_size} \
  --batch_size=${batch_size} \
  --dis_learning_rate=1e-3 \
  --gen_learning_rate=5e-4 \
  --tch_learning_rate=5e-4 \
  --num_epoch=${num_epoch} \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_tch_epoch=10 \
  --num_negative=20 \
  --num_positive=5 \
  --kd_model=distn \
  --kd_soft_pct=0.3 \
  --temperature=3.0 \
  --intelltch_weight=${intelltch_weight} \
  --distilled_weight=${distilled_weight} \
  --intellstd_weight=${intellstd_weight} \
  --evaluate_tch=True \
  --enable_gumbel=True \
  --hidden_size=${hidden_size}
exit

