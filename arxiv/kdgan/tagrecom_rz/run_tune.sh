kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
pretrained_dir=${checkpoint_dir}/pretrained
pickle_dir=${kdgan_dir}/pickles

variant=basic
dataset=yfcc10k
image_model=vgg_16
dis_model_ckpt=${checkpoint_dir}/dis_$variant.ckpt
gen_model_ckpt=${checkpoint_dir}/gen_$variant.ckpt
tch_model_ckpt=${checkpoint_dir}/tch_$variant.ckpt

for alpha in 0.0 0.2 0.4 0.6 0.8 1.0 # 0.1 0.3 0.5 0.7 0.9
do
  for beta in 0.0 0.2 0.4 0.6 0.8 1.0
  do
    for gamma in 0.0 0.2 0.4 0.6 0.8 1.0
    do
      epk_learning_curve_p=${pickle_dir}/tagrecom_yfcc10k_kdgan_${alpha}_${beta}_${gamma}.p
      echo ${epk_learning_curve_p}
      python train_kdgan.py \
        --dis_model_ckpt=${dis_model_ckpt} \
        --gen_model_ckpt=${gen_model_ckpt} \
        --tch_model_ckpt=${tch_model_ckpt} \
        --epk_learning_curve_p=${epk_learning_curve_p} \
        --dataset=$dataset \
        --image_model=${image_model} \
        --optimizer=sgd \
        --learning_rate_decay_type=exp \
        --dis_learning_rate=0.05 \
        --gen_learning_rate=0.01 \
        --tch_learning_rate=0.01 \
        --kd_model=mimic \
        --kd_soft_pct=0.1 \
        --temperature=3.0 \
        --num_epoch=50 \
        --num_dis_epoch=20 \
        --num_gen_epoch=10 \
        --num_tch_epoch=10 \
        --alpha=$alpha \
        --beta=$beta \
        --gamma=$gamma
    done
  done
done

exit
epk_learning_curve_p=${pickle_dir}/tagrecom_yfcc10k_kdgan_dev.p
echo ${epk_learning_curve_p}
python train_kdgan.py \
--dis_model_ckpt=${dis_model_ckpt} \
--gen_model_ckpt=${gen_model_ckpt} \
--tch_model_ckpt=${tch_model_ckpt} \
--epk_learning_curve_p=${epk_learning_curve_p} \
--dataset=$dataset \
--image_model=${image_model} \
--optimizer=sgd \
--learning_rate_decay_type=exp \
--dis_learning_rate=0.05 \
--gen_learning_rate=0.01 \
--tch_learning_rate=0.01 \
--kd_model=mimic \
--kd_soft_pct=0.1 \
--temperature=3.0 \
--num_epoch=50 \
--num_dis_epoch=20 \
--num_gen_epoch=10 \
--num_tch_epoch=10 \
--alpha=0.0 \
--beta=0.0 \
--gamma=0.2