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

intelltch_weight=0.3
distilled_weight=4.000
for intellstd_weight in 1e-0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7
do
  epk_learning_curve_p=${pickle_dir}/tagrecom_yfcc10k_kdgan_${intelltch_weight}_${distilled_weight}_${intellstd_weight}.p
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
    --num_epoch=60 \
    --num_dis_epoch=20 \
    --num_gen_epoch=10 \
    --num_tch_epoch=10 \
    --intelltch_weight=${intelltch_weight} \
    --distilled_weight=${distilled_weight} \
    --intellstd_weight=${intellstd_weight}
done
exit

for intelltch_weight in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  for distilled_weight in 0.125 0.250 0.500 1.000 2.000 4.000 8.000
  do
    epk_learning_curve_p=${pickle_dir}/tagrecom_yfcc10k_kdgan_${intelltch_weight}_${distilled_weight}.p
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
      --num_epoch=60 \
      --num_dis_epoch=20 \
      --num_gen_epoch=10 \
      --num_tch_epoch=10 \
      --intelltch_weight=${intelltch_weight} \
      --distilled_weight=${distilled_weight} 
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
--intelltch_weight=0.5 \
--distilled_weight=1.000 