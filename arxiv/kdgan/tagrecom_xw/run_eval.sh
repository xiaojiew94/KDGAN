proj_dir=$HOME/Projects
kdgan_dir=$proj_dir/kdgan_xw
runs_dir=${kdgan_dir}/results/runs
pkls_dir=${kdgan_dir}/results/pkls

dataset=yfcc10k
image_model=vgg_16
feature_size=4096
train_dataset=yfcc9k
test_dataset=yfcc0k

kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
pretrained_dir=${checkpoint_dir}/pretrained

variant=basic
dataset=yfcc10k
dis_model_ckpt=${checkpoint_dir}/dis_$variant.ckpt
gen_model_ckpt=${checkpoint_dir}/gen_$variant.ckpt
tch_model_ckpt=${checkpoint_dir}/tch_$variant.ckpt


python eval_model.py \
  --dataset=$dataset \
  --image_model=${image_model} \
  --feature_size=${feature_size} \
  --model_name=gen \
  --model_ckpt=${gen_model_ckpt} \
  --model_run=${runs_dir}/${train_dataset}_${test_dataset}_gen.run
exit


