kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
pretrained_dir=${checkpoint_dir}/pretrained
pickle_dir=${kdgan_dir}/pickles
picture_dir=${kdgan_dir}/pictures

num_epoch=200
gen_model_p=${pickle_dir}/tagrecom_yfcc10k_gen@${num_epoch}.p
tch_model_p=${pickle_dir}/tagrecom_yfcc10k_tch@${num_epoch}.p
gan_model_p=${pickle_dir}/tagrecom_yfcc10k_gan@${num_epoch}.p
kdgan_model_p=${pickle_dir}/tagrecom_yfcc10k_kdgan@${num_epoch}.p
python plot_tr_cr.py \
  --gen_model_p=${gen_model_p} \
  --tch_model_p=${tch_model_p} \
  --gan_model_p=${gan_model_p} \
  --kdgan_model_p=${kdgan_model_p} \
  --num_epoch=${num_epoch} \
  --epsfile=${picture_dir}/tagrecom_yfcc10k_cr@${num_epoch}.eps

train_size=50
gen_model_p=${pickle_dir}/mdlcompr_mnist${train_size}_gen@${num_epoch}.p
tch_model_p=${pickle_dir}/mdlcompr_mnist${train_size}_tch@${num_epoch}.p
gan_model_p=${pickle_dir}/mdlcompr_mnist${train_size}_gan@${num_epoch}.p
kdgan_model_p=${pickle_dir}/mdlcompr_mnist${train_size}_kdgan@${num_epoch}.p
python plot_mc_cr.py \
  --gen_model_p=${gen_model_p} \
  --tch_model_p=${tch_model_p} \
  --gan_model_p=${gan_model_p} \
  --kdgan_model_p=${kdgan_model_p} \
  --num_epoch=${num_epoch} \
  --epsfile=${picture_dir}/mdlcompr_mnist${train_size}_cr@${num_epoch}.eps
exit


