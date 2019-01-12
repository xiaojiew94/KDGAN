cy_server=xiaojie@10.100.228.151 # cy
cz_server=xiaojie@10.100.228.149 # cz
xw_server=xiaojie@10.100.228.181 # xw

kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
pickle_dir=${kdgan_dir}/pickles
src_yfcc_dir=${pickle_dir}
dst_yfcc_dir=${pickle_dir}
[ -d $dst_yfcc_dir ] || mkdir -p $dst_yfcc_dir

################################################################
#
# convergence rate
#
################################################################

num_epoch=200
# scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_gen@${num_epoch}.p ${dst_yfcc_dir}
# scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_tch@${num_epoch}.p ${dst_yfcc_dir}
# scp ${cz_server}:${src_yfcc_dir}/tagrecom_yfcc10k_gan@${num_epoch}.p ${dst_yfcc_dir}
# scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_kdgan@${num_epoch}.p ${dst_yfcc_dir}

train_size=50
# scp ${xw_server}:${src_yfcc_dir}/mdlcompr_mnist${train_size}_gen@${num_epoch}.p ${dst_yfcc_dir}
# scp ${xw_server}:${src_yfcc_dir}/mdlcompr_mnist${train_size}_tch@${num_epoch}.p ${dst_yfcc_dir}
# scp ${cz_server}:${src_yfcc_dir}/mdlcompr_mnist${train_size}_gan@${num_epoch}.p ${dst_yfcc_dir}
# scp ${xw_server}:${src_yfcc_dir}/mdlcompr_mnist${train_size}_kdgan@${num_epoch}.p ${dst_yfcc_dir}

################################################################
#
# parameter tuning
#
################################################################

scp ${cy_server}:${src_yfcc_dir}/mdlcompr_mnist*_kdgan_*.p ${dst_yfcc_dir}
scp ${cz_server}:${src_yfcc_dir}/mdlcompr_mnist*_kdgan_*.p ${dst_yfcc_dir}

scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_kdgan_*.p ${dst_yfcc_dir}
scp ${xw_server}:${src_yfcc_dir}/mdlcompr_mnist*_kdgan_*.p ${dst_yfcc_dir}






