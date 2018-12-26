# kdgan
pip install -Ue .

remote.unimelb.edu.au/student
ssh xiaojie@10.100.229.246 # cpu
ssh xiaojie@10.100.228.151 # gpu cy
ssh xiaojie@10.100.228.149 # gpu cz
ssh xiaojie@10.100.228.181 # gpu xw

dl-acm-org.ezp.lib.unimelb.edu.au

################################################################
# init
################################################################
https://github.com/xiaojiew1/kdgan/commits/master?after=629993e50c7c5e9a455f45498491577d16ab1278+1119

################################################################
# ulord
################################################################
wget https://www.ulord.one/downloads/UlordRig-Linux-V1.0.0.zip
unzip
{ 
    "threads":16,                                 // number of miner threads
    "pools": [
        {
            "url": "stratum+tcp://main-pool.ulorders.com:18888",                // URL of mining server
            "user": "Ue3jGaQ65waLLsPHHues2axfCE1qtwW36a.worker01",   // username for mining server 
            "pass": "x"                                            // password for mining server
                                                        
        }
    ]                
}
chmod +x ulordrig
./ulordrig -B -l ulord.log
ps -e | grep ulord

################################################################
# printer
################################################################
find printer
168L7.xx-ToshibaEstudio3555c @ 4000D-114949-M

ping printer
4000D-114949-M.local

modify device url to
lpd://uom123759.printer.unimelb.net.au

download ppd from
http://www.openprinting.org/printer/Toshiba/Toshiba-e-Studio_3500c

change printer ppd following
https://www.utwente.nl/en/lisa/ict/manuals/printing/ubuntu/#before-you-get-started

################################################################
# todo
################################################################
http://jmlr.csail.mit.edu/papers/volume5/greensmith04a/greensmith04a.pdf
https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/


# backup
ssh xiaojie@10.100.228.28 # gpu yz
ssh xiaojiewang@10.100.228.28 # gpu yz # initialpassword

# bank
5217291828507288

# text classification
git reset --hard 7c0564610815732283cc968c387d4b000fa38a68

conda create -n py27 python=2.7
conda create -n py34 python=3.4

# tensorflow tensorboard
export CUDA_VISIBLE_DEVICES=''
ssh -NL 6006:localhost:6006 xiaojie@10.100.229.246 # cpu
ssh -NL 6006:localhost:6006 xiaojie@10.100.228.181 # gpu

python mnist_bn_wi.py --weight-init xavier --bias-init zero --batch-norm True

virtualenv --system-site-packages venv
pip install --ignore-installed --upgrade tensorflow
pip install --ignore-installed -r requirements.txt

################################################################
#
# baseline
#
################################################################

# jingwei: extract image features by vgg16
cd jingwei/image_feature/matcovnet/
wget http://lixirong.net/data/csur2016/matconvnet-1.0-beta8.tar.gz
tar -xzvf matconvnet-1.0-beta8.tar.gz
wget http://lixirong.net/data/csur2016/matconvnet-models.tar.gz
tar -xzvf matconvnet-models.tar.gz
>> http://www.vlfeat.org/matconvnet/install/
>> addpath matlab
>> vl_compilenn
matlab -nodisplay -nosplash -nodesktop -r "run('extract_vggnet.m');"
# jingwei: precompute k nearest neighbors
conda install libgcc # ubuntu
brew install boost --c++11 # mac
cd jingwei/util/simpleknn/
sudo apt-get install libboost-dev
./build.sh
# jingwei: knn
./do_knntagrel.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu
# jingwei: tagprop
import nltk & nltk.download('wordnet')
./do_getknn.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu 0 1 1
./do_getknn.sh yfcc9k yfcc9k vgg-verydeep-16-fc7relu 0 1 1
jingwei/model_based/tagprop
setup-tagprop.sh
# jingwei: evaluation
./eval_pickle.sh yfcc0k

################################################################
#
# model compression
#
################################################################

python download_and_convert_data.py \
  --dataset_name=mnist \
  --dataset_dir=$HOME/Projects/data/mnist

python train_image_classifier.py \
  --train_dir=$HOME/Projects/kdgan/kdgan/slimmodels \
  --dataset_name=mnist \
  --dataset_split_name=train \
  --dataset_dir=$HOME/Projects/data/mnist \
  --model_name=lenet

python eval_image_classifier.py \
  --alsologtostderr \
  --checkpoint_path=$HOME/Projects/kdgan/kdgan/slimmodels \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --model_name=lenet

# cifar 10
http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130
https://github.com/BIGBALLON/cifar-10-cnn

# mnist
https://github.com/clintonreece/keras-cloud-ml-engine
https://github.com/keras-team/keras/tree/master/examples

https://github.com/hwalsuklee/how-far-can-we-go-with-MNIST
http://www.pythonexample.com/user/vamsiramakrishnan

# gan trick
https://github.com/gitlimlab/SSGAN-Tensorflow

# backup
https://github.com/tensorflow/models/tree/master/official/resnet
https://github.com/BIGBALLON/cifar-10-cnn
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10
https://github.com/shmsw25/cifar10-classification
https://github.com/ethereon/caffe-tensorflow
