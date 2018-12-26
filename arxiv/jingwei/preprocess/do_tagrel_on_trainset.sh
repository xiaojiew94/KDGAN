# ./do_tagrel_on_trainset.sh yfcc8k vgg-verydeep-16fc7relu

export BASEDIR=/Users/xiaojiew1/Projects # mac
# export BASEDIR=/home/xiaojie/Projects
export SURVEY_DATA=$BASEDIR/data/yfcc100m/survey_data
export SURVEY_CODE=$BASEDIR/kdgan/jingwei
export SURVEY_DB=$BASEDIR/kdgan/logs
# export MATLAB_PATH=/Applications/MATLAB_R2017b.app # mac
export MATLAB_PATH=/usr/local
export PYTHONPATH=$PYTHONPATH:$SURVEY_CODE

codepath=$SURVEY_CODE/util/tagrel

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 collection feautre"
    exit
fi

collection=$1
feature=$2


if [ "$feature" = "color64+dsift" ]; then
    distance=l1
elif [ "$feature" = "vgg-verydeep-16fc7relu" ]; then 
    distance=cosine
    # distance=l2
else
    echo "unknown feature $feature"
    exit
fi 

python $codepath/dotagrel.py $collection $feature $collection --distance $distance --rootpath $SURVEY_DATA
