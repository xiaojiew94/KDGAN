# ./eval_pickle.sh yfcc_rnd_tn yfcc_rnd_vd
# zip -r survey_data.zip survey_data -x */ImageData/**\*
# scp xiaojie@10.100.228.181:~/Projects/data/yfcc100m/survey_data.zip .
# scp -r xiaojie@10.100.228.181:~/Projects/kdgan_xw/results/temp .

export BASEDIR=/home/xiaojie/Projects
export SURVEY_DATA=$BASEDIR/data/yfcc100m/survey_data
export SURVEY_CODE=$BASEDIR/kdgan_xw/jingwei
export SURVEY_DB=$BASEDIR/kdgan_xw/results/temp
export MATLAB_PATH=/usr/local
export PYTHONPATH=$PYTHONPATH:$SURVEY_CODE

rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 trainCollection testCollection"
    exit
fi

runs_dir=$BASEDIR/kdgan_xw/results/runs
trainCollection=$1
testCollection=$2
testAnnotationName=concepts.txt
conceptfile=$rootpath/$testCollection/Annotations/$testAnnotationName


export SURVEY_EVAL=$BASEDIR/kdgan_xw/results/eval
[ -d $SURVEY_EVAL ] || mkdir $SURVEY_EVAL
runfile=$SURVEY_EVAL/runs_"$testCollection".txt
resfile=$SURVEY_EVAL/runs_"$testCollection".res
echo $runfile
echo $resfile

ls -d "$SURVEY_DB"/* > $runfile
# ls -d "$SURVEY_DB"/* | grep kdgan > $runfile
# if [ -f "$resfile" ]; then
#     echo "result file exists at $resfile"
#     exit
# fi


if [ ! -f "$runfile" ]; then
    echo "runfile $runfile not exists!"
    exit
fi

if [ "$testCollection" == "flickr81" ]; then
    annotationName=concepts81.txt
elif [ "$testCollection" == "mirflickr08" ]; then
    annotationName=conceptsmir14.txt
elif [ "$testCollection" == "flickr55" -o "$testCollection" == "flickr51" ]; then
    annotationName=concepts51ms.txt
elif [ "$testCollection" == "yfcc0k" ]; then
    annotationName=concepts.txt
elif [ "$testCollection" == "yfcc1k" ]; then
    annotationName=concepts.txt
elif [ "$testCollection" == "yfcc_rnd_vd" ]; then
    annotationName=concepts.txt
else
    echo "unknown testCollection $testCollection"
    exit
fi

# python $codepath/eval/eval_pickle.py $testCollection $annotationName $runfile > $resfile
python $codepath/eval/eval_pickle.py $testCollection $annotationName $runfile

