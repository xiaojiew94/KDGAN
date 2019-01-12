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

scp xiaojie@10.100.228.149:${runs_dir}/yfcc9k_yfcc0k_dis.run ${runs_dir}

for runfile in ${runs_dir}/*.run
do
  pklfile=${runfile//run/pkl}
  python $codepath/postprocess/pickle_tagvotes.py $conceptfile $runfile $pklfile
done


