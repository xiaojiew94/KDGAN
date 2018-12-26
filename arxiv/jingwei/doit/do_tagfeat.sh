# ./do_tagfeat.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 trainCollection testCollection vis_feature"
  exit
fi

overwrite=0
do_training=1

trainCollection=$1
testCollection=$2
vis_feature=$3


if [ "$vis_feature" != "color64+dsift" -a "$vis_feature" != "vgg-verydeep-16-fc7relu" ]; then
  echo "unknown visual feature $vis_feature"
  exit
fi

# feature=tag400-$trainCollection+$vis_feature
# if [ $do_training == 1 ]; then
#   feat_dir=$rootpath/$trainCollection/FeatureData/$feature
#   if [ ! -d "$feat_dir" ]; then
#     echo "$feat_dir does not exist"
#     $codepath/doit/do_extract_tagfeat.sh $trainCollection $vis_feature
#     exit
#   fi
# fi
feature=$vis_feature
feat_dir=$rootpath/$trainCollection/FeatureData/$feature

# conceptset=concepts130social
conceptset=concepts
baseAnnotationName=$conceptset.txt

conceptfile=$rootpath/$trainCollection/Annotations/$baseAnnotationName
if [ ! -f "$conceptfile" ]; then
  echo "$conceptfile does not exist"
  exit
fi

# nr_pos=100
# nr_neg_bags=1
# neg_pos_ratio=1
# trainAnnotationName=$conceptset.random$nr_pos.0.npr1.0.txt

# nr_pos=300
# nr_neg_bags=3
# neg_pos_ratio=3

# nr_pos=500
# nr_neg_bags=5
# neg_pos_ratio=5

nr_pos=1000
nr_neg_bags=10
neg_pos_ratio=10

neg_pos_ratio=1
nr_neg=$(($nr_pos * $neg_pos_ratio))
nr_pos_bags=1
pos_end=$(($nr_pos_bags - 1))
neg_end=$(($nr_neg_bags - 1))
neg_bag_num=1

modelAnnotationName=$conceptset.random$nr_pos.0-$pos_end.npr"$neg_pos_ratio".0-$neg_end.txt
trainAnnotationName=$conceptset.random$nr_pos.0.npr1.0.txt
# trainAnnotationName=$conceptset.random$nr_pos.0.npr1.$neg_end.txt

if [ $do_training == 1 ]; then
  python $codepath/model_based/generate_train_bags.py \
      $trainCollection $baseAnnotationName $nr_pos \
      --neg_pos_ratio $neg_pos_ratio \
      --neg_bag_num $nr_neg_bags

  bagfile=$rootpath/$trainCollection/annotationfiles/$conceptset.random$nr_pos.0-$pos_end.npr"$neg_pos_ratio".0-$neg_end.txt
  if [ ! -f "$bagfile" ]; then
    echo "bagfile $bagfile does not exist"
    exit
  fi

  python $codepath/model_based/generate_train_bags.py \
      $trainCollection $baseAnnotationName $nr_pos \
      --neg_pos_ratio $neg_pos_ratio \
      --neg_bag_num $neg_bag_num

  conceptfile=$rootpath/$trainCollection/Annotations/$trainAnnotationName

  if [ ! -f "$conceptfile" ]; then
    echo "conceptfile $conceptfile does not exist"
    exit
  fi

  for modelName in fastlinear fik
  do
    python $codepath/model_based/negative_bagging.py \
        $trainCollection $bagfile $feature $modelName

    python $codepath/model_based/svms/find_ab.py \
        $trainCollection $modelAnnotationName $trainAnnotationName $feature \
        --model $modelName
  done
fi


if [ "$testCollection" = "mirflickr08" ]; then
    testAnnotationName=conceptsmir14.txt
elif [ "$testCollection" = "flickr51" ]; then
    testAnnotationName=concepts51ms.txt
elif [ "$testCollection" = "flickr81" ]; then
    testAnnotationName=concepts81.txt
elif [ "$testCollection" == "yfcc0k" ]; then
    testAnnotationName=concepts.txt
elif [ "$testCollection" == "yfcc9k" ]; then
    testAnnotationName=concepts.txt
elif [ "$testCollection" == "yfcc_rnd_vd" ]; then
    testAnnotationName=concepts.txt
else
    echo "unknown testCollection $testCollection"
    exit
fi


for topk in 5 10 50 100 500
# for topk in 20 40 60 80 100
# for topk in 2 4 6 8 10
do
  for modelName in fastlinear fik50
  do
    python $codepath/model_based/svms/applyConcepts_s.py \
        $testCollection $trainCollection $modelAnnotationName $feature $modelName \
        --prob_output 1 \
        --topk $topk

    tagvotesfile=$rootpath/$testCollection/autotagging/$testCollection/$trainCollection/$modelAnnotationName/$feature,$modelName,$topk,prob/id.tagvotes.txt
    conceptfile=$rootpath/$testCollection/Annotations/$testAnnotationName
    resfile=$SURVEY_DB/"$trainCollection"_"$testCollection"_$vis_feature,tagfeat,$modelName,$nr_pos,$topk.pkl
    python $codepath/postprocess/pickle_tagvotes.py \
        $conceptfile $tagvotesfile $resfile
    # exit
  done
  # exit
done
