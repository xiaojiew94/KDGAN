# ./do_tagvote.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 trainCollection testCollection feature"
    exit
fi

trainCollection=$1
testCollection=$2
feature=$3
tagger=tagvote


if [ "$feature" = "color64+dsift" ]; then
    distance=l1
elif [ "$feature" = "vgg-verydeep-16-fc7relu" ]; then 
    distance=cosine
else
    echo "unknown feature $feature"
    exit
fi 


if [ "$testCollection" == "flickr81" ]; then
    testAnnotationName=concepts81.txt
elif [ "$testCollection" == "flickr51" ]; then
    testAnnotationName=concepts51ms.txt
elif [ "$testCollection" == "mirflickr08" ]; then
    testAnnotationName=conceptsmir14.txt
elif [ "$testCollection" == "yfcc2k" ]; then
    testAnnotationName=concepts.txt
elif [ "$testCollection" == "yfcc0k" ]; then
  testAnnotationName=concepts.txt
elif [ "$testCollection" == "yfcc_rnd_vd" ]; then
  testAnnotationName=concepts.txt
else
    echo "unknown testCollection $testCollection"
    exit
fi

for k in 1 2 4 8 16
do
    annotationName=concepts.txt
    python $codepath/instance_based/apply_tagger.py $testCollection $trainCollection $annotationName $feature --tagger $tagger --distance $distance --k $k
    tagvotesfile=$rootpath/$testCollection/autotagging/$testCollection/$trainCollection/$annotationName/$tagger/$feature,"$distance"knn,$k/id.tagvotes.txt

    if [ ! -f "$tagvotesfile" ]; then
        echo "tagvotes file $tagvotesfile does not exist!"
        exit
    fi

    conceptfile=$rootpath/$testCollection/Annotations/$testAnnotationName
    resultfile=$SURVEY_DB/"$trainCollection"_"$testCollection"_$feature,tagvote,$k.pkl
    python $codepath/postprocess/pickle_tagvotes.py $conceptfile $tagvotesfile $resultfile
done