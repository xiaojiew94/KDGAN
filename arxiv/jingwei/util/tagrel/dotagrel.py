import sys, os, time

from optparse import OptionParser

from util import checkToSkip,niceNumber,printStatus,makedirsforfile,readImageSet
from common import ROOT_PATH
from tagrel import TagrelLearner
from textstore import RecordStore
from simpleknn.bigfile import BigFile


INFO = __file__


def process(options, trainCollection, feature, testCollection):
    rootpath = options.rootpath
    tpp = options.tpp
    distance = options.distance
    k = options.k
    r = options.r
    donefile = options.donefile
    overwrite = options.overwrite
    numjobs = options.numjobs
    job = options.job
    blocksize = options.blocksize

    if options.testset is None:
        testset = testCollection
    
    test_tag_file = os.path.join(rootpath, testCollection, "TextData", "id.userid.%stags.txt"%tpp)
    try:
        testStore = RecordStore(test_tag_file)
        resultName = "tagrel"
    except:
        testStore = None
        printStatus(INFO, "Failed to load %s, will do image auto-tagging" % test_tag_file)
        resultName = "autotagging"

    nnName = distance + "knn"
    resultfile = os.path.join(rootpath, testCollection,resultName,testset,trainCollection,"%s,%s,%d,%s" % (feature,nnName,k,tpp), "id.tagvotes.txt")
    
    if numjobs>1:
        resultfile += ".%d.%d" % (numjobs,job)

    if checkToSkip(resultfile, overwrite):
        return 0

 
    if donefile:
        doneset = set([x.split()[0] for x in open(donefile).readlines()[:-1]])
    else:
        doneset = set()
    printStatus(INFO, "%d images have been done already, and they will be ignored" % len(doneset))
        
    test_imset = readImageSet(testCollection, testset, rootpath)
    test_imset = [x for x in test_imset if x not in doneset]
    test_imset = [test_imset[i] for i in range(len(test_imset)) if (i%numjobs+1) == job]
    test_feat_dir = os.path.join(rootpath, testCollection, 'FeatureData', feature)
    test_feat_file = BigFile(test_feat_dir)

   
    learner = TagrelLearner(trainCollection, feature, distance, tpp=tpp, rootpath=rootpath)
    learner.set_nr_neighbors(k)
    learner.set_nr_autotags(r)
    
    printStatus(INFO, "working on %d-%d, %d test images -> %s" % (numjobs,job,len(test_imset),resultfile))
 
    done = 0
    makedirsforfile(resultfile)
    
    fw = open(resultfile, "w")

    read_time = 0
    test_time = 0
    start = 0

    while start < len(test_imset):
        end = min(len(test_imset), start + blocksize)
        printStatus(INFO, 'processing images from %d to %d' % (start, end-1))

        s_time = time.time()
        renamed, vectors = test_feat_file.read(test_imset[start:end])
        read_time += time.time() - s_time
        nr_images = len(renamed)
        #assert(len(test_imset[start:end]) == nr_images) # some images may have no visual features available

        s_time = time.time()
        output = [None] * nr_images
        for i in range(nr_images):
            if testStore:
                (qry_userid, qry_tags) = testStore.lookup(renamed[i])
            else:
                qry_userid = None
                qry_tags = None

            tagvotes = learner.estimate(vectors[i], qry_tags, qry_userid)
            output[i] = '%s %s\n' % (renamed[i], " ".join(["%s %s" % (tag, niceNumber(vote,8)) for (tag,vote) in tagvotes]))
        test_time += time.time() - s_time
        start = end
        fw.write(''.join(output))
        fw.flush()

        done += len(output)

    # done    
    printStatus(INFO, "%d done. read time %g seconds, test_time %g seconds" % (done, read_time, test_time))
    fw.close()
    return 1


    
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = OptionParser(usage="""usage: %prog [options] trainCollection feature testCollection""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--donefile", default=None, type="string", help="to ignore images that have been done")
    parser.add_option("--k", default=1000, type="int", help="number of neighbors")
    parser.add_option("--r", default=20, type="int", help="number of tags to be predicted for image auto-tagging")
    parser.add_option("--tpp", default="lemm", type="string", help="tag preprocess, can be raw, stem, or lemm")
    parser.add_option("--distance", default="l2", type="string", help="visual distance, can be l1 or l2")
    parser.add_option("--testset", default=None, type="string", help="do tagrel only for a subset of the given test collection")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath where the train and test collections are stored")
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs")
    parser.add_option("--job", default=1, type="int", help="current job")
    parser.add_option("--blocksize", default=500, type="int", help="nr of feature vectors loaded per time")
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    
    assert(options.job>=1 and options.numjobs >= options.job)
    return process(options, args[0], args[1], args[2])

if __name__ == "__main__":
    sys.exit(main())


