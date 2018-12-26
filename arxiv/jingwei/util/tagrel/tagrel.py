
import sys, os, time

from os import path
import os 
curdir = path.dirname(path.realpath(__file__))
pardir = path.dirname(curdir)
sys.path.insert(0, pardir)

from common import ROOT_PATH
from util import printStatus
from textstore import RecordStore
from simpleknn import simpleknn


INFO = 'tagrel.tagrel'


class TagrelLearner:
    
    # tpp (tag preprocessing) has to be chosen from {'raw', 'stem', 'lemm'}
    def __init__(self, collection, feature, distance, tpp='lemm', rootpath=ROOT_PATH):
        feat_dir = os.path.join(rootpath, collection, "FeatureData", feature)
        id_file = os.path.join(feat_dir, "id.txt")
        feat_file = os.path.join(feat_dir, "feature.bin")
        nr_of_images, ndims = map(int, open(os.path.join(feat_dir,'shape.txt')).readline().split())

        self.searcher = simpleknn.load_model(feat_file, ndims, nr_of_images, id_file)
        self.searcher.set_distance(distance)

        tagfile = os.path.join(rootpath, collection, "TextData", "id.userid.%stags.txt" % tpp)
        self.textstore = RecordStore(tagfile)
        
        self.nr_neighbors = 1000
        self.nr_newtags = 100

        printStatus(INFO, "nr_neighbors=%d, nr_newtags=%d" % (self.nr_neighbors, self.nr_newtags))
    
    
    def set_nr_neighbors(self, k):
        self.nr_neighbors = k
        printStatus(INFO, "setting nr_neighbors to %d" % k)    
   
    def set_nr_autotags(self, k):
        self.nr_newtags = k
        printStatus(INFO, "setting nr_autotags to %d" % k)    
             
             
    def neighbor_voting(self, neighbors, qry_tags, qry_userid):
        users_voted = set([qry_userid])
        tag2vote = {}
        voted = 0
        unlabeled = 0
        thesameuser = 0
        
        for (name, dist) in neighbors:
            (userid,tags) = self.textstore.lookup(name)
            
            if not tags:    
                unlabeled += 1                
                continue
                
            if userid in users_voted:
                thesameuser += 1
                continue

            tagset = set(tags.split())
            for tag in tagset:
                tag2vote[tag] = tag2vote.get(tag,0) + 1
                
            users_voted.add(userid)
            voted += 1
            if voted >= self.nr_neighbors:
                break
      
        # assert (voted >= self.nr_neighbors), 'unlabeled %d, thesameuser %d, voted %d, neighbors %d' % (unlabeled, thesameuser, voted, len(neighbors))
        
        if not qry_tags: # no tags given, do image auto-tagging
            autotags = []
            for tag,vote in tag2vote.iteritems():
                score = vote - self.textstore.tagprior(tag, self.nr_neighbors)
                if score > 1e-6:
                    autotags.append((tag, score))
            autotags.sort(key=lambda v:(v[1]), reverse=True)
            return autotags[:self.nr_newtags]
        else: # tag relevance learning
            qry_tagset = set(str.split(qry_tags.lower()))
            tagvotes = [(tag, tag2vote.get(tag,0) - self.textstore.tagprior(tag,self.nr_neighbors)) for tag in qry_tagset]
            tagvotes.sort(key=lambda v:(v[1]), reverse=True)
            return tagvotes
        
        
    def estimate(self, qry_vec, qry_tags, qry_id="", qry_userid=""):
        # Step 1. visual neighbor search
        s_time = time.time()
        neighbors = self.searcher.search_knn(qry_vec, max_hits=self.nr_neighbors*3)
        search_time = time.time() - s_time

        # step 2. neighbor voting
        s_time = time.time()
        results = self.neighbor_voting(neighbors, qry_tags, qry_userid=qry_userid)
        voting_time = time.time() - s_time

        #self.echo(self.estimate.__name__, "search %g, voting %g seconds" % (search_time, voting_time))
    
        return results
     


if __name__ == '__main__':
    tagrel = TagrelLearner('train10k','color64', 'l2')


