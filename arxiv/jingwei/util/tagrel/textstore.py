from util import printStatus

class RecordStore:
    def __init__(self, tagfile):
        printStatus('textstore.RecordStore', 'read from %s' % tagfile)
        self.mapping = {}
        self.tag2freq = {}

        for line in open(tagfile): #.readlines():
            print line.strip()
            [photoid, userid, tags] = line.strip().split('\t')
            self.mapping[photoid] = (userid, tags.lower())
            for tag in set(str.split(tags)):
                self.tag2freq[tag] = self.tag2freq.get(tag,0) + 1
             
        self.nr_images = len(self.mapping)
        self.nr_tags = len(self.tag2freq)
 
        print ("-> %d images, %d unique tags" % (self.nr_images, self.nr_tags))           
             
 
    def tagprior(self, tag, k):
        return float(k) * self.tag2freq.get(tag,0) / self.nr_images

    def lookup(self, photoid):
        return self.mapping.get(photoid, (None, None))


if __name__ == '__main__':
    tagfile = 'id.userid.lemmtags.txt'
    store = RecordStore(tagfile)

