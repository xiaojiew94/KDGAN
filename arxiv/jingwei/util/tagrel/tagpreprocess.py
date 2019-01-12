
import sys, os 
from optparse import OptionParser
import nltk
import codecs

from basic.common import checkToSkip, makedirsforfile

def stemming(worker, tag):
    return worker.stem(tag)

def lemmatize(worker, tag):
    return worker.lemmatize(tag)


def process(options, tagfile, tpp):
    if "stem" == tpp:
        worker = nltk.PorterStemmer()
        func = stemming
    else:
        worker = nltk.WordNetLemmatizer()
        func = lemmatize

    resultfile = os.path.join(os.path.split(tagfile)[0], 'id.userid.%stags.txt' % tpp)
    if checkToSkip(resultfile, options.overwrite):
        return 0

    makedirsforfile(resultfile)

    fw = codecs.open(resultfile, "w", encoding='utf8')
    parsed = 0
    obtained = 0
    for line in open(tagfile):
        elems = line.strip().split()
        parsed += 1
        if len(elems) > 2:
            newtags = []
            for tag in elems[2:]:
                try:
                    newtag = func(worker,tag.lower())
                except:
                    newtag = tag
                
                newtags.append(newtag.decode('utf-8'))

            newline = "\t".join([elems[0], elems[1], " ".join(newtags)])
            fw.write('%s\n' % newline)
            obtained += 1
    fw.close()
    print ('%d lines parsed, %d records obtained' % (parsed, obtained) )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = OptionParser(usage="""usage: %prog [options] tagfile tpp""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1

    tpp = args[1]
    if tpp not in str.split('stem lemm'):
        print ('tpp has to be either stem or lemm')
        return 1
    return process(options, args[0], tpp)


if __name__ == "__main__":
    sys.exit(main())



