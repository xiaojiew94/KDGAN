import os, time

from common import ROOT_PATH

def printMessage(message_type, trace, message):
    print ('%s %s [%s] %s' % (time.strftime('%d/%m/%Y %H:%M:%S'), message_type, trace, message))

def printStatus(trace, message):
    printMessage('INFO', trace, message)

def printError(trace, message):
    printMessage('ERROR', trace, message)


def checkToSkip(filename, overwrite):
    if os.path.exists(filename):
        print ("%s exists." % filename),
        if overwrite:
            print ("overwrite")
            return 0
        else:
            print ("skip")
            return 1
    return 0

def makedirsforfile(filename):
    try:
        os.makedirs(os.path.split(filename)[0])
    except:
        pass


def niceNumber(v, maxdigit=6):
    """Nicely format a number, with a maximum of 6 digits."""
    assert(maxdigit >= 0)

    if maxdigit == 0:
        return "%.0f" % v

    fmt = '%%.%df' % maxdigit
    s = fmt % v

    if len(s) > maxdigit:
        return s.rstrip("0").rstrip(".")
    elif len(s) == 0:
        return "0"
    else:
        return s


def readImageSet(collection, dataset=None, rootpath=ROOT_PATH):
    if not dataset:
        dataset = collection
    imsetfile = os.path.join(rootpath, collection, 'ImageSets', '%s.txt' % dataset)
    imset = map(str.strip, open(imsetfile).readlines())
    return imset
