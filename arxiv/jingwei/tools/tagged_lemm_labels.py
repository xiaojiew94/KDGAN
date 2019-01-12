import os, sys, array, shutil
import numpy as np
import glob

from os import path


# cd yfcc2k & rm -rf tagged,lemm & tagged_lemm_labels.py concepts.txt
# window
# concepts = open('Annotations/' + sys.argv[1]).read().strip().split('\r\n')
# ubuntu
concepts = open('Annotations/' + sys.argv[1]).read().strip().split('\n')
print concepts
outdir = 'tagged,lemm'
if not path.exists(outdir):
	os.mkdir(outdir)

h_files = dict([(C, open(path.join(outdir, C + ".txt"), 'w')) for C in concepts])
cnt = 0
with open('TextData/id.userid.lemmtags.txt') as f:
	for line in f:
		cnt += 1
		if len(line) < 1:
			continue
		id_im,uid,tags = line.split('\t')
		tags = tags.strip().split()
	
		for C in concepts:
			if C in tags:
				h_files[C].write("%s\n" % id_im)
	
for h in h_files:
	h_files[h].close()
	
print "Processed %d lines, %d concepts" % (cnt, len(concepts))
