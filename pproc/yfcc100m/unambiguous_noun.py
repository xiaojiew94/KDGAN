from nltk.corpus import wordnet

wn_nouns = {synset.name().split('.')[0] for synset in wordnet.all_synsets('n')}
print(len(wn_nouns))