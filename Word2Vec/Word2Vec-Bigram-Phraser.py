import os
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence

test_corpus_file = '/test-corpus.txt'
test_corpus = os.path.dirname(os.path.abspath(__file__)) + test_corpus_file

root_path =  os.path.dirname(os.path.abspath(__file__)) + os.sep
corpus_path = os.path.join(os.path.dirname(os.path.dirname(root_path)), 'Corpus' + os.sep + 'wiki' + os.sep + 'text' + os.sep + 'AA' + os.sep  + 'wiki_00')

min_count = 50

saved_file_name = os.path.join(os.path.dirname(os.path.dirname(root_path)), 'Corpus' + os.sep + 'wiki' + os.sep + 'text' + os.sep + 'AA' + os.sep + 'bigram_wiki_AA_00_' + str(min_count))

#words = LineSentence(corpus_path)
#phrases = Phrases(words, min_count=min_count)
#bigram = Phraser(phrases)

#bigram.save(saved_file_name)

bigram = Phraser.load(saved_file_name)

print (bigram['sangat', 'suka'])