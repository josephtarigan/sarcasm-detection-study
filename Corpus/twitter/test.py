import datetime
import time
import nltk

#print (time.localtime(1524576894075/1000))
#print (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1524576894075/1000)))

corpus_file_path = 'D:/Workspaces/python/Sarcasm Detector Study/Corpus/wiki/text/AA/wiki_00'
raw = open(corpus_file_path, encoding='utf8').read()
raw = nltk.Text(nltk.word_tokenize(raw))
fdist = nltk.FreqDist(raw)
print (fdist.N())