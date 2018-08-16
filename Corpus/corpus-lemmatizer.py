import numpy
from gensim.models.word2vec import LineSentence
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

corpus_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Corpus/wiki/text/AA/wiki_00_processed'
output_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Corpus/wiki/text/AA/wiki_00_processed_lemmatized'

output_file = open(output_file_path, mode='w+', encoding='utf-8')
wiki_corpus = LineSentence(corpus_file_path)
concatenated_temp = []

for corpus in wiki_corpus:
    concatenated_temp.append(corpus)

corpus = [b for a in concatenated_temp for b in a]
corpus_len = len(corpus)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

i = 0
while i < corpus_len:
    new_word = stemmer.stem(corpus[i])
    
    if i == 0:
        output_file.write(new_word)
    else:
        output_file.write(' ' + new_word)

    i = i+1

output_file.close