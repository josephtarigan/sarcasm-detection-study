import numpy
from gensim.models.word2vec import LineSentence

corpus_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Corpus/wiki/text/AA/wiki_00'
output_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Corpus/wiki/text/AA/wiki_00_processed'
affix_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Corpus/affix-list.txt'

affix_file = open(affix_file_path, mode='r', encoding='utf-8')
output_file = open(output_file_path, mode='w+', encoding='utf-8')
wiki_corpus = LineSentence(corpus_file_path)
concatenated_temp = []
affixes = [line.rstrip('\n') for line in affix_file]

for corpus in wiki_corpus:
    concatenated_temp.append(corpus)

corpus = [b for a in concatenated_temp for b in a]
corpus_len = len(corpus)

new_word = ''

i = 0

while i < corpus_len:
    if i+1 < corpus_len:
        if corpus[i] in affixes:
            new_word = corpus[i] + '_' + corpus[i+1]
            i = i+2
        else:
            new_word = corpus[i]
            i = i+1
    else:
        new_word = corpus[i]
        i = i+1
    
    output_file.write(new_word)
    if i < corpus_len:
        output_file.write(' ')

output_file.close
affix_file.close