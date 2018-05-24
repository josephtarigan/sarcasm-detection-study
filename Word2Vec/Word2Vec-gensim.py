from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

'''
sentences

sq
1 = skip-gram
0 = CBOW

size
Dimension of feature vectors, aka the hidden layer size

min_count
ignore word that occurs under the given min_count

window
Sliding window size

'''
sg = 1
window = 5
min_count = 5
size = 1000
corpus_file_path = 'D:/Workspaces/python/Sarcasm Detector Study/Corpus/wiki/text/AA/wiki_00'
model_file_path = 'D:/Workspaces/python/Sarcasm Detector Study/Word2Vec/w2v-gensim-model' + '-' + str(sg) + '-' + str(window) + '-' + str(min_count) + '-' + str(size)

#sentences = LineSentence(corpus_file_path)
#w2vmodel = Word2Vec(sentences=sentences, size=size, window=window, min_count=min_count, workers=8, sg=sg)

#sentences = [['hari', 'ini', 'hari', 'senin'], ['besok', 'hari', 'selasa'], ['lusa', 'hari', 'rabu']]
#w2vmodel = Word2Vec(sentences=sentences, size=size, window=window, min_count=min_count, workers=8, sg=sg)  
#w2vmodel.save(model_file_path)

w2vmodel = Word2Vec.load(model_file_path)
#print (w2vmodel.wv.most_similar(positive=['dia', 'suka', 'makan']))
print (w2vmodel.wv.similarity('suka', 'buang'))
#print (w2vmodel.wv['mereka'].size)