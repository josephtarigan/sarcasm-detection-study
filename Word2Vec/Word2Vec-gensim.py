from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os

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
window = 3
min_count = 5
size = 300

# path
root_path =  os.path.dirname(os.path.abspath(__file__)) + os.sep
corpus_path = os.path.join(os.path.dirname(os.path.dirname(root_path)), 'Corpus' + os.sep + 'wiki' + os.sep + 'text' + os.sep + 'AA' + os.sep  + 'wiki_00_processed')

corpus_file_path = corpus_path
model_file_path = root_path + os.sep + 'w2v-gensim-model' + '-' + str(sg) + '-' + str(window) + '-' + str(min_count) + '-' + str(size)

sentences = LineSentence(corpus_file_path)
w2vmodel = Word2Vec(sentences=sentences, size=size, window=window, min_count=min_count, workers=8, sg=sg)

#sentences = [['hari', 'ini', 'hari', 'senin'], ['besok', 'hari', 'selasa'], ['lusa', 'hari', 'rabu']]
w2vmodel = Word2Vec(sentences=sentences, size=size, window=window, min_count=min_count, workers=8, sg=sg)  
w2vmodel.save(model_file_path)

#w2vmodel = Word2Vec.load(model_file_path)
#print (w2vmodel.wv.most_similar(positive=['dia', 'suka', 'makan']))
print (w2vmodel.wv.similarity('tidak_lulus', 'bodoh'))
print (w2vmodel.wv.similarity('tidak_lulus', 'pintar'))
#print (w2vmodel.wv.similarity('sangat', 'pintar'))
#print (w2vmodel.wv.similarity('lulus', 'tidak'))   
#print (w2vmodel.wv['mereka'].size)

