import sys
sys.path.insert(0, 'D:/Workspaces/python/Sarcasm Detector Study')
from gensim.models import Word2Vec
import pandas as pd
import os
import importlib.util
import numpy as np
from Preprocess import Preprocess

#============================================================================
# Load Data
#============================================================================

csv_file = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/ToadCSVFile_2018-05-23T20_33_112018-05-23 20-33-49.csv')
similarity_matrix = np.zeros((csv_file.size, 144*csv_file.size))
#print (csv_file.head())

print (similarity_matrix.shape)

#============================================================================
# Word2Vec Unit
#============================================================================
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

w2vmodel = Word2Vec.load(model_file_path)

for index, row in csv_file.iterrows():

    #============================================================================
    # Preprocessing
    #============================================================================

    words = (Preprocess.preprocessPipeline1(row['tweet_text'])) # before trim
    words = list(filter(lambda x: x in w2vmodel.wv.vocab, words)) # after trim
    vectors = []

    for y in range(0, len(words)):
        for x in range(0, len(words)):
            #print (w2vmodel.wv.similarity(words[y], words[x]))
            vectors.append(w2vmodel.wv.similarity(words[y], words[x]))

    # add to similarity matrix
    similarity_matrix[index, 0:len(vectors)] = vectors

print (similarity_matrix)