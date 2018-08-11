import sys
sys.path.insert(0, 'D:/Workspaces/python/Sarcasm Detector Study')
from gensim.models import Word2Vec
import pandas as pd
import os
import importlib.util
import numpy as np
from Preprocess import Preprocess

# D:/Workspaces/python/Sarcasm Detector Study/Word2Vec/
# /ToadCSVFile_2018-05-23T20_33_112018-05-23 20-33-49.csv
def WordPreprocessing (w2v_model_path, csv_file_path, word_limit, window_size, word_min_count, feature_size):
    #============================================================================
    # Word2Vec Unit
    #============================================================================
    
    sg = 1
    model_file_path = w2v_model_path + '/w2v-gensim-model' + '-' + str(sg) + '-' + str(window_size) + '-' + str(word_min_count) + '-' + str(feature_size)

    w2vmodel = Word2Vec.load(model_file_path)

    #============================================================================
    # Load Data
    #============================================================================

    csv_file = pd.read_csv(csv_file_path)
    num_rows = csv_file.shape[0]

    # these matrixes are representation of words in the csv files
    # 1D similarity matrix
    one_dimensional_similarity_matrix = np.zeros((num_rows, word_limit*word_limit))

    # 2D similarity matrix
    two_dimensional_similarity_matrix = np.zeros((num_rows, word_limit, word_limit))

    # 1D word vectors
    one_dimensional_word_vectors = np.zeros((num_rows, word_limit * feature_size))

    # 2D word vectors
    two_dimensional_word_vectors = np.zeros((num_rows, word_limit, feature_size))

    # label list
    label_list = []

    for index, row in csv_file.iterrows(): # for every row in the table

        #============================================================================
        # Preprocessing
        #============================================================================

        words = (Preprocess.preprocessPipeline1(row['tweet_text'])) # before trim
        words = list(filter(lambda x: x in w2vmodel.wv.vocab, words)) # after trim
        similarity_vectors = []
        word_vectors = []

        for y in range(0, len(words)):
            # similarity vectors, per row
            for x in range(0, len(words)):
                #print (w2vmodel.wv.similarity(words[y], words[x]))
                similarity_vectors.append(w2vmodel.wv.similarity(words[y], words[x]))

            # word vectors, per row
            word_vectors.append(w2vmodel.wv[words[y]])

        
        # 1D similarity matrix
        one_dimensional_similarity_matrix[index, 0:len(similarity_vectors)] = similarity_vectors

        # 2D similarity matrix
        two_dimensional_similarity_matrix[index] = one_dimensional_similarity_matrix[index].reshape((word_limit, word_limit))

        # 1D word vectors
        # flatten it first
        f_flatten_word_vectors = lambda word_vector : [word for row in word_vector for word in row]
        flatten_word_vectors = f_flatten_word_vectors(word_vectors)
        one_dimensional_word_vectors[index, 0:len(flatten_word_vectors)] = flatten_word_vectors

        # 2D word vectors
        two_dimensional_word_vectors[index] = one_dimensional_word_vectors[index].reshape((word_limit, feature_size))

        # store the label
        label_list.append(row['is_sarcasm'])

    return one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list

# debug
#one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list = WordPreprocessing('D:/Workspace/python/Sarcasm Detector Study/Word2Vec/', 'D:/Workspace/python/Sarcasm Detector Study/Data/ToadCSVFile_2018-05-29T18_28_082018-05-29 18-28-45.csv', 50, 5, 5, 300)


# debug
#print (one_dimensional_similarity_matrix.shape)
#print (two_dimensional_similarity_matrix.shape)
#print (one_dimensional_word_vectors.shape)
#print (two_dimensional_word_vectors.shape)
#print (len(label_list))