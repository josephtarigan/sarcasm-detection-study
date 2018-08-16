import sys
sys.path.insert(0, 'D:/Workspaces/python/Sarcasm Detector Study')
from gensim.models import Word2Vec
import pandas as pd
import os
import importlib.util
import numpy as np
from Preprocess import Preprocess
import math
from sklearn.metrics.pairwise import cosine_similarity

def smoteSampling (one_dimensional_word_vectors, one_dimensional_similarity_matrix, sampling_target, word_limit, feature_size, label) :
    k = 5
    sample_size = one_dimensional_similarity_matrix.shape[0]
    concatenatedVector = []
    sampleKNearestNeighbour = []
    synteticSample = []
    word_vector_length = one_dimensional_word_vectors.shape[1]
    similarity_vector_length = one_dimensional_similarity_matrix.shape[1]
    if label == 1 :
        labels = np.ones(sampling_target)
    else :
        labels = np.zeros(sampling_target)

    # concatenate word vectors and similarity vector
    for index, _ in enumerate(one_dimensional_similarity_matrix) :
        concatenatedVector.append(np.concatenate((one_dimensional_word_vectors[index], one_dimensional_similarity_matrix[index])))
    concatenatedVector = np.asarray(concatenatedVector)

    # find k-nearest neighbourhood
    for index, _ in enumerate (one_dimensional_similarity_matrix) :
        popedIndex = [i for i in range(0, sample_size)]
        popedIndex.pop(index)

        nearestNeighbourhood = np.argsort(cosine_similarity(concatenatedVector[[index]], Y=concatenatedVector[popedIndex]))
        nearestNeighbourhood = np.fliplr(nearestNeighbourhood)
        nearestNeighbourhood = nearestNeighbourhood[0][:k]

        sampleKNearestNeighbour.insert(index, nearestNeighbourhood)

    # reset
    odwv = np.ndarray((sampling_target, word_vector_length))
    odsm = np.ndarray((sampling_target, similarity_vector_length))
    tdsm = np.ndarray((sampling_target, word_limit, word_limit))
    tdwv = np.ndarray((sampling_target, word_limit, feature_size))

    for i, _ in enumerate(one_dimensional_word_vectors):
        odwv[i] = one_dimensional_word_vectors[i]
        odsm[i] = one_dimensional_similarity_matrix[i]

    # delta
    delta = sampling_target - sample_size

    # for each sample
    for n in range(sample_size, sample_size+delta) :
        # generate random number
        random_neighbour = np.random.random_integers(0, high=k-1)
        random_sample = np.random.random_integers(0, high=sample_size-1)
        
        # get sample data
        sample_data = concatenatedVector[random_sample]

        # get nearest neighbour data of the sample data
        nearestNeighbourhood_data = concatenatedVector[sampleKNearestNeighbour[random_sample][random_neighbour]]

        # calculate diff
        diff = np.subtract(nearestNeighbourhood_data, sample_data)

        # calculate gap
        gap = np.random.random()

        # new data
        synteticSample.append(sample_data + (diff*gap))

        # split it back
        synteticVector = np.asarray(sample_data + (diff*gap))
        odwv[n, 0:word_vector_length] = synteticVector[0:word_vector_length]
        odsm[n, 0:similarity_vector_length] = synteticVector[word_vector_length:synteticVector.shape[0]]

        # 2D similarity matrix
        tdsm[n] = odsm[n].reshape((word_limit, word_limit))
        # 2D word vectors
        tdwv[n] = odwv[n].reshape((word_limit, feature_size))

    return odsm, odwv, tdsm, tdwv, labels

def loadDataSet (w2v_file_path, positive_csv_file_path, negative_csv_file_path, word_limit, window_size, min_word, feature_size) :

    # calculate positive sample only
    positive_one_dimensional_similarity_matrix, positive_two_dimensional_similarity_matrix, positive_one_dimensional_word_vectors, positive_two_dimensional_word_vectors, positive_label_list = WordPreprocessing(w2v_file_path, positive_csv_file_path, word_limit, window_size, min_word, feature_size)
    
    # calculate negative sample only
    # the holder will be the union holder
    one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list = WordPreprocessing(w2v_file_path, negative_csv_file_path, word_limit, window_size, min_word, feature_size)

    # target
    target_size = 20
    #whole_size = target_size + len(label_list)

    # SMOTE the positive sample
    positive_one_dimensional_similarity_matrix, positive_one_dimensional_word_vectors, positive_two_dimensional_similarity_matrix, positive_two_dimensional_word_vectors, positive_label_list = smoteSampling(positive_one_dimensional_word_vectors, positive_one_dimensional_similarity_matrix, target_size, word_limit, feature_size, 1)

    # put into one array
    one_dimensional_similarity_matrix = np.append(one_dimensional_similarity_matrix, positive_one_dimensional_similarity_matrix, axis=0)
    one_dimensional_word_vectors = np.append(one_dimensional_word_vectors, positive_one_dimensional_word_vectors, axis=0)
    two_dimensional_similarity_matrix = np.append(two_dimensional_similarity_matrix, positive_two_dimensional_similarity_matrix, axis=0)
    two_dimensional_word_vectors = np.append(two_dimensional_word_vectors, positive_two_dimensional_word_vectors, axis=0)
    for i in range(0, target_size):
        label_list.append(positive_label_list[i])

    # return
    return one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list

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
#one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list = WordPreprocessing('D:/Workspace/python/Sarcasm Detector Study/Word2Vec/', 'D:/Workspace/python/Sarcasm Detector Study/Data/positive.csv', 50, 5, 5, 300)


# debug
#print (one_dimensional_similarity_matrix.shape)
#print (two_dimensional_similarity_matrix.shape)
#print (one_dimensional_word_vectors.shape)
#print (two_dimensional_word_vectors.shape)
#print (len(label_list))

#a, aa, b, bb, l = smoteSampling (one_dimensional_word_vectors, one_dimensional_similarity_matrix, 4000, 50, 300, 1)

#print (a.shape)
#print (aa.shape)
#print (b.shape)
#print (bb.shape)

#one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list = loadDataSet()