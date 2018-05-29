# Scenario 1
# Similarity vectors attached with the deep features

import sys
sys.path.insert(0, 'D:/Workspaces/python/Sarcasm Detector Study')
from Feature-Extractor import words-vectorizer

one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list = word_preprocessing('D:/Workspaces/python/Sarcasm Detector Study/Word2Vec/', 'D:/Workspaces/python/Sarcasm Detector Study/Feature-Extractor/ToadCSVFile_2018-05-23T20_33_112018-05-23 20-33-49.csv', 140, 5, 5, 1000)

# debug
#print (one_dimensional_similarity_matrix.shape)
#print (two_dimensional_similarity_matrix.shape)
#print (one_dimensional_word_vectors.shape)
#print (two_dimensional_word_vectors.shape)
#print (len(label_list))