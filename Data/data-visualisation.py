from sklearn.preprocessing import StandardScaler
from FeatureExtractor import WordsVectorUtil
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib as plt

word_limit = 50
window_size = 5
min_word = 5
feature_size = 300

w2v_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Word2Vec/'
positive_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Data/positive.csv'
negative_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Data/negative.csv'

one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list = WordsVectorUtil.loadDataSet(w2v_file_path, positive_file_path, negative_file_path, word_limit, window_size, min_word, feature_size)

x = StandardScaler().fit_transform(one_dimensional_word_vectors)
pca = PCA(n_components=2)

pcaX = pca.fit_transform(x)

principalXdata = pd.DataFrame(data=pca, columns=['pc1', 'pc2'])
finalDfX = pd.concat([principalXdata, label_list], axis=1)