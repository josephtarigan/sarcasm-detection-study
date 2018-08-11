# Scenario 3
# Similarity matrix is upscaled, attached as the second channel of the input matrix

import sys
sys.path.insert(0, 'D:/Workspace/python/Sarcasm Detector Study')
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import LeakyReLU
from keras.layers import Concatenate
from keras.utils import plot_model
from keras.layers.merge import concatenate
from keras.optimizers import SGD

from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from FeatureExtractor import WordsVectorUtil
from keras import backend as K
K.set_image_dim_ordering('th')

# init
word_limit = 50
window_size = 5
min_word = 5
feature_size = 300
dense_input_size = 256

# load the data
w2v_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Word2Vec/'
csv_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Data/ToadCSVFile_2018-05-29T18_28_082018-05-29 18-28-45.csv'
one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list = WordsVectorUtil.WordPreprocessing(w2v_file_path, csv_file_path, word_limit, window_size, min_word, feature_size)

# interpolate the similarity channel dimension from the original size into the size of input layer dimension

# attach the similarity vector as the second channel of the input

# process as usual