# Scenario 1
# Similarity vectors attached with the deep features

import os
import sys
sys.path.insert(0, 'D:/Workspaces/python/Sarcasm Detector Study')
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from FeatureExtractor import WordsVectorUtil
from keras import backend as K
K.set_image_dim_ordering('th')

# init
word_limit = 100
window_size = 5
min_word = 5
feature_size = 100

# path
root_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
w2v_path = os.path.join(os.path.dirname(os.path.dirname(root_path)), 'Word2Vec') + os.sep
data_path = os.path.join(os.path.dirname(os.path.dirname(root_path)), 'Data') + os.sep

# load the data
w2v_file_path = w2v_path
csv_file_path = data_path + 'ToadCSVFile_2018-05-29T18_28_082018-05-29 18-28-45.csv'
one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list = WordsVectorUtil.WordPreprocessing(w2v_file_path, csv_file_path, word_limit, window_size, min_word, feature_size)

# reshape
one_dimensional_word_vectors = one_dimensional_word_vectors.reshape(one_dimensional_word_vectors.shape[0], len(one_dimensional_word_vectors[0]), 1)

# convert to one-hot vectors
labels = np_utils.to_categorical(label_list)

# debug
#print (one_dimensional_similarity_matrix.shape)
#print (two_dimensional_similarity_matrix.shape)
#print (one_dimensional_word_vectors.shape)
#print (two_dimensional_word_vectors.shape)
#print (len(label_list))

def cnn_model (word_count, vector_size):
    # CNN section
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=(500), input_shape=(word_count * vector_size, 1)))
    model.add(LeakyReLU(alpha=0.3))
    #model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=50, kernel_size=(500)))
    model.add(LeakyReLU(alpha=0.3))
    #model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Conv1D(filters=50, kernel_size=(500)))
    model.add(LeakyReLU(alpha=0.3))
    #model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())

    # Dense section
    model.add(Dense(265, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model

model = cnn_model(word_limit, feature_size)
model.summary()

# optimiser
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# compile
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(one_dimensional_word_vectors, labels, epochs=50, batch_size=1, verbose=2)