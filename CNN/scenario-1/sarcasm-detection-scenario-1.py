# Scenario 1
# Similarity vectors attached with the deep features

import sys
sys.path.insert(0, 'D:/Workspaces/python/Sarcasm Detector Study')
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from FeatureExtractor import WordsVectorUtil
from keras import backend as K
K.set_image_dim_ordering('th')

# load the data
w2v_file_path = 'D:/Workspaces/python/Sarcasm Detector Study/Word2Vec/'
csv_file_path = 'D:/Workspaces/python/Sarcasm Detector Study/Data/ToadCSVFile_2018-05-29T18_28_082018-05-29 18-28-45.csv'
one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list = WordsVectorUtil.WordPreprocessing(w2v_file_path, csv_file_path, 140, 5, 5, 1000)

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
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=(500)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Conv1D(filters=32, kernel_size=(500)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())

    # Dense section
    model.add(Dense(4203, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    return model

model = cnn_model(140, 1000)
model.summary()

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(one_dimensional_word_vectors, label_list, epochs=10, batch_size=1, verbose=2)