# Scenario 2
# Similarity vectors attached to each word vectors

import sys
import time
import numpy as np
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
from keras.callbacks import CSVLogger
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
odsm = None
tdsm = None
odwv = None
tdwv = None
data_labels = None
log_file = 'D:/Workspace/python/Sarcasm Detector Study/CNN/Log/scenario2_' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
model_path = 'D:/Workspace/python/Sarcasm Detector Study/CNN/Model/scenario2_' + time.strftime("%Y%m%d-%H%M%S") + '.mdl'

# load the data
w2v_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Word2Vec/'
positive_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Data/positive.csv'
negative_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Data/negative.csv'

one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list = WordsVectorUtil.loadDataSet(w2v_file_path, positive_file_path, negative_file_path, word_limit, window_size, min_word, feature_size)

# randomise the order
# first, generate random non-repetitive index
odsm = np.empty_like(one_dimensional_similarity_matrix)
tdsm = np.empty_like(two_dimensional_similarity_matrix)
odwv = np.empty((one_dimensional_similarity_matrix.shape[0], one_dimensional_word_vectors.shape[1] + one_dimensional_similarity_matrix.shape[1]))
tdwv = np.empty_like(two_dimensional_word_vectors)
data_labels = []
random_index = np.random.choice(one_dimensional_word_vectors.shape[0], one_dimensional_word_vectors.shape[0], replace=False)

# then build the new array for each input
# concat similarity vector with input
for i, j in enumerate(random_index):
    odsm[i] = one_dimensional_similarity_matrix[j]
    tdsm[i] = two_dimensional_similarity_matrix[j]
    odwv[i] = np.append(one_dimensional_word_vectors[j], one_dimensional_similarity_matrix[j])
    tdwv[i] = two_dimensional_word_vectors[j]
    data_labels.append(label_list[j])

# convert to one-hot vectors
labels = np_utils.to_categorical(data_labels)

# reshape
odwv = odwv.reshape(odwv.shape[0], len(odwv[0]), 1)

# the input layer
def input_layer (word_limit, feature_size) :
    input_layer = Input(shape=((word_limit*feature_size) + (word_limit*word_limit), 1), name='main_input')

    return input_layer

# the cnn builder
def conv_layer (input_layer, filter_count, kernel_size, pool_size, word_limit, feature_size, i) :
    if i == 1 :
        conv_layer = Conv1D(filters=filter_count, kernel_size=(kernel_size), input_shape=((word_limit*feature_size) + (word_limit*word_limit), 1), activation='relu', name='conv_' + str(i)) (input_layer)
    else :
        conv_layer = Conv1D(filters=filter_count, kernel_size=(kernel_size), activation='relu', name='conv_' + str(i)) (input_layer)
    act_layer = LeakyReLU(alpha=0.3) (conv_layer)
    dropout_layer = Dropout(0.2, name='dropout_' + str(i)) (act_layer)
    pooling_layer = MaxPooling1D(pool_size=pool_size, name='pool_' + str(i)) (dropout_layer)

    return pooling_layer

# flatten util
def flatten_layer (conv_layer) :
    flatten_layer = conv_layer
    flatten_layer = Flatten(name='flatten_layer') (flatten_layer)

    return flatten_layer

# the dense builder
def dense_layer (dense_input_size, flatten_input, label_size) :
    dense_layer1 = Dense(dense_input_size, activation='relu', name='dense_1') (flatten_input)
    dense_layer2 = Dense(label_size, activation='softmax', name='output_layer') (dense_layer1)

    return dense_layer2

# concat input vector with similarity vector
def concat_input_similarity_vector (input_vectors, similarity_vectors) :
    for index, _ in enumerate(input_vectors) :
        input_vectors[index].concatenate(similarity_vectors[index])

# build the model
input_layer = input_layer(word_limit, feature_size)
conv_layer1 = conv_layer(input_layer, 300, 11, 2, word_limit, feature_size, 1)
conv_layer2 = conv_layer(conv_layer1, 200, 11, 2, word_limit, feature_size, 2)
conv_layer3 = conv_layer(conv_layer2, 100, 10, 2, word_limit, feature_size, 3)
conv_layer4 = conv_layer(conv_layer3, 100, 10, 2, word_limit, feature_size, 4)
flatten_layer = flatten_layer(conv_layer4)
dense_layer = dense_layer(dense_input_size, flatten_layer, len(labels[1]))

# the main model
model = Model(inputs=[input_layer], outputs=dense_layer)

# summary
print (model.summary())

# plot graph
plot_model(model, to_file='D:/Workspace/python/Sarcasm Detector Study/scenario_2_model.png')

# SGD model
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# set log file
csv_logger = CSVLogger(log_file, append=False, separator=';')

# train the model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train the model
history = model.fit({'main_input' : odwv}, {'output_layer' : labels}, epochs=50, batch_size=1, verbose=2, callbacks=[csv_logger])

# save model
model.save(model_path)