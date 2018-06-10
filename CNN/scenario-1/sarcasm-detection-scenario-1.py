# Scenario 1
# Similarity vectors attached with the deep features

import sys
sys.path.insert(0, 'D:/Workspaces/python/Sarcasm Detector Study')
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
word_limit = 100
window_size = 5
min_word = 5
feature_size = 100
dense_input_size = 256

# load the data
w2v_file_path = 'D:/Workspaces/python/Sarcasm Detector Study/Word2Vec/'
csv_file_path = 'D:/Workspaces/python/Sarcasm Detector Study/Data/ToadCSVFile_2018-05-29T18_28_082018-05-29 18-28-45.csv'
one_dimensional_similarity_matrix, two_dimensional_similarity_matrix, one_dimensional_word_vectors, two_dimensional_word_vectors, label_list = WordsVectorUtil.WordPreprocessing(w2v_file_path, csv_file_path, word_limit, window_size, min_word, feature_size)

# reshape
one_dimensional_word_vectors = one_dimensional_word_vectors.reshape(one_dimensional_word_vectors.shape[0], len(one_dimensional_word_vectors[0]), 1)
#one_dimensional_similarity_matrix = one_dimensional_similarity_matrix.reshape(one_dimensional_similarity_matrix.shape[0], one_dimensional_similarity_matrix.shape[1], 1)

# convert to one-hot vectors
labels = np_utils.to_categorical(label_list)

# debug
#print (one_dimensional_similarity_matrix.shape)
#print (two_dimensional_similarity_matrix.shape)
#print (one_dimensional_word_vectors.shape)
#print (two_dimensional_word_vectors.shape)
#print (labels.shape)
#print (len(label_list))
#print(type(one_dimensional_similarity_matrix))
#print(type(one_dimensional_word_vectors))

# similarity matrix
# will produce N dimensional similarity vectors in 1 dimensional array
def augmented_input (similarity_matrix) :
    augmented_input = Input(shape=(similarity_matrix.shape[1],), name='augmented_input')
    return augmented_input

# the input layer
def input_layer (word_limit, feature_size) :
    input_layer = Input(shape=(word_limit*feature_size, 1), name='main_input')

    return input_layer

# the cnn builder
def conv_layer (input_layer, filter_count, kernel_size, pool_size, word_limit, feature_size, i) :
    conv_layer = Conv1D(filters=filter_count, kernel_size=(kernel_size), input_shape=(word_limit * feature_size, 1), activation='relu', name='conv_' + str(i)) (input_layer)
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

# build the model
input_layer = input_layer(word_limit, feature_size)
conv_layer1 = conv_layer(input_layer, 32, 500, 2, word_limit, feature_size, 1)
conv_layer2 = conv_layer(conv_layer1, 50, 500, 4, word_limit, feature_size, 2)
conv_layer3 = conv_layer(conv_layer2, 50, 500, 4, word_limit, feature_size, 3)
flatten_layer = flatten_layer(conv_layer3)

# augmented input
augmented_input_layer = augmented_input(one_dimensional_similarity_matrix)

# concatenate
concat_layer = concatenate([flatten_layer, augmented_input_layer])

# the dense layer
dense_layer = dense_layer(dense_input_size, concat_layer, len(labels[1]))

# the main model
model = Model(inputs=[input_layer, augmented_input_layer], outputs=dense_layer)

# summary
#print (model.summary())

# plot graph
#plot_model(model, to_file='D:/Workspaces/python/Sarcasm Detector Study/multilayer_perceptron_graph.png')

# SGD model
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# train the model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train the model
history = model.fit({'main_input' : one_dimensional_word_vectors, 'augmented_input' : one_dimensional_similarity_matrix}, {'output_layer' : labels}, epochs=50, batch_size=1, verbose=2)