from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.utils import np_utils
from keras import regularizers

'''
x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).
'''
(x_train, y_train),(x_test, y_test) = mnist.load_data()
#print (x_train.shape)
#print (y_train[0])

# serialise the input images
# using numpy reshape method
# 60.000, 28*28
# convert to float32
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

# normalize the output to {0,1}
x_train = x_train / 255
x_test = x_test /  255

# create one hot vector for the label
# use keras np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
#print (num_classes)

# stats
count = 0
loss = 9999
target_loss = 0.0001

def baseline_model():
    # the model
    model = Sequential()
    model.add(Dense(1024, kernel_initializer='normal', input_dim=num_pixels))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, kernel_initializer='normal'))
    model.add(Activation('softmax'))
    model.summary()

    #compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = baseline_model()

while (loss > target_loss):
    history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=200, verbose=2, epochs=1) #epoch bla bla bla came from this line

    #print (history.history)
    count += 1
    loss = history.history['loss'][0]
    print ('\n\n============\nLoss : %.5f\nEpoch : %d\n============\n\n' % (loss, count))

scores = model.evaluate(x_test, y_test, verbose=2)
print ('Baseline error: %.2f%%' % (100-scores[1]*100))
print ('Finished!')