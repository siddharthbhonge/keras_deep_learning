import os

import time

import numpy as np
import scipy.sparse as sparse
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import random
# Import Tensorflow with multiprocessing for use 16 cores on plon.io

if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")




import tensorflow as tf
import multiprocessing as mp

core_num = mp.cpu_count()
print(core_num)
config = tf.ConfigProto(
    inter_op_parallelism_threads=core_num,
    intra_op_parallelism_threads=core_num)
sess = tf.Session(config=config)

#--------------------------------------------------------------------------------

from scipy import sparse
import numpy as np


zero = sparse.load_npz("raw_data.npz")
labesl = sparse.load_npz("outputs.npz")
print(zero)
print(labesl)
print(np.shape(labesl))
print(np.shape(zero))
print(sparse.csr_matrix.count_nonzero(labesl))
print(sparse.csr_matrix.count_nonzero(zero))

print("=========================================")




data = sparse.csr_matrix.todense(zero)
labels = sparse.csr_matrix.todense(labesl)

data_size=2000000
split=0.2
data=np.reshape(data,(data_size,25))
labels=np.reshape(labels,(data_size,1))
print(np.shape(data))
print(np.shape(labels))


test_index = []
testlength = int(200 * 0.2)
test_index = random.sample(range(1, data_size), int(data_size*split))

data_test = []
labels_test1 = []
j = 0
'''
for i in range(0, data_size-101):

    data_test.append(data[i:i + 100, 0:25])
    labels_test1.append(labels[i + 100])

print(np.shape(data_test))
print(np.shape(np.reshape(labels_test1, (int(data_size*split), ))))

batch_test=np.asarray(data_test)
batch_test=np.reshape(batch_test,(int(data_size*split),25,100))
labels_test=np.asarray(labels_test1)
data_test=None
labels_test1=None
'''






data_augmentation = False
num_classes=2
class_names = ['zero','one']

# Convert and pre-processing

#y_test = np_utils.to_categorical(labels_test, num_classes)
batch_train1 = np.zeros((128, 1, 25, 100))
input_shape=batch_train1.shape[1:]
class_weight = {0: 1.,
                1: 95.00}
def base_model():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = SGD(lr = 0.01, decay=1e-6, nesterov=True)

# Train model

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
cnn_n = base_model()
cnn_n.summary()




batch_train = np.zeros((128, 1, 25, 100))
labels_train =np.zeros((128,))


def generator():
    j = 0
    k = 100
    while True:
        if (j >= (1399770)):
            j = 0
        if (k >= (1399770)):
            k = 0

        for i in range(0,128):
             batch_data = data[j:j + 100, 0:25]
             batch_train[i, :, :, :] = np.reshape(batch_data, (1, 25, 100))
             labels_train[i]=labels[j + 100]
             j = j + 1

        batch_labels1 = np_utils.to_categorical(labels_train, num_classes)
        yield (batch_train,batch_labels1)

def val_generator():
    j = 1400000
    k = 100
    while True:
        if (j >= (1599770)):
            j = 0
        if (k >= (1599770)):
            k = 0

        for i in range(0,128):
             batch_data = data[j:j + 100, 0:25]
             batch_train[i, :, :, :] = np.reshape(batch_data, (1, 25, 100))
             labels_train[i]=labels[j + 100]
             j = j + 1

        batch_labels1 = np_utils.to_categorical(labels_train, num_classes)
        yield (batch_train,batch_labels1)


cnn=cnn_n.fit_generator(generator(), steps_per_epoch= 1400000/128,validation_data=val_generator(),validation_steps=1562,class_weight=class_weight, epochs=8, verbose=1)
cnn_n.save('w95_cnn.h5')


np.save("wcnn95_loss",np.asarray(cnn.history['loss']),allow_pickle=True)
np.save("wcnn95_acc",np.asarray(cnn.history['acc']),allow_pickle=True)
np.save("wcnn95_val_loss",np.asarray(cnn.history['val_loss']),allow_pickle=True)
np.save("wcnn95_val_acc",np.asarray(cnn.history['val_acc']),allow_pickle=True)
print("siddharth")
