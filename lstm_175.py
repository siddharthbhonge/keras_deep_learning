import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

# Import Tensorflow with multiprocessing for use 16 cores on plon.io
from keras import backend as K
from keras.models import load_model
import random
from keras.utils import np_utils
from keras.optimizers import SGD
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

# Import Tensorflow with multiprocessing for use 16 cores on plon.io
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
# Convert and pre-processing

#y_train = np_utils.to_categorical(labels_train1, num_classes)
#y_test = np_utils.to_categorical(labels_test, num_classes)
#input_shape=[32,1,25,100]
class_weight = {0: 1.,
                1: 175.00}

def base_model():
    model = Sequential()
    model.add(LSTM(8,input_shape=(25,100),return_sequences=False))
    model.add(Dense(8,kernel_initializer='normal',activation='relu'))
    model.add(Dense(2,kernel_initializer='normal',activation='softmax'))
    #model.compile(loss='mse',optimizer ='adam',metrics=['accuracy'])
    sgd = SGD(lr=0.01, decay=1e-6, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

lstm_n = base_model()
lstm_n.summary()

# Vizualizing model structure


batch_train = np.zeros((128, 25, 100))
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
             batch_train[i, :, :] = np.reshape(batch_data, (25, 100))
             labels_train[i]=labels[j + 100]
             j = j + 1
             batch_labels1 = np_utils.to_categorical(labels_train, 2)

        yield (batch_train,batch_labels1)

def val_generator():
    j = 1400000
    k = 100
    while True:
        if (j >= (1599770)):
            j = 1400000
        if (k >= (1599770)):
            k = 0

        for i in range(0,128):
             batch_data = data[j:j + 100, 0:25]
             batch_train[i, :, :] = np.reshape(batch_data, (25, 100))
             labels_train[i]=labels[j + 100]
             j = j + 1
             batch_labels1 = np_utils.to_categorical(labels_train, 2)

        yield (batch_train,batch_labels1)




#sid=generator()
lstm=lstm_n.fit_generator(generator(), steps_per_epoch= 1400000/128,validation_data=val_generator(),class_weight=class_weight,validation_steps=1562, epochs=25, verbose=1)

lstm_n.save('lstm175.h5')
np.save("lstm175_loss",np.asarray(lstm.history['loss']),allow_pickle=True)
np.save("lstm175_acc",np.asarray(lstm.history['acc']),allow_pickle=True)
np.save("lstm175_val_loss",np.asarray(lstm.history['val_loss']),allow_pickle=True)
np.save("lstm175_val_acc",np.asarray(lstm.history['val_acc']),allow_pickle=True)
print("siddharth")