import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import matplotlib.pyplot as plt
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
#from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras import backend as K
from keras.models import load_model
#if K.backend()=#    K.set_image_dim_ordering("th")

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
from sklearn.model_selection import train_test_split
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
#data=np.zeros((25,data_size))
data=np.reshape(data,(data_size,25))
labels=np.reshape(labels,(data_size,1))

#labels=np.zeros((data_size,1))
#data, labels = np.arange(10).reshape((5, 2)), range(5)
print(np.shape(data))
print(np.shape(labels))

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)

print(np.shape(data_train))
print(np.shape(data_test))
print(np.shape(labels_train))
print(np.shape(labels_test))

data_train=np.reshape(data_train,(25,1600000))
labels_train1=np.asarray(labels_train).flatten()
print(np.shape(labels_train1))


#--------------------------------------------------------------
#for validation data

data_test=np.reshape(data_test,(25,400000))
labels_test=np.reshape(labels_test,(400000,))
labels_test=np.asarray(labels_test).flatten()
batch_test = np.zeros((400000,1,2500))
for i in range (400000-101):
    temp = data_test[0:25, i:i + 100]
    batch_test[i,0, :] = np.reshape(temp, (2500,))

print("-------------------------")
print(np.shape(batch_test))
print(np.shape(labels_test))

#---------------------------------------------------------------------------------------






# Vizualizing model structure

#sequential_model_to_ascii_printout(cnn_n)



from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from keras.callbacks import TensorBoard
from time import time
input_audio=Input(shape=(2500,))
encoded_1 = Dense(100, activation='tanh')(input_audio)
encoded_2 = Dense(300, activation='tanh')(encoded_1)
encoded_3 = Dense(300, activation='tanh')(encoded_2)
decoded_1 = Dense(300, activation='tanh')(encoded_3)
decoded_2 = Dense(100, activation='tanh')(decoded_1)
decoded_3 = Dense(2500, activation='sigmoid')(decoded_2)
ADAM = optimizers.Adam(lr=0.01)
autoencoder = Model(input_audio, decoded_3)
autoencoder.compile(optimizer=ADAM, loss='mse', metrics=['accuracy'])
autoencoder.summary()
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))



batch_train = np.zeros((128,1,2500))


def generator():
    j = 0
    k = 100
    while True:
        if (j >= (4760)):
            j = 0
        if (k >= (4760)):
            k = 0

        for i in range(0, 128):
             batch_data = data_train[0:25, j:j + 100]
             batch_train[i,0,:] = np.reshape(batch_data, (2500,))
             j = j + 1
        #print(np.shape(batch_labels))
        yield (batch_train,batch_train)



#denoiser = autoencoder.fit(generator(),callbacks=[tensorboard])
cnn=autoencoder.fit_generator(generator(), steps_per_epoch= 4800/1, epochs=17,validation_data=(batch_test,y_test), verbose=1)

autoencoder.save('autoencoder.h5')







'''
predict=cnn_n.predict_generator(generator(),steps=data_size/1)

print(np.shape(predict))
np.save("predictions",predict)
'''

# Plots for training and testing process: loss and accuracy
'''
plt.figure(0)
plt.plot(cnn.history['acc'],'r')
plt.plot(cnn.history['val_acc'],'g')
plt.xticks(np.arange(0, 11, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy ")
plt.legend(['train','validation'])

plt.figure(1)
plt.plot(cnn.history['loss'],'r')
plt.plot(cnn.history['val_loss'],'g')
plt.xticks(np.arange(0, 11, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss ")
plt.legend(['train','validation'])
plt.show()
'''
#scores = cnn_n.evaluate(data_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))


# Confusion matrix result

from sklearn.metrics import classification_report, confusion_matrix
Y_pred = cnn_n.predict(batch_test, verbose=0)
#print(Y_pred)

y_pred = np.argmax(Y_pred, axis=1)
#print('sid')

for ix in range(2):
    print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(cm)

print("siddharth")