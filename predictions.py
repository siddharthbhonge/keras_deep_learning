from keras.models import load_model
from keras.utils import np_utils
import matplotlib.pyplot as plt


cnn=load_model('cnn_10epochs_120wt\\8epochs\\175\\w175_cnn.h5')

from scipy import sparse
import numpy as np


zero = sparse.load_npz("raw_data.npz")
labesl = sparse.load_npz("outputs.npz")
#print(zero)
#print(labesl)
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


temp=labels[1300100:1428100]
#     *0.0124





data_augmentation = False
num_classes=2
class_names = ['zero','one']

# Convert and pre-processing

#y_test = np_utils.to_categorical(labels_test, num_classes)
batch_train1 = np.zeros((128, 1, 25, 100))
input_shape=batch_train1.shape[1:]
batch_train = np.zeros((128, 1, 25, 100))
labels_train =np.zeros((128,))
def val_generator():
    j = 1300000
    k = 100
    while True:
        if (j >= (3642104)):
            j = 3000000
        if (k >= (3642104)):
            k = 0

        for i in range(0,128):
             batch_data = data[j:j + 100, 0:25]
             batch_train[i, :, 0:25, :] = np.reshape(batch_data, (1, 25, 100))
             labels_train[i]=labels[j + 100]
             print(j+100)
             j = j + 1

        batch_labels1 = np_utils.to_categorical(labels_train, num_classes)
        yield (batch_train,batch_labels1)

y_pred=cnn.predict_generator(val_generator(),steps=1000,verbose=0)
np.save("pred",y_pred,allow_pickle=True)

t=np.arange(128000)
#y_pred=np.load('pred.npy',allow_pickle=True)
plt.figure(1)
plt.subplot(211)
plt.plot(t,y_pred[:,1])
plt.plot(t,temp,'ro')

plt.subplot(212)
plt.plot(t,temp)
plt.show()




print(np.shape(y_pred))