#Importing all the required packages
from scipy.io import wavfile
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#fs, data = wavfile.read('sine440.wav')
#fs, data = wavfile.read('Think And Grow Rich (Trimmed Audiobook).wav')
fs, data = wavfile.read('(Full Audiobook)  This Book  Will Change Everything! (Trimmed Audiobook).wav')

print(fs)
print(np.size(data))
#Selecting only 1 channel for further use.
data = data[0:150000,0]
#Number of previous samples based on which the next sample is computed.
N = 100

#Extracting each frame of N samples and respective output sample.
X = []
Y = []
mse_list=[]
input_list=[]
output_list=[]
mse_list_t=[]

#---------------------------------------------------------------------------------------

#normalize the signal
d=1/np.max(np.abs(data))
data=data*d
print("Max is:"+str(d))
#plt.plot(data[0:10000])
#plt.show()


#---------------------------------------------------------------------------------------

#Range is len(data) - (N + 1) so as to deal with peculiarity at the last sample.

for i in range(len(data)-(N + 1)):

    X.append(data[i:i+N])
    Y.append(data[i+(N + 1)])

#Separate out training, testing and validation data using train_test_split.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size = 0.10, random_state = 42)

m_train = len(X_train)
m_labels = len(Y_train)
m_test = len(X_test)

x = np.asarray(X_train,float)
y = np.asarray(Y_train,float)

x_t = np.asarray(X_test)
y_t = np.asarray(Y_test)


print('Number of training examples: m_train = ' + str(m_train)+str(type(x)))
print('Number of training examples: m_train = ' + str(m_labels)+str(type(y)))

print('Number of testing examples: m_test = ' + str(m_test))
print('Number of previous samples considered to predict next one: N = ' + str(N))


#--------------------------------------------------------------------------------------------------


def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

w=np.ones(100,)

for i in range(1,90000):
    input=np.dot(w.T,x[i,])
    error=y[i,]-input
    output=sigmoid(error)

    #print("Actual Value=" + str(y[i,]) + "Value Predicted=" + str(input) + ".Error=" + str(error))

    if(i%100==0):
        mse = 0.5 * ((error) ** 2)
        mse_list.append(mse)
    output_list.append(input)
    dw=0.001*(output*x[i,])
    w= w +  (dw)

for i in range(1,10000):
    pred = np.dot(w.T, x_t[i,])
    error_t = y_t[i,] - pred
    if (i % 100 == 0):
        mse_t = 0.5 * ((error_t) ** 2)
        mse_list_t.append(mse_t)

'''
f, axarr = plt.subplots(2)
f.suptitle('Mean Square Error')

axarr[0].plot(mse_list)
axarr[0].
axarr[1].plot(mse_list_t)
#axarr[2].plot(output_list)
plt.show()
'''

plt.subplot(2, 1, 1)
plt.plot(mse_list)
plt.title('Mean Squared error')
plt.ylabel('Training Set')

plt.subplot(2, 1, 2)
plt.plot(mse_list_t)
plt.ylim(ymax=0.1)
plt.ylim(ymin=0)

plt.xlabel('Samples')
plt.ylabel('Test Set')

plt.show()


