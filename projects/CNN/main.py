from matplotlib import pyplot as plt
from skimage import feature
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten

# tvorba trenovaci mnoziny
inputs = np.zeros((400, 40, 40, 3))
targets = np.zeros((400, 2))
path_positive = "..\\..\\DataSet\\grapes\\TrainingSet\\Positive\\"
path_negative = "..\\..\\DataSet\\grapes\\TrainingSet\\Negative\\"

files1 = os.listdir(path_positive)
files2 = os.listdir(path_negative)
for ind in range(len(files1)):
    str1 = path_positive + files1[ind]
    img = plt.imread(str1)
    inputs[ind, :, :, :] = img
    targets[ind, :] = np.array([1, 0])

for ind in range(len(files2)):
    str1 = path_negative + files2[ind]
    img = plt.imread(str1)
    inputs[ind + len(files1), :, :, :] = img
    targets[ind + len(files1), :] = np.array([0, 1])

model = Sequential()

model.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(40, 40, 3), strides=(1, 1), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())

model.add(Dense(10, activation='tanh'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='rmsprop', loss='mse')
model.summary()

epochs = 1000
batch = 16
val = 0.15
hist = model.fit(x=inputs, y=targets, epochs=epochs, batch_size=batch, validation_split=val, verbose=2)
model.save('model.h5')
