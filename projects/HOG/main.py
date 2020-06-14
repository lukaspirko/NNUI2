from matplotlib import pyplot as plt
from skimage import feature
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# tvorba trenovaci mnoziny
n_features = 729
inputs = np.zeros((400, n_features))
targets = np.zeros((400, 2))
path_positive = "..\\..\\DataSet\\grapes\\TrainingSet\\Positive\\"
path_negative = "..\\..\\DataSet\\grapes\\TrainingSet\\Negative\\"

files1 = os.listdir(path_positive)
files2 = os.listdir(path_negative)
for ind in range(len(files1)):
    str1 = path_positive + files1[ind]
    img = plt.imread(str1)
    inputs[ind, :] = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=False)
    targets[ind, :] = np.array([1, 0])

for ind in range(len(files2)):
    str1 = path_negative + files2[ind]
    img = plt.imread(str1)
    inputs[ind + len(files1), :] = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=False)
    targets[ind + len(files1), :] = np.array([0, 1])

model = Sequential()
model.add(Dense(10, input_dim=729, activation='tanh'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='rmsprop', loss='mse')
model.summary()

epochs = 1000
batch = 16
val = 0.15
hist = model.fit(x=inputs, y=targets, epochs=epochs, batch_size=batch, validation_split=val, verbose=2)
model.save('model.h5')
