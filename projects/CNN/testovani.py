from matplotlib import pyplot as plt
from skimage import feature
import numpy as np
import os

from tensorflow.keras.models import load_model

# tvorba testovaci mnoziny
inputs = np.zeros((400, 40, 40, 3))
targets = np.zeros((400, 2))
path_positive = "..\\..\\DataSet\\grapes\\TestingSet\\Positive\\"
path_negative = "..\\..\\DataSet\\grapes\\TestingSet\\Negative\\"

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

model = load_model('model.h5')
outputs = model.predict(inputs)

print(outputs[5, :])
trues = 0
falses = 0
for ind in range(400):
    if (outputs[ind, 0] > outputs[ind, 1] and ind < 200) or (outputs[ind, 0] < outputs[ind, 1] and ind >= 200):
        trues = trues + 1
    else:
        falses = falses + 1

print(trues)
print(falses)
print(trues / (trues + falses))
