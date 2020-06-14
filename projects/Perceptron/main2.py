from Perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt

p1 = Perceptron(2)

inputs = np.genfromtxt('..\\..\\DataSet\\chipsy\\inputs.csv', delimiter=';')  # nacteni vstupu trenovaci mnoziny
targets = np.genfromtxt('..\\..\\DataSet\\chipsy\\targets.csv', delimiter=';')  # nacteni cilu

inputs_tren = inputs[:, 0:140]
targets_tren = targets[0:140]

inputs_val = inputs[:, 141:170]
targets_val = targets[141:170]

alfa = 0.001

epochs = 100
E = np.zeros(epochs)

for n in range(epochs):
    E[n] = p1.evaluate_error(inputs_val, targets_val)
    p1.error_train(inputs_tren, targets_tren, alfa)

plt.plot(E)
plt.ylabel('E')
plt.xlabel('Epochs')
plt.show()
