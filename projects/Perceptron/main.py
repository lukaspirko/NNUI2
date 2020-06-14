from Perceptron import Perceptron
import numpy as np

p1 = Perceptron(2)

vstupy = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]])
cile = np.array([-1, -1, -1, 1])

# p1.hebb(vstupy, cile)

alfa = 0.1
epochs = 100

for n in range(epochs):
    p1.error_train(vstupy, cile, alfa)

print(p1.calc_output(np.array([-1, -1])))
print(p1.calc_output(np.array([-1, 1])))
print(p1.calc_output(np.array([1, -1])))
print(p1.calc_output(np.array([1, 1])))
