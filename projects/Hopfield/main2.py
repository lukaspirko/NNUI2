import numpy as np
from zobrazeni import show
from zobrazeni import noise
from Hopfield import Hopfield

vzor1 = np.genfromtxt('..\\..\\Patterns\\bpm\\a.csv', delimiter=';')
vzor2 = np.genfromtxt('..\\..\\Patterns\\bpm\\b.csv', delimiter=';')
vzor3 = np.genfromtxt('..\\..\\Patterns\\bpm\\c.csv', delimiter=';')
vzor4 = np.genfromtxt('..\\..\\Patterns\\bpm\\d.csv', delimiter=';')


n = 125 * 100
h = Hopfield(n)

h.train_iter(vzor1.flatten())
h.train_iter(vzor2.flatten())
h.train_iter(vzor3.flatten())
h.train_iter(vzor4.flatten())

show(vzor2)
test = noise(vzor2, 0.4)
show(test)

y = h.equip_iter(test.flatten())
show(y.reshape((125, 100)))
y = h.equip_iter(y)
show(y.reshape((125, 100)))
