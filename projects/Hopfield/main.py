import numpy as np
from zobrazeni import show
from zobrazeni import noise
from Hopfield import Hopfield

vzor1 = np.genfromtxt('..\\..\\Patterns\\csv\\a.csv', delimiter=';')
vzor2 = np.genfromtxt('..\\..\\Patterns\\csv\\b.csv', delimiter=';')
vzor3 = np.genfromtxt('..\\..\\Patterns\\csv\\c.csv', delimiter=';')
vzor4 = np.genfromtxt('..\\..\\Patterns\\csv\\d.csv', delimiter=';')
vzor5 = np.genfromtxt('..\\..\\Patterns\\csv\\e.csv', delimiter=';')
vzor6 = np.genfromtxt('..\\..\\Patterns\\csv\\f.csv', delimiter=';')
vzor7 = np.genfromtxt('..\\..\\Patterns\\csv\\g.csv', delimiter=';')
vzor8 = np.genfromtxt('..\\..\\Patterns\\csv\\h.csv', delimiter=';')


n = 12 * 10
h = Hopfield(n)
h.train_iter(vzor1.flatten())
h.train_iter(vzor2.flatten())
h.train_iter(vzor3.flatten())
h.train_iter(vzor4.flatten())
h.train_iter(vzor5.flatten())
h.train_iter(vzor6.flatten())
h.train_iter(vzor7.flatten())
h.train_iter(vzor8.flatten())

show(vzor4)
test = noise(vzor4, 0.3)
show(test)
y = h.equip_iter(test.flatten())
show(y.reshape((12, 10)))
y = h.equip_iter(y)
show(y.reshape((12, 10)))
