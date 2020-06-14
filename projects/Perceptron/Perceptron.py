import numpy as np

class Perceptron:

    def __init__(self, n_inputs):
        self.init_weights(n_inputs)

    def init_weights(self, n_inputs):
        #self.w = np.zeros((n_inputs + 1, 1))
        self.w = np.random.randn(n_inputs + 1, 1)

    def evaluate_error(self, inputs_val, targets_val):
        # vyhodnoceni celkove chyby site
        dim = np.size(inputs_val, 1)
        E = 0
        for i in range(dim):
            y = self.calc_output(inputs_val[:, i])
            e = targets_val[i] - y
            E = E + e * e

        E = E / dim
        return E

    def hebb(self, x, t):
        # Hebbovo uceni perceptronu
        dim = np.size(x, 1)
        for i in range(dim):
            self.hebb_iter(x[:,i], t[i])

    def hebb_iter(self, x, t):
        # jedna iterace Hebbova uceni
        x = np.append(1, x)
        for i in range(len(self.w)):
            self.w[i] = self.w[i] + x[i] * t

    def error_train(self, x, t, alfa):
        # chybove uceni perceptronu
        dim = np.size(x, 1)
        for i in range(dim):
            self.error_train_iter(x[:,i], t[i], alfa)

    def error_train_iter(self, x, t, alfa):
        # jedna iterace chyboveho uceni perceptronu
        y = self.calc_output(x)
        e = t - y
        x = np.append(1, x)
        for i in range(len(self.w)):
            self.w[i] = self.w[i] + alfa * x[i] * e

    def calc_output(self, x):
        # vypocet vystupu z perceptronu
        y_a = self.aggregate(x)
        y = self.activate(y_a)
        return y

    def aggregate(self, x):
        # agregacni funkce
        y_a = self.w[0]
        for i in range(len(x)):
            y_a = y_a + self.w[i + 1] * x[i]

        return y_a

    def activate(self, y_a):
        # aktivacni funkce
        if y_a >= 0:
            return 1
        else:
            return -1
    
    
