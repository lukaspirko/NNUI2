import numpy as np

class Hopfield:
    def __init__(self, n_inputs):
        self.init_weights(n_inputs)

    def init_weights(self, n_inputs):
        # inicializace vah
        self.w = np.zeros((n_inputs, n_inputs))

    def train_iter(self, x):
        # natrenovani jednoho vzoru
        n = len(x)
        #delta_w = np.zeros((n, n))
        #for i in range(n):
        #    for j in range(n):
        #        delta_w[i, j] = x[i] * x[j]
        delta_w = np.tensordot(x, x, axes=0)

        self.w = self.w + delta_w

    def equip_iter(self, x):
        # jedna iterace vybavovani
        n = len(x)
        #y_a = np.zeros((n, 1))
        #for i in range(n):
        #    for j in range(n):
        #        y_a[i] = y_a[i] + self.w[i, j] * x[j]
        y_a = np.tensordot(self.w, x, axes=1)

        y = np.zeros((n, 1))
        for i in range(n):
            if y_a[i] >= 0:
                y[i] = 1
            else:
                y[i] = -1

        return y
                    
                












            
        

    
