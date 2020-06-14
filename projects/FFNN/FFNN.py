import numpy as np


class FFNN:
    # trida pro feedforward network podle mych prednasek
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.v = self.init_weights(n_inputs, n_hidden)
        self.w = self.init_weights(n_hidden, n_outputs)
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

    def init_weights(self, n1, n2):
        return np.random.randn(n1 + 1, n2)

    def agregate_hidden(self, x):
        x = np.append(1, x)
        y_a = np.zeros((self.n_hidden, 1))
        for j in range(self.n_hidden):
            for i in range(self.n_inputs + 1):
                y_a[j] = y_a[j] + self.v[i, j] * x[i]

        return y_a

    def activate_hidden(self, y_a):
        y = np.zeros((self.n_hidden, 1))
        for j in range(self.n_hidden):
            y[j] = np.tanh(y_a[j])

        return y

    def d_activate_hidden(self, z):
        # derivace aktivacni funkce tansig
        return (1 + z) * (1 - z)

    def agregate_output(self, x):
        x = np.append(1, x)
        y_a = np.zeros((self.n_outputs, 1))
        for k in range(self.n_outputs):
            for j in range(self.n_hidden + 1):
                y_a[k] = y_a[k] + self.w[j, k] * x[j]

        return y_a

    def activate_output(self, y_a):
        y = np.zeros((self.n_outputs, 1))
        for k in range(self.n_outputs):
            y[k] = y_a[k]

        return y

    def response(self, x):
        z_a = self.agregate_hidden(x)
        z = self.activate_hidden(z_a)
        y_a = self.agregate_output(z)
        y = self.activate_output(y_a)
        return y, z

    def compute_delta_k(self, t, y):
        delta_k = np.zeros((self.n_outputs, 1))
        for k in range(self.n_outputs):
            delta_k[k] = 2 * (t[k] - y[
                k])  # tady by mohla byt jeste derivace aktivacni funkce, ale pocitame linearni identickou akt. funkci, takce derivace = 1

        return delta_k

    def compute_delta_j(self, delta_k, z):
        delta_j = np.zeros((self.n_hidden, 1))
        for j in range(self.n_hidden):
            for k in range(self.n_outputs):
                delta_j[j] = delta_j[j] + delta_k[k] * self.w[j, k]

            delta_j[j] = delta_j[j] * self.d_activate_hidden(z[j])

        return delta_j

    def act_w(self, alpha, delta_k, z):
        z = np.append(1, z)
        for j in range(self.n_hidden + 1):
            for k in range(self.n_outputs):
                self.w[j, k] = self.w[j, k] + alpha * delta_k[k] * z[j]

    def act_v(self, alpha, delta_j, x):
        x = np.append(1, x)
        for i in range(self.n_inputs + 1):
            for j in range(self.n_hidden):
                self.v[i, j] = self.v[i, j] + alpha * delta_j[j] * x[i]

    def BPG_iter(self, x, t, alpha):
        y, z = self.response(x)
        delta_k = self.compute_delta_k(t, y)
        delta_j = self.compute_delta_j(delta_k, z)
        self.act_w(alpha, delta_k, z)
        self.act_v(alpha, delta_j, x)

    def BPG_epoch(self, inputs, targets, alpha):
        dim = np.size(inputs, 1)
        for ind in range(dim):
            self.BPG_iter(inputs[:, ind], targets[:, ind], alpha)

    def compute_E(self, inputs, targets):
        E = 0
        for ind in range(np.size(inputs, 1)):
            output, z = self.response(inputs[:, ind])
            e = 0
            for k in range(len(output)):
                e = e + (targets[k, ind] - output[k]) ** 2

            E = E + e

        E = E / np.size(inputs, 1)
        return E
