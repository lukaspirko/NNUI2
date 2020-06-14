from FFNN import FFNN
import numpy as np
import matplotlib.pyplot as plt

net = FFNN(1, 6, 1)
alpha = 0.1
vstup = np.linspace(-0.8, 0.8, 1000)
vstup.shape = (1, len(vstup))
cil = np.square(vstup)

epochs = 1000
E = np.zeros(1000)
for ep in range(epochs):
    net.BPG_epoch(vstup, cil, alpha)
    print(ep)
    E[ep] = net.compute_E(vstup, cil)

vystup = np.zeros(cil.shape)
for ind in range(np.size(vstup, 1)):
    vystup[:, ind], z = net.response(vstup[:, ind])

plt.plot(vstup[0, :], cil[0, :])
plt.plot(vstup[0, :], vystup[0, :])

plt.show()

plt.semilogy(E)
plt.show()
