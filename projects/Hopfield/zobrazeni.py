import numpy as np
import matplotlib.pyplot as plt


def show(vzor):
    # zobrazi prislusny vzor
    plt.imshow(vzor, cmap="gray")
    plt.show()

    
def noise(vzor, prob):
    # zasumeni vzoru s danou pravdepodobnosti
    size = vzor.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if np.random.rand() <= prob:
                vzor[i, j] = -vzor[i,j]

    return vzor
