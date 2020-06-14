import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model.h5')
model.summary()

cil = np.zeros(11)
for ind1 in np.arange(-1.0, 1.0, 0.2):
    for ind2 in np.arange(-1.0, 1.0, 0.2):
        cil = ind1 * ind2
        vstup = np.array([[ind1, ind2]])
        vystup = model.predict(vstup)
        print(str(ind1) + ' * ' + str(ind2) + ' = ' + str(cil))
        print('Vystup ze site: ' + str(vystup))
