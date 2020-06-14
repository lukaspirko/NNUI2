import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# neuronova sit pro nasobeni dvou cisel

# tvorba trenovaci mnoziny
pocet_dat = 1000
vstupy = 2 * np.random.rand(pocet_dat, 2) - 1  # rozsah -1 az 1
cile = np.zeros((pocet_dat, 1))
for ind in range(pocet_dat):
    cile[ind, 0] = vstupy[ind, 0] * vstupy[ind, 1]

# tvorba modelu
model = Sequential()
model.add(Dense(10, input_dim=2, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='rmsprop', loss='mse')
model.summary()

epoch = 100
batch = 16
val = 0.15
hist = model.fit(x=vstupy, y=cile, epochs=epoch, batch_size=batch, validation_split=val, verbose=2)
model.save('model.h5')
