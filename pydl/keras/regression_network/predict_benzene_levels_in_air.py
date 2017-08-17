from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

DATA_DIR = '../../../data'
AIRQUALITY_FILE = os.path.join(DATA_DIR, 'AirQualityUCI.csv')

aqdf = pd.read_csv(AIRQUALITY_FILE, sep=";", decimal=",", header=0)

del aqdf["Date"]
del aqdf["Time"]
del aqdf["Unnamed: 15"]
del aqdf["Unnamed: 16"]

aqdf = aqdf.fillna(aqdf.mean())

Xorig = aqdf.as_matrix()

scaler = StandardScaler()
Xscaled = scaler.fit(Xorig)
Xmeans = scaler.mean_
Xstds = scaler.scale_

y = Xscaled[:, 3]
X = np.delete(Xscaled, 3, axis=1)

train_size = int(0.7 * X.shape[0])
Xtrain, Xtest, ytrain, ytest = X[0:train_size], X[train_size:], y[0:train_size], y[train_size:]

readings = Input(shape=(12, ))
x = Dense(1)(readings)
benzene = Dense(1)(x)

model = Model(inputs=[readings], outputs=[benzene])
model.compile(loss='mse', optimizer='adam')

NUM_EPOCHS = 20
BATCH_SIZE = 10

history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2)

ytest_ = model.predict(Xtest).flatten()
for i in range(10):
    label = (ytest[i] * Xstds[3]) + Xmeans[3]
    prediction = (ytest_[i] * Xstds[3]) + Xmeans[3]
    print("Benzene Conc. expected: {:.3f}, predicted: {:.3f}".format(label, prediction))
