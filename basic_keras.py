# basic_keras.py
# Try to make basic reinforcement learning algorithm.
# ---------------------------------------------------

# Imports
import gym
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
inputs = dataset[:, 0:8]
outputs = dataset[:, 8]

# define model
model = keras.Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(inputs, outputs, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(inputs, outputs)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

