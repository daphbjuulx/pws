import numpy as np
import csv
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import to_categorical

training = np.genfromtxt('training.csv', delimiter=',')[1:]
training = np.array([i[1:-2] for i in training])
with open('training.csv') as csvfile:
    training_data = list(csv.reader(csvfile, delimiter=","))
training_labels = np.array([1 if row[-1] == 's' else 0 for row in training_data[1:]])
# test = np.genfromtxt('test.csv', delimiter=',')[1:]

# print(training_labels)
# print(training)

print(np.shape(training))

# Build the model
model = Sequential([
  InputLayer(input_shape=(30,)),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(2, activation='sigmoid'),
])

# Compile the model.
model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
model.fit(
  training,
  to_categorical(training_labels),
  epochs=5,
  batch_size=32,
)

model.save_weights('weights.h5')
