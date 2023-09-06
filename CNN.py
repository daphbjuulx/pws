import numpy as np
import csv
from tensorflow import keras
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical


training = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')

training_labels = training.copy()[["Label"]]
training_labels.replace(('b', 's'), (0, 1), inplace=True)

# print(training_labels)
# print(training)

# select input features
training_data = training.copy().drop(columns=["EventId", "Weight", "Label"])
selected_inputs = []
# for value in training_data.columns.values.tolist():
#   # if "PRI" in value:
#   #   selected_inputs.append(value)
#   selected_inputs.append(value)
print(selected_inputs)
# training_data = training_data[selected_inputs]
# test_data = test.copy()[selected_inputs]

print(training_data.shape)

# Build the model
model = Sequential([
  InputLayer(input_shape=(training_data.shape[1], training_data.shape[0], 1, )),
  Conv2D(32, (10, 10), activation="relu"),
  MaxPooling2D((4, 4)),
  Dense(2, activation='relu'),
])

# Compile the model.
model.compile(
  optimizer='sgd',
  loss='mse',
  metrics=['accuracy'],
)

# Train the model.
model.fit(
  training_data,
  training_labels,
  epochs=5,
  batch_size=32,
)

model.save_weights('weights.h5')
