import numpy as np
import math
from tensorflow import keras
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Normalization
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import SGDClassifier

training = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')



training_weights = training.copy()[["Weight"]]
training_labels = training.copy()[["Label"]]
training_labels.replace(('b', 's'), (0, 1), inplace=True)



training_data = training.copy().drop(columns=["EventId", "Weight", "Label"])
test_data = test.copy().drop(columns=["EventId"])



training_data.replace(-999, 0, inplace=True)


def create_model():
  model = Sequential([
  InputLayer(input_shape=(training_data.shape[1],)),
  Normalization(),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(1, activation='softmax'),
  ])

  # Compile the model.
  model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy'],
  )
  return model

model = KerasClassifier(model=create_model, epochs=1, batch_size=64, verbose=0)

sfs = SFS(SGDClassifier(),
          k_features=4,
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=0)

sfs.fit(training_data, training_labels)
print(sfs.k_feature_names_)

# 'DER_mass_transverse_met_lep', 'DER_prodeta_jet_jet', 'DER_pt_tot', 'PRI_tau_pt'

# ValueError: Input 0 of layer "sequential_30" is incompatible with the layer: expected shape=(None, 30), found shape=(None, 2)
