import numpy as np
import math
import csv
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Normalization
from keras.utils import to_categorical
# from scikeras.wrappers import KerasClassifier, KerasRegressor
# # from tensorflow.keras.layers.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_val_predict
# from sklearn import linear_model, tree, ensemble
# from sklearn.model_selection import cross_val_score
# #from tensorflow.keras.wrappers.scikit_learn import kerasClassifier
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder, normalize

from scikeras.wrappers import KerasClassifier

seed = 7
np.random.seed(seed)

training = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')


training_weights = training.copy()[["Weight"]]
training_labels = training.copy()[["Label"]]
training_labels.replace(('b', 's'), (0, 1), inplace=True)

# print(training_labels)
# print(training)

# select input features
training_data = training.copy().drop(columns=["EventId", "Weight", "Label"])
test_data = test.copy().drop(columns=["EventId"])
# selected_inputs = ['DER_mass_transverse_met_lep', 'DER_prodeta_jet_jet', 'DER_pt_tot', 'PRI_tau_pt']
# # for value in training_data.columns.values.tolist():
# #   if "PRI" in value:
# #     selected_inputs.append(value)
# # print(selected_inputs)
# training_data = training_data[selected_inputs]
# test_data = test_data[selected_inputs]

training_data.replace(-999, 0, inplace=True)
# training_data.replace(-999, np.NAN, inplace=True)
# mean = training_data.mean()
# training_data.fillna(mean, inplace=True)

print(training_data.shape)


s = sum(training_weights["Weight"][i] for i in training_labels["Label"] if training_labels["Label"][i] == 1)
b = sum(training_weights["Weight"][i] for i in training_labels["Label"] if training_labels["Label"][i] == 0)
# b = 0
print(s)
print(b)
def AMS(s, b):
  assert s >= 0
  assert b >= 0
  bReg = 10.0
  return math.sqrt(2 * ((s + b + bReg) * math.log(1 + s / (b + bReg)) - s))
print(AMS(s, b))

# Build the model
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
  Dense(2, activation='softmax'),
  ])

  # Compile the model.
  model.compile(
    optimizer='adam',
    # loss='mse',
    loss='binary_crossentropy',
    metrics=['accuracy'],
  )
  return model

model = KerasClassifier(model=create_model, epochs=10, batch_size=64, verbose=1)

# history = model.fit


#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(model, training_data, y=training_labels, cv= kfold)
#results = cross_val_score(model, training_data, training_labels, scoring=scorer(model, training_data, training_labels), cv=kfold)
#sklearn.model.selection.cross_val_score(model,training_data, y = training_labels, cv = kfold)
#print(results)
# cvscores = [kfold]
# print(cvscores)


# Train the model.
#model.fit(
 # training_data,
  #to_categorical(training_labels),
  #epochs=1,
  #batch_size=32,
#)


kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
# results = cross_val_score(model, training_data, y=to_categorical(training_labels), cv=kfold)
# print(results, results.mean())
results = cross_val_predict(model, training_data, y=to_categorical(training_labels), cv=kfold)
print(results)
# print(history.results.keys())

s_list = [i for i, prediction in enumerate(results) if prediction.tolist() == [0, 1]]
print(s_list)
# print(s_list)
s = sum(training_weights["Weight"][i] for i in s_list if training_labels["Label"][i] == 1) # sum of all weights predicted as s that are actually s
b = sum(training_weights["Weight"][i] for i in s_list if training_labels["Label"][i] == 0) # sum of all weights predicted as s that are actually b
print(s)
print(b)

# s = sum(training_weights["Weight"][i] for i in training_labels["Label"] if training_labels["Label"][i] == 1)
# b = sum(training_weights["Weight"][i] for i in training_labels["Label"] if training_labels["Label"][i] == 0)
# print(s)
# print(b)
# def AMS(s, b):
#   assert s >= 0
#   assert b >= 0
#   bReg = 10.0
#   return math.sqrt(2 * ((s + b + bReg) * math.log(1 + s / (b + bReg)) - s))
print(AMS(s, b))


#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(model, training_data, training_labels, cv=kfold)
#print(results)

#splitter=StratifiedShuffleSplit(n_splits=1,random_state=12)
#cross score = cross_val_score(estimator=model, scoring="accuracy", X=X_train, y=y_train, cv=5)

# model.save_weights('weights.h5')

# make predictions for test set
# predictions = model.predict(test_data)
#
# # format predictions
# output = []
# for pred in predictions:
#   if pred[0] > pred[1]:
#     output.append("b")
#   else:
#     output.append("s")
#
#
# print(predictions[:5])
# print(output[:5])



#print(np.argmax(predictions, axis=1))
# print(pd.Series.argmax(predictions, axis = 1))