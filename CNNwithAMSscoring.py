import numpy as np
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Conv1D, MaxPooling1D, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_val_predict
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
# selected_inputs = []
# for value in training_data.columns.values.tolist():
#     if "PRI" in value:
#         selected_inputs.append(value)
# print(selected_inputs)
# training_data = training_data[selected_inputs]
# test_data = test.copy()[selected_inputs]

training_data.replace(-999, 0, inplace=True)

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
        InputLayer(input_shape=(training_data.shape[1], 1)),
        Conv1D(32, 3, activation="relu", padding="same"),
        MaxPooling1D(2),
        Conv1D(32, 3, activation="relu", padding="same"),
        MaxPooling1D(2),
        Conv1D(32, 3, activation="relu", padding="same"),
        MaxPooling1D(2),
        Flatten(),
        Dense(2, activation='softmax'),
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model

model = KerasClassifier(model=create_model, epochs=10, batch_size=64, verbose=1)

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

results = cross_val_predict(model, training_data, y=to_categorical(training_labels), cv=kfold)
print(results)

s_list = [i for i, prediction in enumerate(results) if prediction.tolist() == [0, 1]]
print(s_list)
# print(s_list)
s = sum(training_weights["Weight"][i] for i in s_list if training_labels["Label"][i] == 1) # sum of all weights predicted as s that are actually s
b = sum(training_weights["Weight"][i] for i in s_list if training_labels["Label"][i] == 0) # sum of all weights predicted as s that are actually b
# print(s)
# print(b)

ams_score = AMS(s, b)
print(ams_score)
