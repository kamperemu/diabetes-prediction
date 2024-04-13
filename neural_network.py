# input.py file contains input functions that are common to all models.
from input import load_dataset, preprocess, XY_split
import tensorflow as tf

# loading and preprocessing the data
data = preprocess(load_dataset('dataset_files/main.csv'))
# reading the data for debugging
# print(data.head())
# splitting the data for the machine learning model
x_train, x_test, y_train, y_test = XY_split(data)

# creating the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
numEpochs = 10

history = model.fit(x_train, y_train, epochs=numEpochs, validation_data=(x_test, y_test))

# different kinds of data metrics for the model
