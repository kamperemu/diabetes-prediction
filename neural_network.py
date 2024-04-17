# input.py file contains input functions that are common to all models.
from input import load_dataset, preprocess, XY_split, traintest_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# loading and preprocessing the data
data = preprocess(load_dataset('dataset_files/main.csv'))
# reading the data for debugging
# print(data.head())
# splitting the data for the machine learning model
x_train, x_test, y_train, y_test = traintest_split(data)

# creating the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
numEpochs = 10

history = model.fit(x_train, y_train, epochs=numEpochs, validation_data=(x_test, y_test))

# different kinds of data metrics for the model (during the training process tensorflow already provides some meterics)
y_pred = tf.round(model.predict(x_test))
confusion_array = confusion_matrix(tf.round(y_test),y_pred,labels=[0,1])
print("Confusion Matrix")
print(confusion_array)

"""
# visual display for data meterics
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

import seaborn as sn
import pandas as pd

plt.figure()
df_cm = pd.DataFrame(array, range(2), range(2))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
"""