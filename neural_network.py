import tensorflow as tf
import helper
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from scikeras.wrappers import KerasRegressor

import warnings
warnings.filterwarnings("ignore")

'''
dataset = 'set1', feature_selection = 0 -> ['relu', 'sigmoid', 'relu', 'adam', 'mean_squared_error']
dataset = 'set2', feature_selection = 0 -> ['sigmoid', 'relu', 'relu', 'adam', 'mean_squared_error']
dataset = 'combined', feature_selection = 0 -> ['sigmoid', 'relu', 'relu', 'adam', 'mean_squared_error']
dataset = 'combined', feature_selection = 2 -> ['sigmoid', 'relu', 'sigmoid', 'adam', 'binary_crossentropy']
'''


def buildmodel(input_neurons, hidden_neurons=8, input_activation="sigmoid", hidden_activation="relu", output_activation="relu", model_optimizer="adam", model_loss="mean_squared_error"):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_neurons, activation=input_activation, input_shape=(input_neurons,)),
    tf.keras.layers.Dense(hidden_neurons, activation=hidden_activation),
    tf.keras.layers.Dense(1, activation=output_activation)
  ])
  model.compile(optimizer=model_optimizer, loss=model_loss, metrics=['accuracy', 'mse'])
  return(model)

def basicNN():
  # load the data
  data = helper.loadXYtraintest()

  input_neurons = data['x_train'].shape[1]
  

  # creating the neural network model
  model = buildmodel(input_neurons)
  numEpochs = 1

  history = model.fit(data['x_train'], data['y_train'], epochs=numEpochs, validation_data=(data['x_test'], data['y_test']))

  # different kinds of data metrics for the model (during the training process tensorflow already provides some meterics)
  data['y_pred'] = tf.round(model.predict(data['x_test']))

  # output of data metrics of the model
  helper.print_common_data_metrics(data['y_test'], data['y_pred'])

  return model

def crossvalidateNN():
  # load the data
  X, Y = helper.loadXY()

  # https://stackoverflow.com/questions/48085182/cross-validation-in-keras

  input_neurons = X.shape[1]

  def buildcvmodel():
    return buildmodel(input_neurons)

  estimator= KerasRegressor(build_fn=buildcvmodel, epochs=10, batch_size=10, verbose=0)
  kfold= KFold(n_splits=5)
  results= cross_val_score(estimator, X, Y, cv=kfold)  # 2 cpus
  return results.mean()

def optimizeparameterNN():
  # load the data
  data = helper.loadXYtraintest()

  input_neurons = data['x_train'].shape[1]
  activations = ["relu", "sigmoid"]
  optimizers = ["adam", "sgd"]
  losses = ["binary_crossentropy", "mean_squared_error"]

  best_results = 0
  best_model = []
  for activation1 in activations:
    for activation2 in activations:
      for activation3 in activations:
        for optimizer in optimizers:
          for loss in losses:
            model = buildmodel(input_neurons, input_activation=activation1, hidden_activation=activation2, output_activation=activation3,model_optimizer=optimizer, model_loss=loss)
            numEpochs = 10

            history = model.fit(data['x_train'], data['y_train'], epochs=numEpochs, validation_data=(data['x_test'], data['y_test']))

            # accuracy
            results = model.evaluate(data['x_test'], data['y_test'])

            if results[1] > best_results:
              best_results = results[1]
              best_model = [activation1, activation2, activation3, optimizer, loss]
  print(best_model)
  print(best_results)
if __name__ == "__main__":
  model = basicNN()
  #crossvalidateNN()
  #optimizeparameterNN()
  while True:
    x = pd.DataFrame(data=None, columns=helper.get_header()[:-1], index=[0])
    for i in helper.get_header()[:-1]:
      x[i] = int(input(f"Input {i}: "))
    print()
    if model.predict([x]) == 0:
      print("No Diabetes")
    else:
      print("Diabetes")
    print()

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