import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import pathlib
import os, sys
from optparse import OptionParser
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.image as mpimage

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
        'pop', 'reggae', 'rock']

DATA_SIZE = len(genres) * 100
SEED = 42

def plot_confusion_matrix(conf_matrix):
  conf_matrix_percentage = np.zeros(shape=(10,10))
  for i in range(10):
    for j in range(10):
        conf_matrix_percentage[i][j] = conf_matrix[i][j] / np.sum(conf_matrix[i,:])
  conf_matrix_percentage = np.transpose(np.round(conf_matrix_percentage,2))
  plt.figure(figsize=(8,6))
  plt.matshow(conf_matrix, fignum=1, cmap=plt.cm.Blues)
  plt.colorbar()
  for x in range(len(conf_matrix_percentage)):
    for y in range(len(conf_matrix_percentage)):
      plt.annotate(conf_matrix_percentage[x,y], xy=(x,y), horizontalalignment='center', verticalalignment='center')
  plt.title('Result of CNN')
  plt.xticks(range(10),genres[:10],color='black',fontsize=7)
  plt.yticks(range(10),genres[:10],color='black',fontsize=7)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

##model testing
if __name__ == '__main__':

  f_path = os.path.join(os.path.dirname(__file__),
                  'data/data.pkl')
  with open(f_path, 'rb') as f:
    data = pickle.load(f)

  x = data['x']
  y = data['y']

  (train_set, test_set, train_labels, test_labels) = train_test_split(x, y, test_size=0.3, random_state=SEED)

  model = tf.keras.models.load_model('models/model.h5')
  model.summary()
  loss, acc = model.evaluate(test_set, test_labels)
  print("model accuracy: {:5.2f}%".format(100*acc))

  probability_model = tf.keras.Sequential([model,
                                           tf.keras.layers.Softmax()])
  predictions = probability_model.predict(test_set)
  predictions = [np.argmax(i) for i in predictions]
  test_labels = [np.where(r==1)[0][0] for r in test_labels] ##convert one-hot encoding to encoding with integers

  conf_matrix = confusion_matrix(test_labels, predictions)
  plot_confusion_matrix(conf_matrix)
