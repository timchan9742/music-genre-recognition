from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Dropout, Activation, \
        TimeDistributed, Convolution1D, Convolution2D, LSTM, MaxPooling1D, AveragePooling1D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import pickle
from optparse import OptionParser
from sys import stderr, argv
import os

SEED = 42
FILTER_SIZE = 5
FILTER_COUNT = 256
BATCH_SIZE = 32
EPOCH_COUNT = 150

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
        'pop', 'reggae', 'rock']

class TerminateOnBaseline(Callback):
    def __init__(self, monitor='acc', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True

class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='accuracy', value=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def train_CNN_1D(data, model_path):

    x = data['x']
    y = data['y']
    track_path = data['track_paths']
    (x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.3,
            random_state=SEED)

    print('Building model...')

    print(x.shape)
    print(y.shape)

    n_features = x_train.shape[2]
    input_shape = (x_train.shape[1], n_features)
    model_input = Input(input_shape, name='input')
    layer = model_input

    layer = Convolution1D(filters=FILTER_COUNT,
                          kernel_size=FILTER_SIZE,
                          name='convolution_' + '1'
                          )(layer)
    layer = BatchNormalization(momentum=0.9)(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling1D(4)(layer)
    layer = Dropout(0.5)(layer)

    layer = Convolution1D(filters=FILTER_COUNT,
                          kernel_size=FILTER_SIZE,
                          name='convolution_' + '2'
                          )(layer)
    layer = BatchNormalization(momentum=0.9)(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling1D(2)(layer)
    layer = Dropout(0.5)(layer)

    layer = Convolution1D(filters=FILTER_COUNT,
                          kernel_size=FILTER_SIZE,
                          name='convolution_' + '3'
                          )(layer)
    layer = BatchNormalization(momentum=0.9)(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling1D(2)(layer)
    layer = Dropout(0.5)(layer)

    #This wrapper allows to apply a layer to every temporal slice of an input.
    layer = TimeDistributed(Dense(len(GENRES)))(layer)
    layer = Dropout(0.5)(layer)

    time_distributed_merge_layer = Lambda(
            function=lambda x: K.mean(x, axis=1),
            output_shape=lambda shape: (shape[0],) + shape[2:],
            name='output_merged'
        )
    layer = time_distributed_merge_layer(layer)
    layer = Activation('softmax', name='output_realtime')(layer)

    model_output = layer
    model = Model(model_input, model_output)
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

    model.summary()

    print('Training...')
    history = model.fit(
        x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
        validation_data=(x_val, y_val), verbose=1, callbacks=[
            ModelCheckpoint(
                model_path, save_best_only=True, monitor='val_acc', verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_delta=0.01,
                verbose=1
            ),
            ]
    )

    model.save(model_path)
    plot(history)

    return model

def plot(history):
    history_dict = history.history
    history_dict.keys()
    train_acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plot_loss(train_loss, val_loss)
    plt.subplot(1,2,2)
    plot_accuracy(train_acc, val_acc)
    plt.show()

def plot_loss(train_loss, val_loss):
    epochs = range(1, len(train_loss)+1)
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'y', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

def plot_accuracy(train_acc, val_acc):
    epochs = range(1, len(train_acc)+1)
    plt.plot(epochs, train_acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'y', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-d', '--data_path', dest='data_path',
            default=os.path.join(os.path.dirname(__file__),
                'data/data.pkl'),
            help='path to the data pickle', metavar='DATA_PATH')
    parser.add_option('-m', '--model_path', dest='model_path',
            default=os.path.join(os.path.dirname(__file__),
                'models/model.h5'),
            help='path to the output model HDF5 file', metavar='MODEL_PATH')
    options, args = parser.parse_args()

    with open(options.data_path, 'rb') as f:
        data = pickle.load(f)

    model = train_CNN_1D(data, options.model_path)
