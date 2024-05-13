from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import librosa
import soundfile as sf
import numpy as np
from functools import partial
from optparse import OptionParser
import pickle
import os

WINDOW_SIZE = 2048
WINDOW_STRIDE = 1024
N_MELS = 128
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}

##returns a function which will give the output of the layer given the input
def get_layer_output_function(model, layer_name):
    input = model.get_layer('input').input
    output = model.get_layer(layer_name).output
    f = K.function([input], [output])
    return lambda x: f([x])

def compose(f, g):
    return lambda x: f(g(x))

##undo operations
def undo_layer(length, stride, coords):   ##e.g. (0,0) -> (0,4) (0,1) -> (0,5) (0,2) -> (0,6)
    (i, j) = coords
    return (stride * i, stride * (j - 1) + length)

def extract_filters(model, data, filters_path, COUNT):
    x = data['x']
    track_paths = data['track_paths']
    conv_layer_names = []
    i = 1
    while True:
        name = 'convolution_' + str(i)
        try:
            model.get_layer(name)
        except ValueError:
            break
        conv_layer_names.append(name)
        i += 1
    ##Generate undoers for every convolutional layer
    conv_layer_undoers = []
    ##undo the mel spectrogram extraction
    undoer = partial(undo_layer, WINDOW_SIZE, WINDOW_STRIDE)
    for index, name in enumerate(conv_layer_names):
        layer = model.get_layer(name)
        (length,) = layer.kernel_size
        (stride,) = layer.strides
        ##undo the convolution layer
        undoer = compose(undoer, partial(undo_layer, length, stride))
        conv_layer_undoers.append(undoer) #e.g. mel>con3x3  mel>con3x3>pool>con3x3  mel>con3x3>pool>con3x3>pool>con3x3
        ##undo the pooling layer
        if(index == 0):
            undoer = compose(undoer, partial(undo_layer, 4, 4))
        else:
            undoer = compose(undoer, partial(undo_layer, 2, 2))
        conv_layer_output_funs = list(
                map(partial(get_layer_output_function, model), conv_layer_names))

    ##Extract track chunks with highest activations for each filter in each
    ##convolutional layer.
    for (layer_index, output_fun) in enumerate(conv_layer_output_funs):
        layer_path = os.path.join(filters_path, conv_layer_names[layer_index])
        if not os.path.exists(layer_path):
            os.makedirs(layer_path)

        print('Computing outputs for layer', conv_layer_names[layer_index])
        output = output_fun(x)
        output = output[0]
        #print(output.shape) # (1000, 643, 256)
        ##matrices of shape n_tracks x time x n_filters
        max_over_time = np.amax(output, axis=1)
        print(max_over_time.shape) # (1000, 256)
        argmax_over_time = np.argmax(output, axis=1)
        print(argmax_over_time.shape) # (1000, 256)
        ##number of input chunks to extract for each filter
        count = COUNT // 2 ** layer_index
        print("number of chunks to extract: ", count)
        argmax_over_track = \
                np.argpartition(max_over_time, -count, axis=0)[-count :, :]
        #print(argmax_over_track.shape) # (8, 256)
        undoer = conv_layer_undoers[layer_index]
        ##for each filter(256)
        for filter_index in range(argmax_over_track.shape[1]):
            print('Processing layer', conv_layer_names[layer_index], \
                    'filter', filter_index)
            track_indices = argmax_over_track[:, filter_index]
            #print("track_indices:", track_indices)
            time_indices = argmax_over_time[track_indices, filter_index]
            #print("time_indices:", time_indices)
            sample_rate = [None]
            def extract_sample_from_track(undoer, indexes):
                (track_index, time_index) = indexes
                track_path = track_paths[track_index]
                (track_samples, sample_rate[0]) = librosa.load(track_path,
                        mono=True)
                (t1, t2) = undoer((time_index, time_index + 1))
                # print("(time_index, time_index+1): ", time_index, time_index+1)
                # print("(t1, t2): ", t1, t2)
                return track_samples[t1 : t2]

            samples_for_filter = np.concatenate(
                    list(map(partial(extract_sample_from_track, undoer),
                            zip(track_indices, time_indices))))

            filter_path = os.path.join(layer_path,
                    '{}.wav'.format(filter_index))
            sf.write(filter_path, samples_for_filter,
                    sample_rate[0])

'''
'visualize' the filters learned by the cnn layers, concatenates several chunks
resulting in its maximum activation from the tracks of the dataset.
'''
if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-m', '--model_path', dest='model_path',
            default=os.path.join(os.path.dirname(__file__),
                'models/model.h5'),
            help='path to the model HDF5 file', metavar='MODEL_PATH')
    parser.add_option('-d', '--data_path', dest='data_path',
            default=os.path.join(os.path.dirname(__file__),
                'data/data.pkl'),
            help='path to the data pickle',
            metavar='DATA_PATH')
    parser.add_option('-f', '--filters_path', dest='filters_path',
            default=os.path.join(os.path.dirname(__file__),
                'visualizations'),
            help='path to the output directory',
            metavar='FILTERS_PATH')
    parser.add_option('-c', '--COUNT', dest='COUNT',
            default='8',
            help=('number of chunks to extract from the first convolutional ' +
                'layer, this number is halved for each next layer'),
            metavar='COUNT')
    options, args = parser.parse_args()

    model = load_model(options.model_path)

    with open(options.data_path, 'rb') as f:
        data = pickle.load(f)

    extract_filters(model, data, options.filters_path, int(options.COUNT))
