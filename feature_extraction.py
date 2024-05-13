import numpy as np
import librosa
from pickle import dump
import os
import sys
from optparse import OptionParser

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
        'pop', 'reggae', 'rock']

WINDOW_SIZE = 2048
WINDOW_STRIDE = 1024
N_MELS = 128
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}
DATA_SIZE = len(GENRES) * 100

def getShape(dataset_path):
    features, _ = loadTrack(os.path.join(dataset_path,
        'blues/blues.00000.wav'))
    return features.shape

def loadTrack(filename, enforce_shape=None):
    y, sr = librosa.load(filename, mono=True)
    features = librosa.feature.melspectrogram(y=y, sr=sr, **MEL_KWARGS).T

    if enforce_shape is not None:
        if features.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - features.shape[0],
                    enforce_shape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0) ##append zeros
        elif features.shape[0] > enforce_shape[0]:
            features = features[: enforce_shape[0], :] ##truncate

    features[features == 0] = 1e-6
    return (np.log(features), float(y.shape[0]) / sr)

def loadAll(dataset_path):
    default_shape = getShape(dataset_path)
    x = np.zeros((DATA_SIZE,) + default_shape, dtype=np.float32)
    y = np.zeros((DATA_SIZE, len(GENRES)), dtype=np.float32)
    track_paths = {}

    for (genre_index, genre_name) in enumerate(GENRES):
        for i in range(DATA_SIZE // len(GENRES)):
            file_name = '{}/{}.000{}.wav'.format(genre_name,
                    genre_name, str(i).zfill(2))
            print('Loading and Processing', file_name)
            path = os.path.join(dataset_path, file_name)
            track_index = genre_index  * (DATA_SIZE // len(GENRES)) + i
            x[track_index], _ = loadTrack(path, default_shape)
            y[track_index, genre_index] = 1
            track_paths[track_index] = os.path.abspath(path)

    return (x, y, track_paths)

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-d', '--dataset_path', dest='dataset_path',
            default=os.path.join(os.path.dirname(__file__), 'genres'),
            help='path to the GTZAN dataset', metavar='DATASET_PATH')
    parser.add_option('-o', '--output_pkl_path', dest='output_pkl_path',
            default=os.path.join(os.path.dirname(__file__), 'data/data.pkl'),
            help='path to the output pickle', metavar='OUTPUT_PKL_PATH')
    options, args = parser.parse_args()

    (x, y, track_paths) = loadAll(options.dataset_path)

    print(x.shape)
    print(y.shape)

    data = {'x': x, 'y': y, 'track_paths': track_paths}
    with open(options.output_pkl_path, 'wb') as f:
        dump(data, f)
