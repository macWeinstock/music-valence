import keras
import numpy as np
import pandas as pd
import os
import sys

from librosa.feature import melspectrogram
from librosa.core import load

from tqdm import trange, tqdm

def load_emotify(data_path):
    data = pd.read_csv(data_path)

    # drop columns
    data.drop(axis=1, columns=[' mood', ' liked', ' disliked', ' age', ' gender', ' mother tongue'], inplace=True)

    data.rename(mapper={'track id': 'track_id',
                        ' genre': 'genre',
                        ' amazement': 'amazement',
                        ' solemnity': 'solemnity',
                        ' tenderness': 'tenderness',
                        ' nostalgia': 'nostalgia',
                        ' calmness': 'calmness',
                        ' power': 'power',
                        ' joyful_activation': 'joy',
                        ' tension': 'tension',
                        ' sadness': 'sadness'},
                axis=1, inplace=True)

    data.loc[:, 'path'] = ""
    inputs = []
    outputs = []
    for i in trange(data.shape[0]):
        if (data.loc[i, 'track_id'] % 100) == 0:
            data.loc[i, 'path'] = os.path.join(os.path.split(data_path)[0], 'emotifymusic', data.loc[i, 'genre'],
                                               "100.mp3")
        else:
            data.loc[i, 'path'] = os.path.join(os.path.split(data_path)[0], 'emotifymusic', data.loc[i, 'genre'],
                                            str(data.loc[i, 'track_id'] % 100) + '.mp3')

    return data


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, song_df, batch_size=32, dim=(64,100), n_channels=1,
                 n_classes=9, shuffle=True, counter_pos=1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.song_df = song_df.reset_index(drop=True).copy()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.audio = {}
        self.window = 100 # size of audio subsamples
        self.step_size = 20 # number of steps to take between subsamples

        # make pbar to keep track of how many files loaded
        self.total_files = len(song_df.path.unique())
        self.pbar = tqdm(position=counter_pos, total=self.total_files)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # FIXME: SLOPPY HACK - HARDCODING NUMBER OF SUBSAMPLES PER TRACK
        n_subsamples = 28
        return int(np.floor(self.song_df.shape[0]/self.batch_size))
        #return int(np.floor((self.song_df.shape[0]*n_subsamples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #if index > (self.song_df.shape[0]-self.batch_size):
        #    index = index % self.song_df.shape[0]

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = self.song_df.iloc[indexes,:].reset_index(drop=True).copy()

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.song_df.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.bool)

        # Generate data
        for i, row in list_IDs_temp.iterrows():
            if row.path not in self.audio.keys():
                #print('{} - loading {}'.format(i, row.path))
                #sys.stdout.flush()
                aud, fs = load(row.path)
                coefs = melspectrogram(aud, sr=fs, n_fft=2 ** 12, hop_length=2 ** 11,
                                       n_mels=64, fmax=10000)
                self.audio[row.path] = coefs
                #print('{} - loaded!'.format(i))
                #sys.stdout.flush()
                # we've loaded one more track, add it to the counter
                self.pbar.update(1)

            start_ind = np.random.randint(low=0,high=self.audio[row.path].shape[1]-self.window)
            clip = self.audio[row.path][:,start_ind:start_ind+self.window]
            #
            # start_ind = np.random.randint(low=0,high=coefs.shape[1]-self.window)
            # clip = coefs[:,start_ind:start_ind+self.window]
            X[i,:,:,0] = clip
            Y[i,:] = row.iloc[2:-1].values.astype(np.int64)

        return X, Y