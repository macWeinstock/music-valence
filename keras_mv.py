import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D

import os
import pandas as pd
from tqdm import tqdm

from librosa.feature import melspectrogram
from librosa.core import load

import numpy as np

data_dir = '/Users/jonny/Desktop/valence_data'
data_file = os.path.join(data_dir, 'data_trim.csv')
data = pd.read_csv(data_file)

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
for i in range(data.shape[0]):
    data.loc[i, 'path'] = os.path.join(data_dir, 'emotifymusic', data.loc[i, 'genre'],
                                       str(data.loc[i, 'track_id']) + '.mp3')

for i, row in tqdm(data.iterrows()):
    aud, fs = load(row.path)

    # get a chromagram from the audio
    coefs = melspectrogram(aud, sr=fs, n_fft=2 ** 12, hop_length=2 ** 11,
                           n_mels=64, fmax=10000)

    window = 100
    step_size = 20
    for k in range(0, coefs.shape[1] - window, step_size):
        clip = coefs[:, k:k + window]
        inputs.append(clip)
        outputs.append(row.iloc[2:-1].values.astype(np.int64))


################

model = Sequential()
model.add(Dense(128, input_dim=100))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adagrad",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

X = np.row_stack(inputs)
Y = np.row_stack(outputs)

data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))
model.fit(data, labels)





