import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten

import os
import pandas as pd
from tqdm import tqdm

from librosa.feature import melspectrogram
from librosa.core import load
from keras_generator import load_emotify, DataGenerator

import numpy as np

data_dir = '/Users/jonny/Desktop/valence_data'
data_file = os.path.join(data_dir, 'data.csv')
data = load_emotify(data_file)

# split into training and testing
percent_test = 0.05

unique_songs = data.path.unique()
test_songs = np.random.choice(unique_songs, round(len(unique_songs)*percent_test))
train_songs = unique_songs[np.invert(np.isin(unique_songs, test_songs))]

test_set = data.loc[np.isin(data.path, test_songs),:]
train_set = data.loc[np.isin(data.path, train_songs),:]

################

# make generators
train_generator = DataGenerator(train_set)
test_generator = DataGenerator(test_set)


model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), input_shape=(64,100, 1)))
model.add(Conv2D(64, kernel_size=(3,3)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(9, activation='sigmoid'))

model.compile(optimizer="adagrad",
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(generator=train_generator,
                    validation_data=test_generator,
                    use_multiprocessing=False)
                    #workers=6)





