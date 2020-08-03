#This file is to split data into testing and training sets, create generators and the training
#model using Keras ML library. Last step fits the model and adds a checkpoint with weight values.

#Usage: python3 keras_mv.py
#Author: Mac Weinstock


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.callbacks import ModelCheckpoint
from keras_generator import load_emotify, DataGenerator

import os
import pandas as pd
from tqdm import tqdm

from librosa.feature import melspectrogram
from librosa.core import load

import numpy as np

import subprocess

#Bug fix from OSX
os.system('export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES')

data_dir = '/Users/Mac/Desktop/MusicValence/mvdata'
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

#Make generators
train_generator = DataGenerator(train_set, batch_size=4, counter_pos=1)
test_generator = DataGenerator(test_set, batch_size=4, counter_pos=2)

#Create the model
model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), input_shape=(64,100, 1)))
model.add(Conv2D(64, kernel_size=(3,3)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(9, activation='sigmoid'))

model.compile(optimizer="adagrad",
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#Checkpoint
check_dir = '/Users/Mac/Desktop/MusicValence/mvcheckpoint/mv_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(check_dir, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]

#Fit the model
model.fit_generator(generator=train_generator,
                    validation_data=test_generator,
                    use_multiprocessing=True,
                    workers=6,
                    callbacks=callbacks_list)





