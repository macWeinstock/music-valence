#Convert a dataset with with paths to .mp3 files and associated valences into list
#of tuples with spectrograms and associated valences (songData(spectrograms, valences))

#Usage: python3 mv_musicval.py
#Author: Mac Weinstock

#imports
import pandas as pd
import numpy as np
from scipy.signal import spectrogram
from scipy.io import wavfile
from librosa.feature import melspectrogram
from librosa.core import load

# some .csv file with path, valence, etc. information as columns
valence_file = "/Users/Mac/Desktop/MusicValence/mvdata/mv_tempdata.csv"

#read csv file with pandas method
val = pd.read_csv(valence_file)

#create list of spectrograms and dictionary of valences
spectrograms = []
valences = []    

#iterate through rows of the valence file, storing a spectrogram and valence for each
for i, row in val.iterrows():
    #read in filename
    mp3_path = val['PATH'][i]

    # read in sampling rate and the samples of the audio from the filename
    aud, fs = load(mp3_path)
    
    # get a spectrogram from the audio
    coefs = melspectrogram(aud, sr=fs, n_fft=2**15, hop_length=2**14)

    # store a flattened version of the spectrogram -- flatten so that we make a n_songs x n_features array later
    spectrograms.append(coefs.flatten())

    #append valences to variable
    valences.append(row[2:11])

# convert the list of 1-d arrays we made in the loop to a n_songs x n_features array
spectrograms = np.row_stack(spectrograms) 

#declare songData
songData = []

#append spectrograms and valences to list songData
songData.append(spectrograms)
songData.append(valences)

print(songData)

