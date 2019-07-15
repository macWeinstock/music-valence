from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import os
import pandas as pd
from tqdm import tqdm

from librosa.feature import chroma_stft
from librosa.core import load

@registry.register_problem
class MusicValence(problem.Problem):


    @property
    def dataset_splits(self):
      """Splits of data to produce and number of output   shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 100,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def is_generate_per_split(self):
        # please generate training/test splits for us
        return False

    def generate_samples(self, data_dir, tmp_dir, dataset_split):

        data_dir = '/Users/jonny/Dropbox/school/PSY 348 Share/19SU/projects/music_valence/emotify'
        data_file = os.path.join(data_dir, 'data.csv')
        data = pd.read_csv(data_file)

        # drop columns
        data.drop(axis=1, columns=[' mood', ' liked', ' disliked', ' age', ' gender', ' mother tongue'], inplace=True)

        data.rename(mapper={'track id': 'track_id',
                            ' genre':'genre',
                            ' amazement':'amazement',
                            ' solemnity':'solemnity',
                            ' tenderness':'tenderness',
                            ' nostalgia':'nostalgia',
                            ' calmness':'calmness',
                            ' power':'power',
                            ' joyful_activation':'joy',
                            ' tension':'tension',
                            ' sadness':'sadness'},
                    axis=1, inplace=True)

        data.loc[:,'path'] = ""
        for i in range(data.shape[0]):
            data.loc[i,'path'] = os.path.join(data_dir, 'emotifymusic', data.loc[i,'genre'], str(data.loc[i,'track_id'])+'.mp3')

        for i, row in tqdm(data.iterrows()):
            aud, fs = load(row.path)


            # get a chromagram from the audio
            coefs = chroma_stft(aud, sr=fs, n_fft=2 ** 15, hop_length=2 ** 14)

            window = 10
            step_size = 2
            for k in range(0,coefs.shape[1]-window,step=step_size):
                clip = coefs[:,k:k+window]
                yield {
                    'inputs': clip,
                    'targets': row['']
                }












    

