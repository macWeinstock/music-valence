from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators.image_utils import Image2ClassProblem
from tensor2tensor.data_generators.speech_recognition import SpeechRecognitionProblem
from tensor2tensor.utils import registry
from tensor2tensor.layers import modalities
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import audio_encoder

import tensorflow as tf

import os
import pandas as pd
from tqdm import tqdm

import numpy as np

from librosa.feature import chroma_stft
from librosa.feature import melspectrogram
from librosa.core import load

@registry.register_problem
class MusicValence(SpeechRecognitionProblem):

    @property
    def input_space_id(self):
        return problem.SpaceID.AUDIO_SPECTRAL

    @property
    def target_space_id(self):
        return problem.SpaceID.GENERIC

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

    def feature_encoders(self, _):
        return {
            "inputs": None,  # Put None to make sure that the logic in
            # decoding.py doesn't try to convert the floats
            # into text...
            "waveforms": audio_encoder.AudioEncoder(),
            "targets": None,
        }

    @property
    def is_generate_per_split(self):
        # please generate training/test splits for us
        return False

    @property
    def already_shuffled(self):
        return False

    def generate_data(self, data_dir, tmp_dir, task_id=-1):

        filepath_fns = {
            problem.DatasetSplit.TRAIN: self.training_filepaths,
            problem.DatasetSplit.EVAL: self.dev_filepaths,
            problem.DatasetSplit.TEST: self.test_filepaths,
        }

        split_paths = [(split["split"], filepath_fns[split["split"]](
            data_dir, split["shards"], shuffled=self.already_shuffled))
                       for split in self.dataset_splits]
        all_paths = []
        for _, paths in split_paths:
            all_paths.extend(paths)

        if self.is_generate_per_split:
            raise NotImplementedError()
        else:
            generator_utils.generate_files(
                self.generator(
                    data_dir, tmp_dir, problem.DatasetSplit.TRAIN),
                all_paths)

        generator_utils.shuffle_dataset(all_paths)

    def generator(self, data_dir, tmp_dir, dataset_split):

        data_file = os.path.join(data_dir, 'data_trim.csv')
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
            aud = aud.astype(np.float32)

            window = fs

            for k in range(0, aud.shape[0]-fs,round(fs/2)):
                yield {
                    'waveforms': aud[k:k+window].tolist(),
                    'targets':row.iloc[2:-1].values.astype(np.int64).tolist()
                }

            #
            # # get a chromagram from the audio
            # coefs = melspectrogram(aud, sr=fs, n_fft=2 ** 12, hop_length=2 ** 11,
            #                        n_mels=64, fmax=10000)
            #
            #
            # window = 100
            # step_size = 20
            # for k in range(0,coefs.shape[1]-window,step_size):
            #     clip = coefs[:,k:k+window]
            #     yield {
            #         'inputs': clip.flatten().tolist(),
            #         'targets': row.iloc[2:-1].values.astype(np.int64).tolist()
            #     }

    def example_reading_spec(self):
        data_fields = {
            'inputs': tf.VarLenFeature(tf.float32),
            'targets': tf.VarLenFeature(tf.int64)
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    # def preprocess_example(self, example, mode, hparams):
    #     return preprocess_example_common(example, mode, hparams)
    #

    def hparams(self, defaults, model_hparams):

        p = model_hparams
        # Filterbank extraction in bottom instead of preprocess_example is faster.
        p.add_hparam("audio_preproc_in_bottom", False)
        # The trainer seems to reserve memory for all members of the input dict
        p.add_hparam("audio_keep_example_waveforms", False)
        p.add_hparam("audio_sample_rate", 22050)
        p.add_hparam("audio_preemphasis", 0.97)
        p.add_hparam("audio_dither", 1.0 / np.iinfo(np.int16).max)
        p.add_hparam("audio_frame_length", 25.0)
        p.add_hparam("audio_frame_step", 10.0)
        p.add_hparam("audio_lower_edge_hertz", 20.0)
        p.add_hparam("audio_upper_edge_hertz", 8000.0)
        p.add_hparam("audio_num_mel_bins", 80)
        p.add_hparam("audio_add_delta_deltas", True)
        p.add_hparam("num_zeropad_frames", 250)

        hp = defaults
        hp.modality = {"inputs": modalities.ModalityType.AUDIO_SPECTRAL,
                       "targets": modalities.ModalityType.CLASS_LABEL}
        # hp.vocab_size = {"inputs": None,
        #                  "targets": 256}












    

