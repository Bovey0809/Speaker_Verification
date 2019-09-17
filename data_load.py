import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random


class SpeakerDatasetTIMITPreprocessed(Dataset):

    def __init__(self, data_path, M, shuffle=True, utter_start=0):

        # data path
        self.path = data_path
        self.utter_num = M
        self.file_list = os.listdir(self.path)
        self.shuffle = shuffle
        self.utter_start = utter_start

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        np_file_list = os.listdir(self.path)

        if self.shuffle:
            selected_file = random.sample(np_file_list, 1)[0]
            # select random speaker
        else:
            selected_file = np_file_list[idx]

        # load utterance spectrogram of selected speaker
        utters = np.load(os.path.join(self.path, selected_file))
        if self.shuffle:
            # select M utterances per speaker
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)
            utterance = utters[utter_index]
        else:
            # utterances of a speaker [batch(M), n_mels, frames]
            utterance = utters[self.utter_start: self.utter_start+self.utter_num]

        # TODO implement variable length batch size
        utterance = utterance[:, :, :160]

        # transpose [batch, frames, n_mels]
        utterance = torch.tensor(np.transpose(utterance, axes=(0, 2, 1)))
        return utterance
