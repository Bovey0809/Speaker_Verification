'''
Created on August 29, 2019 22:05:51 JST

@author: HOU BOWEI

Mail:  my364007886@gmail.com
'''
import numpy as np
import librosa
from scipy.io import wavfile
import torch

from embedder_net import SpeechEmbedder


def convert(model_path, wav_file_path):
    # data preprocessing
    tisv_frame = 180
    hop = 0.01
    window = 0.025
    nfft = 512
    nmels = 40
    sr, _ = wavfile.read(wav_file_path)
    utter_min_len = (tisv_frame * hop + window) * sr
    utter, sr = librosa.core.load(wav_file_path, sr)
    intervals = librosa.effects.split(utter, top_db=60)
    utterances_spec = []
    for interval in intervals:
        if (interval[1]-interval[0]) > utter_min_len:
            utter_part = utter[interval[0]:interval[1]]
            S = librosa.core.stft(y=utter_part, n_fft=nfft,
                                  win_length=int(window * sr), hop_length=int(hop * sr))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(
                sr=sr, n_fft=nfft, n_mels=nmels)
            # log mel spectrogram of utterances
            S = np.log10(np.dot(mel_basis, S) + 1e-6)
            # first 180 frames of partial utterance
            utterances_spec.append(S[:, :tisv_frame])
            # last 180 frames of partial utterance
            utterances_spec.append(S[:, -tisv_frame:])
    utterances_spec = np.array(utterances_spec)
    utterance = utterances_spec[:, :, :160]
    utterance = torch.tensor(np.transpose(utterance, axes=(0, 2, 1)))

    # load model
    device = torch.device('cuda')
    checkpoint = torch.load(model_path)
    hidden, num_layers, proj = checkpoint['hyper_parameters']
    embedder_net = SpeechEmbedder(40, hidden, num_layers, proj).to(device)
    embedder_net.load_state_dict(checkpoint['model_state_dict'])
    embedder_net.eval()
    mel_db = utterance.flatten(0, 1)
    mel_db = mel_db.unsqueeze(0).to(device)
    embeddings = embedder_net(mel_db)
    return embeddings


if __name__ is '__main__':
    convert('./models/vctk_tf',
            './datasets/jnas_smartcallcenter/F003/F003_BF003A01DT.wav')
