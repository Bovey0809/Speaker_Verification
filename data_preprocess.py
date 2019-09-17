import numpy as np
import os
import librosa
from shutil import rmtree
import sys
import glob
# import scipy


def save_spectrogram_tisv(audio_path,
                          dataset_name='preprocessed_jnas',
                          sr=16000):
    tisv_frame = 180
    hop = 0.01
    window = 0.025
    # sr = 8000
    nfft = 512
    nmels = 40

    print("start text independent utterance feature extraction")
    audio_path = glob.glob(os.path.join(audio_path, '*'))
    train_dataset_path = f"./{dataset_name}/train_tisv"
    test_dataset_path = f"./{dataset_name}/test_tisv"
    rmtree(train_dataset_path, ignore_errors=True)
    rmtree(test_dataset_path, ignore_errors=True)
    # make folder to save train file
    os.makedirs(train_dataset_path, exist_ok=True)
    # make folder to save test file
    os.makedirs(test_dataset_path, exist_ok=True)
    _minium_name_file_per_speaker = 10000
    utter_min_len = (tisv_frame * hop + window) * \
        sr    # lower bound of utterance length
    total_speaker_num = len(audio_path)
    # split total data 90% train and 10% test
    train_speaker_num = (total_speaker_num//10)*9
    print("total speaker number : %d" % total_speaker_num)
    print("train : %d, test : %d" %
          (train_speaker_num, total_speaker_num-train_speaker_num))
    for i, folder in enumerate(audio_path):
        print("%dth speaker processing..." % i)
        utterances_spec = []
        for utter_name in os.listdir(folder):
            if utter_name[-4:] == '.wav':
                # path of each utterance
                utter_path = os.path.join(folder, utter_name)
                utter, sr = librosa.core.load(
                    utter_path, sr)        # load utterance audio
                intervals = librosa.effects.split(
                    utter, top_db=30)         # voice activity detection
                for interval in intervals:
                    # If partial utterance is sufficient long,
                    if (interval[1]-interval[0]) > utter_min_len:
                        # save first and last 180 frames of spectrogram.
                        utter_part = utter[interval[0]:interval[1]]
                        S = librosa.core.stft(y=utter_part,
                                              n_fft=nfft,
                                              win_length=int(window * sr),
                                              hop_length=int(hop * sr))
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
        if utterances_spec.shape[0] < 6:
            print(f"{i}th speaker {folder} not enough files")
            continue
        if utterances_spec.shape[0] < _minium_name_file_per_speaker:
            _minium_name_file_per_speaker = utterances_spec.shape[0]
        if i < train_speaker_num:
            np.save(os.path.join(train_dataset_path, "speaker%d.npy" %
                                 i), utterances_spec)
        else:
            np.save(os.path.join(test_dataset_path,
                                 f"speaker{i-train_speaker_num}"),
                    utterances_spec)
        print(utterances_spec.shape)


if __name__ == '__main__':
    save_spectrogram_tisv(sys.argv[1], sys.argv[2])
