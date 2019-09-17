import sys
import os
from glob import glob
import torch
from data_load import SpeakerDatasetTIMITPreprocessed
from embedder_net import SpeechEmbedder, GE2ELoss
from torch.utils.data import DataLoader
from utils import get_acc, get_eer


def test(model_path, data_path, **kargs):
    # test network
    test_files = glob(os.path.join(data_path, '*.npy'))
    N = len(test_files)
    M = kargs['M']
    test_dataset = SpeakerDatasetTIMITPreprocessed(data_path, M)
    test_loader = DataLoader(test_dataset, batch_size=N,
                             drop_last=True, shuffle=True, num_workers=12)
    embedder_net = SpeechEmbedder(
        40, kargs['hidden'], kargs['num_layers'], kargs['proj'], N, M)

    embedder_net.load_state_dict(torch.load(model_path))
    device = torch.device('cuda')
    embedder_net.to(device)
    embedder_net.eval()

    criterion = GE2ELoss(device)
    criterion.eval()
    epochs = 10
    for e in range(epochs):
        for batch_id, mel_db in enumerate(test_loader):
            mel_db = mel_db.to(device)
            utt_num = mel_db.shape[1]
            assert utt_num % 2 == 0
            mel_db = mel_db.flatten(0, 1)
            embeddings = embedder_net(mel_db)
            sim_matrix = criterion(embeddings)
            acc = get_acc(sim_matrix, N, M)
            thresh, EER, FAR, FRR = get_eer(embeddings, N, M)
            print(
                f"ACC:{acc},EER:{EER},thresh:{thresh},FAR:{FAR},FRR:{FRR}")
    return acc


if __name__ == '__main__':
    test(*sys.argv)
