import torch
from data_load import SpeakerDatasetTIMITPreprocessed
from embedder_net import SpeechEmbedder
from torch.utils.data import DataLoader
from utils import get_cossim


def test(model_path, **kargs):
    # test network
    N = 30
    M = 6
    test_dataset = SpeakerDatasetTIMITPreprocessed('./test_tisv/', M)
    test_loader = DataLoader(test_dataset, batch_size=N,
                             drop_last=True, shuffle=True, num_workers=12)
    embedder_net = SpeechEmbedder(
        40, kargs['hidden'], kargs['num_layers'], kargs['proj'], N, M//2)

    embedder_net.load_state_dict(torch.load(model_path))
    device = torch.device('cuda')
    embedder_net.to(device)
    embedder_net.eval()
    epochs = 10
    acc = 0
    avg_EER = 0
    for e in range(epochs):
        batch_avg_EER = 0
        for batch_id, mel_db in enumerate(test_loader):
            mel_db = mel_db.to(device)
            utt_num = mel_db.shape[1]
            assert utt_num % 2 == 0
            enrollment, verification = torch.split(mel_db, utt_num // 2, dim=1)
            enrollment_flatten = enrollment.flatten(0, 1)
            verification_flatten = verification.flatten(0, 1)
            enrollment_embeddings = embedder_net(enrollment_flatten)
            verification_embeddings = embedder_net(verification_flatten)
            enrollment_centroids = enrollment_embeddings.mean(1)
            # what if verification get centorids
            sim_matrix = get_cossim(
                verification_embeddings, enrollment_centroids)  # N * M * proj

            # calculating EER
            diff = 1
            EER = 0
            EER_thresh = 0
            EER_FAR = 0
            EER_FRR = 0

            for thres in [0.01*i+0.5 for i in range(50)]:
                sim_matrix_thresh = sim_matrix > thres
                FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i, :, i].float().sum() for i in range(int(N))])
                       / (N-1.0)/(float(M/2))/N)
                FRR = (sum([M/2-sim_matrix_thresh[i, :, i].float().sum() for i in range(int(N))])
                       / (float(M/2))/N)
                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            print("EER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" %
                  (EER, EER_thresh, EER_FAR, EER_FRR))

            # calculating ACC
            batch_acc = 0
            acc += batch_acc
            print(f"BATCH ACC:{batch_acc}")
        avg_EER += batch_avg_EER/(batch_id+1)
    avg_EER = avg_EER / epochs
    print(f"AVG EER:{avg_EER}")
    return acc/10


