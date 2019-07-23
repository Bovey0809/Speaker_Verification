import sys
import os
import torch
from torch.optim.lr_scheduler import StepLR
from data_load import SpeakerDatasetTIMITPreprocessed
from embedder_net import SpeechEmbedder, GE2ELoss
from torch.utils.data import DataLoader
from utils import get_cossim, get_lr
from torch.utils.tensorboard import SummaryWriter


def train(dataset, log_dir='log', N=16, lr=0.01, epochs=100, proj=256, hidden=768, num_layers=3, opt='Adam', debug=False, step_size=2e3):
    subdir = '_'.join(
        [str(i) for i in [N, lr, epochs, proj, hidden, num_layers, opt, step_size]])
    print(subdir)
    writer = SummaryWriter(f'./{log_dir}/{subdir}')
    # define net, loss, optimizer
    device = torch.device('cuda')
    N = int(N)
    M = 6
    embedder_net = SpeechEmbedder(
        hidden=hidden, num_layers=num_layers, proj=proj, N=N, M=M)
    embedder_net.to(device)
    criterion = GE2ELoss(device)
    if opt == 'Adam':
        optimizer = torch.optim.Adam([{'params': embedder_net.parameters()},
                                      {'params': criterion.parameters()}],
                                     lr,
                                     (0.5, 0.9))
    elif opt == 'SGD':
        optimizer = torch.optim.SGD([{'params': embedder_net.parameters()},
                                     {'params': criterion.parameters()}],
                                    lr)
        scheduler = StepLR(optimizer, step_size, gamma=0.5)

    train_dataset = SpeakerDatasetTIMITPreprocessed(dataset, M)
    train_loader = DataLoader(
        train_dataset, N, drop_last=True, shuffle=True, num_workers=0)

    iteration = 0
    thresholds = [0.01*i+0.5 for i in range(50)]
    for epoch in range(epochs):
        if opt == 'SGD':
            scheduler.step()
        epoch_loss = 0
        batch_avg_EER = 0
        # print(epoch_loss, e)
        for batch_id, mel_db in enumerate(train_loader):
            mel_db = mel_db.to(device)
            # assert utt_num % 2 == 0
            mel_db = mel_db.flatten(0, 1)
            optimizer.zero_grad()
            # writer.add_graph(embedder_net, mel_db)
            writer.add_scalar('lr', get_lr(optimizer), iteration)
            embeddings = embedder_net(mel_db)
            loss = criterion(embeddings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            writer.add_scalar('batch loss', loss, iteration)
            iteration += 1

            # validation for eer
            enrollment_embeddings, verification_embeddings = torch.split(
                embeddings,  M//2, dim=1)
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
            for thres in thresholds:
                sim_matrix_thresh = sim_matrix > thres
                FAR = (sum([sim_matrix_thresh[i].sum()-sim_matrix_thresh[i, :, i].sum()
                            for i in range(int(N))]) / (N-1.0)/(M/2)/N)
                FRR = (sum([M/2-sim_matrix_thresh[i, :, i].float().sum()
                            for i in range(int(N))]) / (float(M/2))/N)
                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            writer.add_scalar('EER', EER, iteration)
            writer.add_scalar('BATCH EER', batch_avg_EER, iteration)
            writer.add_scalar('epoch loss', epoch_loss, epoch)
    if not os.path.exists('./models'):
        os.makedirs('./models')
    save_model_path = os.path.join('models', subdir)
    torch.save(embedder_net.state_dict(), save_model_path)
    return save_model_path


if __name__ == '__main__':
    train(sys.argv[1])
