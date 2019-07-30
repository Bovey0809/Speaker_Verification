import sys
import os
import torch
from torch.optim.lr_scheduler import StepLR
from data_load import SpeakerDatasetTIMITPreprocessed
from embedder_net import SpeechEmbedder, GE2ELoss
from torch.utils.data import DataLoader
from utils import get_cossim, calc_loss, get_acc
from torch.utils.tensorboard import SummaryWriter


def train(dataset, log_dir='test_acc', N=128, lr=0.0001, epochs=2000, proj=512, hidden=768, num_layers=3, opt='Adam', step_size=2e3, save_model=True):
    '''
    Training the model with preprocessed datasets.
    Example
    python train.py dataset/train_tisv/
    
    With grid search method together
    python grid_search.py
    '''
    subdir = '_'.join(
        [str(i) for i in [N, lr, epochs, proj, hidden, num_layers, opt, step_size]])
    print(subdir)
    if log_dir:
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
            # writer.add_scalar('lr', get_lr(optimizer), iteration)
            embeddings = embedder_net(mel_db)
            sim_matrix = criterion(embeddings)
            # get acc_val

            loss, per_embedding_loss = calc_loss(sim_matrix)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            acc = get_acc(sim_matrix, N, M)
            if log_dir:
                writer.add_scalar('batch loss', loss, iteration)
                writer.add_scalar('accuracy', acc, iteration)
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
            
            if log_dir:
                writer.add_text('EER', f"{EER}: thres: {EER_thresh:.2f}, FAR:{EER_FAR:.2f}, FRR: {EER_FRR:.2f}")
                writer.add_scalar('FAR', EER_FAR, iteration)
                writer.add_scalar('FRR', EER_FRR, iteration)
                writer.add_scalar('EER', EER, iteration)
                writer.add_scalar('BATCH EER', batch_avg_EER, iteration)
                writer.add_scalar('epoch loss', epoch_loss, epoch)
    if save_model is True:
        if not os.path.exists('./models'):
            os.makedirs('./models')
        save_model_path = os.path.join('models', subdir+'transfer_model')
        torch.save(embedder_net.state_dict(), save_model_path)
        return save_model_path


if __name__ == '__main__':
    train(sys.argv[1])
