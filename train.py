import sys
import os
import torch
from torch.optim.lr_scheduler import StepLR
from data_load import SpeakerDatasetTIMITPreprocessed
from embedder_net import SpeechEmbedder, GE2ELoss
from torch.utils.data import DataLoader
from utils import get_cossim, calc_loss, get_acc, get_eer
from torch.utils.tensorboard import SummaryWriter


def train(dataset, log_dir='log_acc', N=64, lr=0.0001, epochs=2000, proj=512, hidden=768, num_layers=3, opt='Adam', step_size=2e3, save_model=True):
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

    phrases = ['train_tisv', 'test_tisv']
    datasets = {x: SpeakerDatasetTIMITPreprocessed(
        os.path.join(dataset, x)) for x in phrases}
    dataloaders = {x: DataLoader(datasets[x], N, drop_last=True, shuffle=True)
                   for x in phrases}
    # train_dataset = SpeakerDatasetTIMITPreprocessed(dataset, M)
    # train_loader = DataLoader(
    #     train_dataset, N, drop_last=True, shuffle=True, num_workers=0)

    iters = 0

    for epoch in range(epochs):
        for phrase in phrases:
            if phrase == 'train_tisv':
                if opt == 'SGD':
                    scheduler.step()
                    epoch_loss = 0
                embedder_net.train()
                criterion.train()
            else:
                embedder_net.eval()
                criterion.eval()
            for batch_id, mel_db in enumerate(dataloaders[phrase]):
                mel_db = mel_db.to(device)
                mel_db = mel_db.flatten(0, 1)
                
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phrase == 'train_tisv'):
                    embeddings = embedder_net(mel_db)
                    sim_matrix = criterion(embeddings)
                    loss, per_embedding_loss = calc_loss(sim_matrix)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        embedder_net.parameters(), 3.0)
                    torch.nn.utils.clip_grad_norm_(
                        criterion.parameters(), 1.0)
                    optimizer.step()
                epoch_loss += loss.item()
                acc = get_acc(sim_matrix, N, M)
                EER_thresh, EER, EER_FAR, EER_FRR = get_eer(
                    embeddings, N, M)
                iters += 1

                if log_dir:
                    writer.add_scalar('batch loss', loss, iters)
                    writer.add_scalar('accuracy', acc, iters)
                    writer.add_scalar('FAR', EER_FAR, iters)
                    writer.add_scalar('FRR', EER_FRR, iters)
                    writer.add_scalar('EER', EER, iters)
                    writer.add_scalar('epoch loss', epoch_loss, epoch)
            if save_model is True:
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                    save_model_path = os.path.join(
                        'models', subdir+'transfer_model')
                    torch.save(embedder_net.state_dict(), save_model_path)
        # validation
        else:

    return save_model_path


if __name__ == '__main__':
    train(sys.argv[1])
