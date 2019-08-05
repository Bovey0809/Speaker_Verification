import sys
import os
import torch
from torch.optim.lr_scheduler import StepLR
from data_load import SpeakerDatasetTIMITPreprocessed
from embedder_net import SpeechEmbedder, GE2ELoss
from torch.utils.data import DataLoader
from utils import calc_loss, get_acc, get_eer
from torch.utils.tensorboard import SummaryWriter


def train(dataset, log_dir='test_vad', N=64, lr=0.0001, epochs=200, proj=512, hidden=768, num_layers=3, opt='Adam', step_size=2e3, save_model=False):
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
    writer.add_text('dataset_dir', f'{dataset}')
    writer.add_text('log_dir', f'{log_dir}')
    device = torch.device('cuda')
    N = int(N)
    M = 6
    embedder_net = SpeechEmbedder(
        hidden=hidden, num_layers=num_layers, proj=proj)
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
        os.path.join(dataset, x), M) for x in phrases}
    test_N = len(datasets['test_tisv'])
    dataloaders = {
        'train_tisv': DataLoader(
            datasets['train_tisv'], N, drop_last=True, shuffle=True),
        'test_tisv': DataLoader(
            datasets['test_tisv'], test_N, drop_last=True, shuffle=True
        )}

    # train_dataset = SpeakerDatasetTIMnnITPreprocessed(dataset, M)
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
                    if phrase == 'train_tisv':
                        embeddings = embeddings.view(N, M, proj)
                    else:
                        embeddings = embeddings.view(test_N, M, proj)
                    sim_matrix = criterion(embeddings)
                    loss, per_embedding_loss = calc_loss(sim_matrix)
                    if phrase == 'train_tisv':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            embedder_net.parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(
                            criterion.parameters(), 1.0)
                        optimizer.step()
                if phrase == 'train_tisv':
                    acc = get_acc(sim_matrix, N, M)
                    EER_thresh, EER, EER_FAR, EER_FRR = get_eer(
                        embeddings, N, M)
                    epoch_loss += loss.item()
                else:
                    vad_acc = get_acc(sim_matrix, test_N, M)
                    _, vad_EER, _, _ = get_eer(
                        embeddings, test_N, M)
                iters += 1

                if log_dir:
                    if phrase == 'train_tisv':
                        writer.add_scalar('batch loss', loss, iters)
                        writer.add_scalar('accuracy', acc, iters)
                        writer.add_scalar('FAR', EER_FAR, iters)
                        writer.add_scalar('FRR', EER_FRR, iters)
                        writer.add_scalar('EER', EER, iters)
                        writer.add_scalar('epoch_loss', epoch_loss, epoch)
                    else:
                        writer.add_scalar('vad acc', vad_acc, epoch)
                        writer.add_scalar('vad EER', vad_EER, epoch)
            if save_model is True:
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                    save_model_path = os.path.join(
                        'models', subdir+'transfer_model')
                    torch.save(embedder_net.state_dict(), save_model_path)


if __name__ == '__main__':
    train(sys.argv[1])
