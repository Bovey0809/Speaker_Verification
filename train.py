import torch
import os
from torch.optim.lr_scheduler import StepLR
from data_load import SpeakerDatasetTIMITPreprocessed
from embedder_net import SpeechEmbedder, GE2ELoss
from time import ctime, time
from torch.utils.data import DataLoader


# add timing for train
def logging(f):
    def _f(**kargs):
        print(f"PARMS: {kargs}")
        current = time()
        result = f(**kargs)
        print(f"TIME: {(time() - current):.3f} seconds")
        return result
    return _f


@logging
def train(N=16, lr=0.01, epochs=10, proj=256, hidden=768, num_layers=3, opt='SGD', debug=False, schedule=True, step_size=2e3):
    # define net, loss, optimizer
    device = torch.device('cuda')
    N = N
    M = 6
    embedder_net = SpeechEmbedder(
        hidden=hidden, num_layers=num_layers, proj=proj, N=N, M=M)
    embedder_net.to(device)
    criterion = GE2ELoss(device)
    if opt == 'Adam':
        optimizer = torch.optim.Adam([{'params': embedder_net.parameters()},
                                      {'params': criterion.parameters()}], lr)
    elif opt == 'SGD':
        optimizer = torch.optim.SGD([{'params': embedder_net.parameters()},
                                     {'params': criterion.parameters()}], lr)

    # schedule
    if schedule:
        scheduler = StepLR(optimizer, step_size, gamma=0.5)
    train_dataset = SpeakerDatasetTIMITPreprocessed('./train_tisv/', M)
    train_loader = DataLoader(
        train_dataset, batch_size=N, drop_last=True, shuffle=True)

    # train network
    total_length = len(train_dataset)
    loss_log = []
    iteration = 0
    for e in range(epochs):
        if schedule:
            scheduler.step()
        epoch_loss = 0
        for batch_id, mel_db in enumerate(train_loader):
            mel_db = mel_db.to(device)
            mel_db = mel_db.flatten(0, 1)
            optimizer.zero_grad()
            embeddings = embedder_net(mel_db)
            loss = criterion(embeddings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            if iteration % 100 == 0:
                mesg = f"{ctime()}\tEpoch:{e} BATCH:{batch_id}/{total_length//N}\tLoss: {loss:.4f}\t TLoss:{epoch_loss/(batch_id+1):.4f}"
                print(mesg)
            iteration += 1
        loss_log.append(epoch_loss)
    if not os.path.exists('./models'):
        os.makedirs('./models')
    save_model_path = f'./models/lr{lr}epochs{epochs}proj{proj}hidden{hidden}num_layers{num_layers}opt{opt}.model'
    torch.save(embedder_net.state_dict(), save_model_path)
    return save_model_path, loss_log
