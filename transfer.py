import sys
import os
import torch
from torch.utils.data import DataLoader
from data_load import SpeakerDatasetTIMITPreprocessed
from embedder_net import SpeechEmbedder, GE2ELoss
from utils import calc_loss, get_acc, get_eer
from torch.utils.tensorboard import SummaryWriter


def transfer(model_path, data_path, log_dir, epochs=200, N=64, M=6):
    '''
    mode: freeze or finetuning
    '''
    # LOG
    info = f'{model_path}_{data_path}_{N}'
    writer = SummaryWriter(f'./{log_dir}{info}')
    # LOAD MODEL
    device = torch.device('cuda')
    checkpoint = torch.load(model_path)
    hidden, num_layers, proj = checkpoint['hyper_parameters']
    embedder_net = SpeechEmbedder(40, hidden, num_layers, proj).to(device)
    criterion = GE2ELoss(device)
    optimizer = torch.optim.Adam([{'params': embedder_net.parameters()},
                                  {'params': criterion.parameters()}],
                                 0.0001,
                                 (0.5, 0.9))

    embedder_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # LOAD DATA
    phases = ['train_tisv', 'test_tisv']
    datasets = {x: SpeakerDatasetTIMITPreprocessed(
        os.path.join(data_path, x), M) for x in phases}
    test_N = len(datasets['test_tisv'])
    dataloaders = {
        'train_tisv': DataLoader(
            datasets['train_tisv'], N, drop_last=True, shuffle=True),
        'test_tisv': DataLoader(
            datasets['test_tisv'], test_N, drop_last=True, shuffle=True
        )}

    # TRAIN
    iters = 0
    for epoch in range(epochs):
        for phase in phases:
            if phase == 'train_tisv':
                epoch_loss = 0
                embedder_net.train()
                criterion.train()
            else:
                embedder_net.eval()
                criterion.eval()
            for batch_id, mel_db in enumerate(dataloaders[phase]):
                mel_db = mel_db.to(device)
                mel_db = mel_db.flatten(0, 1)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train_tisv'):
                    embeddings = embedder_net(mel_db)

                    if phase == 'train_tisv':
                        embeddings = embeddings.view(N, M, proj)
                    else:
                        embeddings = embeddings.view(test_N, M, proj)
                        vad_sim_matrix = criterion(embeddings)
                        vad_loss, _ = calc_loss(vad_sim_matrix)
                    sim_matrix = criterion(embeddings)
                    loss, per_embedding_loss = calc_loss(sim_matrix)
                    if phase == 'train_tisv':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            embedder_net.parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(
                            criterion.parameters(), 1.0)
                        optimizer.step()
                if phase == 'train_tisv':
                    acc = get_acc(sim_matrix, N, M)
                    EER_thresh, EER, EER_FAR, EER_FRR = get_eer(
                        embeddings, N, M)
                    epoch_loss += loss.item()
                else:
                    vad_acc = get_acc(sim_matrix, test_N, M)
                    _, vad_EER, _, _ = get_eer(
                        embeddings, test_N, M)
                iters += 1
                if phase == 'train_tisv':
                    writer.add_scalar('batch loss', loss, iters)
                    writer.add_scalar('accuracy', acc, iters)
                    writer.add_scalar('FAR', EER_FAR, iters)
                    writer.add_scalar('FRR', EER_FRR, iters)
                    writer.add_scalar('EER', EER, iters)
                    writer.add_scalar('epoch_loss', epoch_loss, epoch)
                else:
                    writer.add_scalar('vad acc', vad_acc, epoch)
                    writer.add_scalar('vad EER', vad_EER, epoch)
                    writer.add_scalar('vad loss', vad_loss, epoch)


if __name__ == '__main__':
    data_path = "./preprocessed_company_data"
    # model_path = "./models/64_0.0001_10_512_32_3_Adam_2000.0transfer_model"
    # transfer(model_path, data_path)
    transfer(sys.argv[1], sys.argv[2], sys.argv[3], epochs=50000)
