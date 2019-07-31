import torch
from torch import nn
from torch.nn import functional as F
from utils import get_centroids, get_cossim


class SpeechEmbedder(nn.Module):

    def __init__(self, mels=40, hidden=128, num_layers=3, proj=64, N=4, M=6):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(
            mels, hidden, num_layers=num_layers, batch_first=True, dropout=0.5)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:  # 把bias初始化为0
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:  # weight用xavier初始化
                nn.init.xavier_normal_(param)  # 这个说不定可以改变用uniform之类的
        self.projection = nn.Linear(hidden, proj)
        self.n = N
        self.m = M
        self.proj = proj

    def forward(self, x):
        # (batch, frames, n_mels) 这个是由于使用了batch_first=True
        x, _ = self.LSTM_stack(x.float())
        # only use last frame
        x = x[:, -1]  # [batch, hidden]
        x = self.projection(x)  # [batch, projection]
        x = F.normalize(x)  # x.shape == [batch, projection]
        x = torch.reshape(x, (self.n, self.m, self.proj))
        # This line is different from original
        return x


class GE2ELoss(nn.Module):

    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(
            10.0, device=device), requires_grad=True)
        self.b = nn.Parameter(
            torch.tensor(-5.0, device=device), requires_grad=True)

    def forward(self, X, **kwargs):
        torch.clamp(self.w, 1e-6)  # 论文里面是0, 代码用的1e-7
        centroids = get_centroids(X)
        cossim = get_cossim(X, centroids)
        sim_matrix = self.w*cossim + self.b
        return sim_matrix
        # loss, _ = calc_loss(sim_matrix)
        # return loss
