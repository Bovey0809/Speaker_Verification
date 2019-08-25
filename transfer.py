import sys
import os
import torch
from torch.utils.data import DataLoader
from train import train


from embedder_net import SpeechEmbedder, GE2ELoss

data_path = ""
model_path = "./models/128_0.0001_2000_128_768_3_Adam_10000000.0transfer_model"

model = SpeechEmbedder(hidden=32, num_layers=3, proj=64)
optimizer = GE2ELoss()

checkpoint = torch.load(model_path)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# for param in model.parameters():
#    param.requires_grad = False

dataloader = 

