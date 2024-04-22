############
# Omic Model
############

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Sequential, Linear, ReLU,Tanh,LeakyReLU,ELU,SELU,GELU
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, Sampler
from torch.nn import Parameter

# batched, so omics tensor is concatenated to single long vector. 
# must reshape after passing to net
class MaxNet(nn.Module):
    def __init__(self, 
        input_dim=95,
        out_dim=32,
        dropout_rate=0.25,
        init_max=True
        ):
        super(MaxNet, self).__init__()

        self.input_dim = input_dim

        hidden = [64, 48, 32, 32]
        
        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], out_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        
        # if init_max: init_max_weights(self)

    def forward(self, x):
        
        # x comes in as one long vector instead of width 95
        # cannot solve the batching problem. just leave the reshape as is
        omics_in = x.reshape([-1,self.input_dim])
        
        return self.encoder(omics_in)