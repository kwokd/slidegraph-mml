import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Sequential, Linear, ReLU,Tanh,LeakyReLU,ELU,SELU,GELU
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, Sampler

from torchinfo import summary

############
# Omic Model
############
class MaxNet(nn.Module):
    def __init__(self, 
        input_dim=80, 
        omic_dim=32,
        dropout_rate=0.25, 
        act=None, 
        label_dim=1, 
        init_max=True
        ):
        super(MaxNet, self).__init__()

        hidden = [64, 48, 32, 32]
        self.act = act
        
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
            nn.Linear(hidden[2], omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))
        
        if init_max: init_max_weights(self)
        
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, x):
        # x comes in as one long vector 
        # instead of width 95
        # cannot solve the batching problem. just leave the reshape as is
        omics_in = x.reshape([-1,OMICS_LEN])
        
        features = self.encoder(omics_in)
        out = self.classifier(features)
        if self.act is not None:
            out = self.act(out)
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift
        
        return features, out