import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Sequential, Linear, ReLU,Tanh,LeakyReLU,ELU,SELU,GELU
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, Sampler

from torchinfo import summary


############
# Fusion
############
class SimpleFusion(nn.Module):
    def __init__(self, dim1=32, dim2=32, dim_out=64, bilinear=True, cat_only=False):
        super(SimpleFusion,self).__init__()
        self.bilinear = bilinear
        self.cat_only = cat_only
        if self.bilinear:
            self.fuse = nn.Bilinear(dim1, dim2, dim_out)
            self.relu = nn.ReLU()
        else:
            self.fuse = nn.Sequential(nn.Linear(dim1+dim2, dim_out), nn.ReLU())
        
    def forward(self, vec1, vec2):
        if self.bilinear:
            return self.relu(self.fuse(vec1,vec2))
        else:
            conc = torch.cat((vec1,vec2),dim=1)
            if self.cat_only: 
                return conc
            else: 
                return self.fuse(conc)

class BilinearFusion(nn.Module):
    def __init__(self, 
        skip=True, use_bilinear=True, 
        gate1=True, gate2=True, 
        dim1=32, dim2=32, 
        scale_dim1=1, scale_dim2=1, 
        out_dim=64, 
        dropout_rate=0.25
        ):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        
        dim1_og, dim2_og = dim1, dim2
        dim1, dim2 = dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1+dim2+2 if skip else 0
        
        # for graph vec
        self.linear_h1 = nn.Sequential(
            nn.Linear(dim1_og, dim1), 
            nn.ReLU())
        
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else 
            nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1)))
        
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), 
            nn.ReLU(), 
            nn.Dropout(p=dropout_rate))
        
        # for omics vec
        self.linear_h2 = nn.Sequential(
            nn.Linear(dim2_og, dim2), 
            nn.ReLU())
        
        self.linear_z2 = (
            nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else 
            nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2)))
        
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), 
            nn.ReLU(), 
            nn.Dropout(p=dropout_rate))
        
        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1+1)*(dim2+1), out_dim), 
            nn.ReLU(), 
            nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(
            nn.Linear(out_dim+skip_dim, out_dim), 
            nn.ReLU(), 
            nn.Dropout(p=dropout_rate))
        init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        # h: rescale vec1/vec2 by scale_dim, pass relu
        # z: bilinear/linear by concat with both vec input, output to be same size as corresponding h
        # o: sigmoid of z multiplied by h as input to fc layer

        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = (
                self.linear_z1(vec1, vec2) if self.use_bilinear else
                self.linear_z1(torch.cat((vec1, vec2), dim=1)))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)
        
        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec1, vec2) if self.use_bilinear else 
                self.linear_z2(torch.cat((vec1, vec2), dim=1)))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)
        
        ### Fusion
        # append 1 to the end of each feature vector in the batch
        o1 = torch.cat((o1,torch.ones(o1.shape[0],1,dtype=torch.float, device='cuda')),1)
        o2 = torch.cat((o2,torch.ones(o2.shape[0],1,dtype=torch.float, device='cuda')),1)
        
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        
        if self.skip: out = torch.cat((out, o1, o2), 1)
        
        out = self.encoder2(out)
        return out