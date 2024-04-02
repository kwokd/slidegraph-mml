# %%
## try torchinfo

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, Sampler
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from copy import deepcopy
from numpy.random import randn 
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU,Tanh,LeakyReLU,ELU,SELU,GELU
from torch_geometric.nn import GINConv,EdgeConv, PNAConv,DynamicEdgeConv,global_add_pool, global_mean_pool, global_max_pool
import time
from tqdm import tqdm
from scipy.spatial import distance_matrix, Delaunay
import random
from torch_geometric.data import Data, DataLoader
import pickle
from glob import glob
import os
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.model_selection import StratifiedKFold
import pdb
from statistics import mean, stdev
from glob import glob
import os
import pandas as pd
import numpy as np
import pickle
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.model_selection import StratifiedKFold, train_test_split
import math
from random import shuffle
from itertools import islice
from lifelines.utils import concordance_index as cindex
from lifelines import KaplanMeierFitter
from sklearn.model_selection import StratifiedShuffleSplit
from collections import OrderedDict
import re

# %% [markdown]
# Set up the variables:

# %%
LEARNING_RATE = 0.00002
WEIGHT_DECAY = 0.005
L1_WEIGHT = 0.001
SCHEDULER = None
BATCH_SIZE = 10
NUM_BATCHES = 2000
NUM_LOGS = 150 # How many times in training the loss value is stored

#Select what feature set to use
SHUFFLE_NET = True
# SHUFFLE_NET = False

VALIDATION = True
NORMALIZE = False
CENSORING = True
FRAC_TRAIN = 0.8
CONCORD_TRACK = True
FILTER_TRIPLE = False
EARLY_STOPPING = True
MODEL_PATH = 'Best_model/'
VARIABLES = 'DSS'
TIME_VAR = VARIABLES + '.time'
ON_GPU = True
USE_CUDA = torch.cuda.is_available()
rng = np.random.default_rng()
device = {True:'cuda:0',False:'cpu'}[USE_CUDA] 

bdir = r'./graphs_json/'
        # Set up directory for on disk dataset
directory = r'./graphs_pkl'
OMICS_SIZE = 95

# %% [markdown]
# Accessory methods:

# %%
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v

def toTensor(v,dtype = torch.float,requires_grad = True):
    return torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad)

def toTensorGPU(v,dtype = torch.float,requires_grad = True):
    return cuda(torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad))

def toNumpy(v):
    if type(v) is not torch.Tensor: return np.asarray(v)
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()

def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)
    
def toGeometric(Gb,y,tt=1e-3):
    return Data(x=Gb.x, edge_index=(Gb.get(W)>tt).nonzero().t().contiguous(),y=y)

def toGeometricWW(X,W,y,tt=0):    
    return Data(x=toTensor(X,requires_grad = False), edge_index=(toTensor(W,requires_grad = False)>tt).nonzero().t().contiguous(),y=toTensor([y],dtype=torch.long,requires_grad = False))

def pair_find(graphs,features):
    indexes = []
    for j in range(len(graphs)):
        graph_j = graphs[j]
        if features == 'BRCA-SHUFFLE':
            event_j = graph_j[1][0]
            time_j = graph_j[1][1]
        else:
            event_j, time_j = graph_j.event, graph_j.e_time
        if event_j == 1:
            for i in range(len(graphs)): 
                graph_i = graphs[i]            
                if features == 'BRCA-SHUFFLE':
                    time_i = graph_i[1][1]
                else:
                    time_i = graph_i.e_time
                if graph_j != graph_i and time_i > time_j:
                    indexes.append((i,j))
    shuffle(indexes)
    return indexes

def SplitBrcaData(dataset, numSplits, isShuffle, testSize):
    if isShuffle:
        eventVars = [dataset[i][1][0] for i in range(len(dataset))]
    else:
        eventVars = [int(dataset[i].event.detach().numpy()) for i in range(len(dataset))]  
    x = np.zeros(len(dataset))
    shuffleSplit = StratifiedShuffleSplit(n_splits = numSplits, test_size = testSize)
    return shuffleSplit.split(x,eventVars)

def disk_graph_load(batch):
    # takes list of patient codes, returns list of graph Data objects
    return [torch.load(directory + '/' + graph + '.g') for graph in batch]

def get_predictions(model,graphs,features = 'BRCA-CC',device=torch.device('cuda:0')) -> list:
    outputs = []
    e_and_t = []
    model.eval()
    with torch.no_grad():
        for i in range(len(graphs)):
            graph = graphs[i]
            if features == 'BRCA-SHUFFLE':
                tag = [graph[0]]
                temp = [graph[1][0], graph[1][1]]
                graph = disk_graph_load(tag)
            else:
                temp = [graph.event.item(),graph.e_time.item()]
                graph = [graph]
            size = 1
            loader = DataLoader(graph, batch_size=size)
            for d in loader:
                d = d.to(device)
            # z,_,_ = model(d)
            z,_ = model(d)
            z = toNumpy(z)
            outputs.append(z[0][0])
            e_and_t.append(temp)
    return outputs, e_and_t

import ujson as json
from pathlib import Path
def loadfromjson(graph):
    with Path(graph).open() as fptr:    
        graph_dict = json.load(fptr)
        
    graph_dict = {k: np.array(v) for k, v in graph_dict.items()}
    graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
    return graph_dict

# %% [markdown]
# GNN architecture

# %%
class GNN(torch.nn.Module):
    """
    Parameters
    ----------
    dim_features : TYPE Int
        DESCRIPTION. Number of features of each node
    dim_target : TYPE Int
        DESCRIPTION. Number of outputs
    layers : TYPE, optional List of number of nodes in each layer
        DESCRIPTION. The default is [6,6].
    pooling : TYPE, optional
        DESCRIPTION. The default is 'max'.
    dropout : TYPE, optional
        DESCRIPTION. The default is 0.0.
    conv : TYPE, optional Layer type string {'GINConv','EdgeConv'} supported
        DESCRIPTION. The default is 'GINConv'.
    gembed : TYPE, optional Graph Embedding
        DESCRIPTION. The default is False. Pool node scores or pool node features
    **kwargs : TYPE
        DESCRIPTION.
    Raises
    ------
    NotImplementedError
        DESCRIPTION.
    Returns
    -------
    None.
    """
    def __init__(
        self,
        dim_features,
        dim_target,
        layers=[16,16,8],
        pooling='max',
        dropout = 0.0,
        conv='GINConv',
        gembed=False,
        **kwargs
        ) -> None:
        super(GNN, self).__init__()
        self.dropout = dropout
        self.embeddings_dim=layers
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.pooling = {
            'max':global_max_pool,
            'mean':global_mean_pool,
            'add':global_add_pool
        }[pooling]
        self.gembed = gembed 
        #if True then learn graph embedding for final classification (classify pooled node features) otherwise pool node decision scores

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            # shufflenet = 1024 features
            # features -> (0,16) (1,16) (2,8) -> target
            if layer == 0:
                #first layer
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim),GELU())
                self.linears.append(Sequential(Linear(out_emb_dim, dim_target),GELU()))
                
            else:
                #subsequent layers
                #input embedding dimension is output of previous layer
                input_emb_dim = self.embeddings_dim[layer-1]
                self.linears.append(Linear(out_emb_dim, dim_target))
                subnet = Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim))              
                if conv=='GINConv':
                    self.nns.append(subnet)
                    self.convs.append(GINConv(self.nns[-1], **kwargs))  # Eq. 4.2 eps=100, train_eps=False
                elif conv=='EdgeConv':
                    subnet = Sequential(Linear(2*input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim))
                    self.nns.append(subnet)                    
                    self.convs.append(EdgeConv(self.nns[-1],**kwargs))#DynamicEdgeConv#EdgeConv  aggr='mean'
                else:
                    raise NotImplementedError  
                    
        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input
        
    def forward(self, data) -> torch.tensor:
        x, edge_index, batch, pooling = data.x, data.edge_index, data.batch, self.pooling
        out = 0
        Z = 0
        # Z is the result of linear layers
        import torch.nn.functional as F
        for layer in range(self.no_layers):            
            if layer == 0:
                x = self.first_h(x)
                z = self.linears[layer](x)
                Z+=z
                dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                out += dout
            else:
                x = self.convs[layer-1](x,edge_index)
                if not self.gembed:
                    z = self.linears[layer](x)
                    Z+=z
                    dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                else:
                    dout = F.dropout(self.linears[layer](pooling(x, batch)), p=self.dropout, training=self.training)
                out += dout

        return out,Z,x

# %% [markdown]
# ~TODO: Multimodal fusion

# %% [markdown]
# Multimodal utils

# %%
def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)

# %% [markdown]
# Bilinear fusion

# %%
class BilinearFusion(nn.Module):
    def __init__(self, 
        skip=1, 
        use_bilinear=1, 
        gate1=1, 
        gate2=1, 
        dim1=32, 
        dim2=32, 
        scale_dim1=1, 
        scale_dim2=1, 
        mmhid=64, 
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

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, o1, o2), 1)
        out = self.encoder2(out)
        return out

# %% [markdown]
# MaxNet

# %%

from torch.nn import Parameter
############
# Omic Model
############
class MaxNet(nn.Module):
    def __init__(self, 
        input_dim=80, 
        omic_dim=32, 
        #^output dim; i.e. transform input 80 to output 32?
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

    def forward(self, data):
        x = data
        # print(x)
        # print(x.shape)
        features = self.encoder(x)
        ## problem is here
        out = self.classifier(features)
        if self.act is not None:
            out = self.act(out)

            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out

# %% [markdown]
# Graphomic Net

# %%
##############################################################################
# Graph + Omic
##############################################################################
class GraphomicNet(nn.Module):
    def __init__(self, act=None):
        super(GraphomicNet, self).__init__()
        
        # Set up model and optimizer
        self.grph_net = GNN(
            dim_features=1024,
            # G.x.shape[1]
            dim_target = 32,
            layers = [64,48,32,32],
            dropout = 0.0, 
            pooling = 'mean', 
            conv='EdgeConv', 
            aggr = 'max'
            ).to(device)
        
        self.omic_net = MaxNet(
            input_dim=OMICS_SIZE, 
            omic_dim=32, 
            #^output dim; i.e. transform input 95 to output 32
            dropout_rate=0.25, 
            act=None, 
            label_dim=1, 
            init_max=True
            ).to(device)
        
        self.fusion = BilinearFusion()
        # self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.classifier = nn.Sequential(nn.Linear(64, 1))
        self.act = act

        # dfs_freeze(self.grph_net)
        # dfs_freeze(self.omic_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, data):
        # expects as input a x_grph and x_omic
        grph_vec, _, _ = self.grph_net(data)
        omics_input = data.omics.reshape([-1,OMICS_SIZE])
        
        omic_vec, _ = self.omic_net(omics_input)
        features = self.fusion(grph_vec, omic_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return hazard, features

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False

# %% [markdown]
# Wrapper for GNN

# %%
class NetWrapper:
    def __init__(self, device='cuda:0',features='BRCA-CC') -> None:
        self.model = GraphomicNet().to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(),
                                lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)
        
        self.device = torch.device(device)
        self.features = features

    def loss_fn(self,batch,optimizer) -> float:
        unzipped = [j for pair in batch for j in pair]
        # batch is list of batch pairs, this flattens list
        
        # This can be changed when using a system with large RAM
        if self.features == 'BRCA-SHUFFLE':
            graph_set = list(set(unzipped))
            graphs = disk_graph_load(graph_set)
        else:
            graph_set = unzipped
            graphs = graph_set
        unzipped = None
            
        batch_load = DataLoader(graphs, batch_size = len(graphs), follow_batch=['omics'])
        
        for data in batch_load: 
            data = data.to(self.device)
            
        z = toTensorGPU(0)
        loss = 0
        ######################### edit to use graphomicnet VVV
        
        self.model.train()
        optimizer.zero_grad()
        
        output,_ = self.model(data)
        
        num_pairs = len(batch)
        for (xi,xj) in batch:
            graph_i, graph_j = graph_set.index(xi), graph_set.index(xj)
            # Compute loss function
            dz = output[graph_i] - output[graph_j]
            loss += torch.max(z, 1.0 - dz)
        loss = loss/num_pairs
        
        loss.backward()
        optimizer.step()
        
        ########################### ^^^
        
        return loss.item()

    def validation_loss_and_Cindex_eval(self,graphs,pairs) -> float:
        tot_loss = 0
        print('Number of Validation Pairs: ' + str(len(pairs)))
        predictions, e_and_t = get_predictions(self.model,graphs,self.features)
        for j in range(len(pairs)):
            p_graph_i = predictions[pairs[j][0]]
            p_graph_j = predictions[pairs[j][1]]
            dz = p_graph_i - p_graph_j
            loss = max(0, 1.0 - dz)
            tot_loss += loss
        epoch_val_loss = tot_loss / len(pairs)
        T = [x[1] for x in e_and_t]
        E = [x[0] for x in e_and_t]
        concord = cindex(T,predictions,E)
        return epoch_val_loss, concord

    def censor_data(self,graphs, censor_time): # The censor time measured in years
        cen_time = 365 * censor_time
        for graph in graphs:
            if self.features == 'BRCA-SHUFFLE':
                time = graph[1][1]
            else:
                time = graph.e_time
            if time > cen_time:
                if self.features == 'BRCA-SHUFFLE':
                    graph[1] = (0,cen_time)
                else:
                    graph.event = toTensor(0)
                    graph.e_time = toTensor(cen_time)
            else:
                continue
        return graphs

    def train(self,training_data,validation_data,max_batches=500,num_logs=50,
              early_stopping = 10,return_best = False,batch_size = 10) -> float:
        # format of training_data: [TAG,(event,event_time)]
        
        return_best = return_best and validation_data is not None
        log_interval = max_batches // num_logs
        
        loss_vals = { 'train': [], 'validation': [] }
        
        concords = []
        c_best = 0.5
        best_batch = 1000
        patience = early_stopping
        
        training_indexes = pair_find(training_data,self.features)
        # list of tuples i,j: indices of graphs where time(i) > time(j)
        
        print("Number of batches used for training "+ str(max_batches))
        print('Num Pairs: ' + str(len(training_indexes)))
        
        counter = 0 # To resolve list index errors with large NUM_BATCHES vals
        
        for i in tqdm(range(1,max_batches + 1)):
            if counter < len(training_indexes) - batch_size:
                batch_pairs = []
                index_pairs = training_indexes[counter:counter+batch_size]
                for j in range(len(index_pairs)):
                    if self.features == 'BRCA-SHUFFLE':
                        graph_i = training_data[index_pairs[j][0]][0]
                        graph_j = training_data[index_pairs[j][1]][0]
                    else:
                        graph_i = training_data[index_pairs[j][0]]
                        graph_j = training_data[index_pairs[j][1]]
                    batch_pairs.append((graph_i,graph_j))
                loss = self.loss_fn(batch_pairs,self.optimizer)
                counter += batch_size
            else:
                counter = 0
                
            loss_vals['train'].append(loss)
                        
        return loss_vals, concords, self.model

# %% [markdown]
# Evaluator

# %%
class Evaluator:
    def __init__(self, model, device='cuda:0',features = 'BRCA-CC') -> None:
        self.model = model
        self.device = device
        self.features = features

    def get_predictions(self,model,graphs,device=torch.device('cuda:0')) -> list:
        outputs = []
        e_and_t = []
        model.eval()
        with torch.no_grad():
            for i in range(len(graphs)):
                graph = graphs[i]
                if self.features == 'BRCA-SHUFFLE':
                    tag = [graph[0]]
                    temp = [graph[1], graph[1]]
                    graph = disk_graph_load(tag)
                else:
                    temp = [graph.event.item(),graph.e_time.item()]
                    graph = [graph]
                size = 1
                loader = DataLoader(graph, batch_size=size)
                for d in loader:
                    d = d.to(device)
                z,_,_ = model(d)
                z = toNumpy(z)
                outputs.append(z[0])
                e_and_t.append(temp)
        return outputs, e_and_t
    
    def test_evaluation(self,testDataset):
        predictions, e_and_t = get_predictions(self.model,testDataset,self.features)
        T = [x[1] for x in e_and_t]
        E = [x[0] for x in e_and_t]
        concord = cindex(T,predictions,E)
        return concord
    
    def K_M_Curves(self, graphs, split_val, mode = 'Train') -> None:
        outputs, e_and_t = get_predictions(self.model,graphs,self.features)
        T = [x[1] for x in e_and_t]
        E = [x[0] for x in e_and_t]
        mid = np.median(outputs)
        if mode != 'Train':
            if split_val > 0:
                mid = split_val
        else:
            print(mid)
        T_high = []
        T_low = []
        E_high = [] 
        E_low = []
        for i in range(len(outputs)):
          if outputs[i] <= mid:
            T_high.append(T[i])
            E_high.append(E[i])
          else:
            T_low.append(T[i])
            E_low.append(E[i])
        km_high = KaplanMeierFitter()
        km_low = KaplanMeierFitter()
        ax = plt.subplot(111)
        ax = km_high.fit(T_high, event_observed=E_high, label = 'High').plot_survival_function(ax=ax)
        ax = km_low.fit(T_low, event_observed=E_low, label = 'Low').plot_survival_function(ax=ax)
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(km_high, km_low, ax=ax)
        plt.title('Kaplan-Meier estimate')
        plt.ylabel('Survival probability')
        plt.show()
        plt.tight_layout()
        from lifelines.statistics import logrank_test
        results = logrank_test(T_low, T_high, E_low, E_high)
        print("p-value %s; log-rank %s" % (results.p_value, np.round(results.test_statistic, 6)))

# %% [markdown]
# TODO: Genomic data import

# %%
supp_file = './NIHMS978596-supplement-1.xlsx'
label = pd.read_excel(supp_file).rename(columns={'bcr_patient_barcode':'PATIENT'}).set_index('PATIENT')
label = label[["DSS","DSS.time"]]
# label['DSS'] = label['DSS'].astype('bool')

import os
from natsort import natsorted
clini_file = './TCGA-BRCA-DX_CLINI (8).xlsx'
df = pd.read_excel(clini_file).set_index('PATIENT')  # path to clinical file
print(df.shape)

# one-hot for each mutation value
mut = pd.get_dummies(df.filter(regex="_mutation$"),dtype=float)

# log1p all expression values
expr = df.filter(regex="_expression$").apply(lambda x: [np.log1p(item) for item in x])

# standard scale across all cnv values
from sklearn import preprocessing
cnv = df.filter(regex="_CNV$|ZNF703")
scaler = preprocessing.StandardScaler()
scaler.fit(cnv)
scaled = scaler.fit_transform(cnv)
df = pd.DataFrame(scaled, columns=cnv.columns, index=cnv.index)

df = df.join(expr).join(mut).join(label,"PATIENT","inner").dropna()
df = df.drop(columns=["DSS","DSS.time"])
print(df.shape)

# %%
omtest = toTensor(df.loc['TCGA-3C-AALJ'])

# %%
SAVE_SHUFFLENET = True

# %%
class OmicGraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'omics':
            return OMICS_SIZE
        return super().__inc__(key, value, *args, **kwargs)

# %%
if __name__ == '__main__':
    device = {True:'cuda:0',False:'cpu'}[USE_CUDA] 
    import pandas as pd
    import os
    from natsort import natsorted
    survival_file = r'./NIHMS978596-supplement-1.xlsx'
    cols2read = [VARIABLES,TIME_VAR]
    TS = pd.read_excel(survival_file).rename(columns= {'bcr_patient_barcode':'ID'}).set_index('ID')  # path to clinical file
    TS = TS[cols2read][TS.type == 'BRCA']
    print(TS.shape)
    
    if SHUFFLE_NET:
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass
    graphlist = glob(os.path.join(bdir, "*.json"))#[0:100]
    
    print(len(graphlist))

    try:
        os.mkdir(MODEL_PATH)
    except FileExistsError:
        pass
    graphlist = natsorted(graphlist)
    
    ##### remove graphs that have no omics data
    graphlist = [graph for graph in graphlist if os.path.split(graph)[-1].split('_')[0][:12] in df.index]
    # mask = df.index.isin(graphlist)
    print(len(graphlist))
    #####
    dataset = []
    from tqdm import tqdm
    for graph in tqdm(graphlist):
        TAG = os.path.split(graph)[-1].split('_')[0][:12]
        status = TS.loc[TAG,:][1]
        event, event_time = TS.loc[TAG,:][0], TS.loc[TAG,:][1]
        if np.isnan(event):
            continue
        if SHUFFLE_NET:
            G = loadfromjson(graph)
            # G = Data(**G)
            G = OmicGraphData(**G)
        else:
            if USE_CUDA:
                G = pickleLoad(graph)
                G.to('cpu')
            else:
                G = torch.load(graph, map_location=device)
        try:
            G.y = toTensorGPU([int(status)], dtype=torch.long, requires_grad = False)
        except ValueError:
            continue
        W = radius_neighbors_graph(toNumpy(G.coords), 1500, mode="connectivity",include_self=False).toarray()
        g = toGeometricWW(toNumpy(G.x),W,toNumpy(G.y))
        g.coords = G.coords
        g.event = toTensor(event)
        g.e_time = toTensor(event_time)
        #
        g.omics = toTensor(df.loc[TAG])
        # g.omics = torch.from_numpy(np.array(df.loc[TAG])).type(torch.float).requires_grad_(True)
        #
        if SHUFFLE_NET:
            dataset.append([TAG,(event,event_time)])
            if SAVE_SHUFFLENET:
                torch.save(g,directory+'/'+TAG+'.g')
        else:
            dataset.append(g)

# %%
G = torch.load(directory + '/TCGA-3C-AALJ.g')
print(G.x.shape[1])

# %%
trainingDataset = dataset

folds = 5

if SHUFFLE_NET:
    G = torch.load(directory + '/TCGA-3C-AALJ.g')
else:
    G = dataset[0]

converg_vals = []
fold_concord = []
eval_metrics = []

for train_index, vali_index in SplitBrcaData(trainingDataset,folds,SHUFFLE_NET,0.2):
    # get indices for training and testing
    
    # moved model/optimiser setup inside
    net = NetWrapper(device = device, features = 'BRCA-SHUFFLE')
    ### format of dataset: [TAG,(event,event_time)]
    x_train = [trainingDataset[i] for i in train_index]
    # Only censoring the test data
    # x_val = net.censor_data(x_val,10) 
    losses, concords, BestModel = net.train(x_train,
                                            None,
                                            return_best = True,
                                            max_batches = NUM_BATCHES)
    # Evaluate
    testDataset = [trainingDataset[i] for i in vali_index]
    testDataset = net.censor_data(testDataset,10)
    eval = Evaluator(BestModel,features='BRCA-SHUFFLE')
    
    concord = eval.test_evaluation(testDataset)
    print(concord)
    
    converg_vals.append(losses)
    fold_concord.append(concords)
    eval_metrics.append(concord)
    #m = max(concords)

avg_c = mean(eval_metrics)
stdev_c = stdev(eval_metrics)
print("Performance on test data over %d folds: \n" % folds)
print(str(avg_c)+' +/- '+str(stdev_c))
print(f"perf on each split was: {eval_metrics}")


