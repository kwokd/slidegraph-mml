import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pdb
import pickle
import math
import re

from glob import glob
from copy import deepcopy
from tqdm import tqdm
from random import shuffle
from itertools import islice
from collections import OrderedDict
from statistics import mean, stdev

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Sequential, Linear, ReLU,Tanh,LeakyReLU,ELU,SELU,GELU
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, Sampler

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv,EdgeConv, PNAConv,DynamicEdgeConv,global_add_pool, global_mean_pool, global_max_pool

from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index as cindex

from torchinfo import summary


"parameters for the model"
LEARNING_RATE = 0.00002
WEIGHT_DECAY = 0.005
NUM_BATCHES = 2000
FRAC_VAL = 0.2

"feature parameters"
OMICS_LEN = 95

FEATURE_SET = "RES"

FEATURES = {
    "SHUFFLE":1024,
    "RES":2048,
    "CC":4
}[FEATURE_SET]

GRAPH_PATH = {
    "SHUFFLE":'../data/graphs_shufflenet/',
    "RES":'../data/graphs_resnet50/',
    "CC":'../data/graphs_cellcomp/'
}[FEATURE_SET]

PKL_PATH = {
    "SHUFFLE":'../data/graphs_shufflenet_update/',
    "RES":'../data/graphs_resnet50_update/',
    "CC":'../data/graphs_cellcomp_update/'
}[FEATURE_SET]

# "directories"
# BDIR = r'./graphs_json/'
# PKLDIR = r'./graphs_pkl/'
BDIR = GRAPH_PATH
PKLDIR = PKL_PATH

"cuda/device"
USE_CUDA = torch.cuda.is_available()
DEVICE = {
    True:'cuda:0',
    False:'cpu'
}[USE_CUDA] 

PROCESS_GRAPHS = True


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

def toGeometricWW(X,W,y,tt=0):    
    return Data(x=toTensor(X,requires_grad = False), 
                edge_index=(toTensor(W,requires_grad = False)>tt).nonzero().t().contiguous(),
                y=toTensor([y],dtype=torch.long,requires_grad = False))

def pair_find(data):
    # from data [[tag,(event,time)]] produce index pairs
    # where i's time is longer than j's
    indexes = []
    for j in range(len(data)):
        graph_j = data[j]
        event_j = graph_j[1][0]
        time_j = graph_j[1][1]
        if event_j == 1:
            for i in range(len(data)): 
                graph_i = data[i]            
                time_i = graph_i[1][1]
                if j != i and time_i > time_j:
                    indexes.append((i,j))
    shuffle(indexes)
    return indexes

def SplitOnEvent(event_time, numSplits, testSize):
    # input is a list of event-time tuples
    eventVars = [pair[0] for pair in event_time]
    x = np.zeros(len(event_time))
    shuffleSplit = StratifiedShuffleSplit(n_splits = numSplits, test_size = testSize)
    return shuffleSplit.split(x,eventVars)

def disk_graph_load(batch):
    # takes list of patient codes, returns list of graph Data objects
    return [torch.load(PKLDIR + '/' + graph + '.g') for graph in batch]

def get_predictions(model,graphs,omics_df,device=torch.device('cuda:0')) -> list:
    # removed dependency on omics_df, can cascade deletion
    
    outputs = []
    e_and_t = []
    
    model.eval()
    with torch.no_grad():
        for graph in graphs:
            tag = [graph[0]]
            temp = [graph[1][0], graph[1][1]]
            graph = disk_graph_load(tag)
            size = 1
            loader = DataLoader(graph, batch_size=size)
            for d in loader:
                d = d.to(device)
            
            z,_ = model(d)

            print(z)
            z=torch.flatten(z)
            ########## this is the problem
            z = toNumpy(z)
            outputs.append(z)
            e_and_t.append(temp)
    
    return outputs, e_and_t

import ujson as json
from pathlib import Path

def get_patient_tags(directory = BDIR):
    # returns a list of all patient tags that have json-stored graphs
    json_list = glob(os.path.join(directory, "*.json"))
    # [os.path.split(filename)[-1][:12] for filename in json_list]
    # first 12 characters of graph json filename should be the patient barcode
    # return [os.path.split(filename)[-1][:12] for filename in json_list if filename[-6] == "1"]
    return [os.path.split(filename)[-1][:12] for filename in json_list if "DX1" in filename]

def resolve_graph_filename(tag, directory = BDIR):
    # takes a single patient tag, returns a single filename Path
    return Path(directory) / (tag + "-01Z-00-DX1.json")

def loadfromjson(tag):
    # loads a single graph from a json file as a dict. still needs to cast to pyg Data (G = Data(**graph_dict))
    filename = resolve_graph_filename(tag)
    with Path(filename).open() as fptr:    
        graph_dict = json.load(fptr)
    graph_dict = {k: torch.tensor(np.array(v)) for k, v in graph_dict.items()}
    # return graph_dict (or should we convert here?)
    return Data(**graph_dict).to(DEVICE)

def json_graph_load(batch):
    # given a list of tags, load graphs into a list
    return [loadfromjson(tag) for tag in batch]

def get_updated_connectivity(batch):
    # performs the connectivity update for all graphs in json_graph_load
    out = []
    for graph in json_graph_load(batch):
        W = radius_neighbors_graph(toNumpy(graph.coords), 1500, mode="connectivity",include_self=False).toarray()
        g = toGeometricWW(toNumpy(graph.x),W,toNumpy(graph.y))
        # g.coords = graph.coords
        # attach the event/event time here?
        out += g
    return out

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

class MMLData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'omics_tensor':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

class OmicsWrapper:
    # wrapper for the dataframe containing omics data
    def __init__(self):
        # import DSS event-time data
        survival_file = r'./NIHMS978596-supplement-1.xlsx'
        label = pd.read_excel(survival_file).rename(columns={'bcr_patient_barcode':'PATIENT'}).set_index('PATIENT')
        #filter to event/time columns for brca only
        label = label[["DSS","DSS.time"]][label.type == 'BRCA']
        
        ### import genomic data
        clini_file = './TCGA-BRCA-DX_CLINI (8).xlsx'
        df = pd.read_excel(clini_file).set_index('PATIENT')  # path to clinical file
        # print(df.shape)
        
        # one-hot for each mutation value
        mut = pd.get_dummies(df.filter(regex="_mutation$"),dtype=float)
        # log1p all expression values
        expr = df.filter(regex="_expression$").apply(lambda x: [np.log1p(item) for item in x])
        # standard scale across all of the cnv values
        
        cnv = df.filter(regex="_CNV$|ZNF703")
        scaler = preprocessing.StandardScaler()
        scaler.fit(cnv)
        scaled = scaler.fit_transform(cnv)
        
        # join parts together
        df = pd.DataFrame(scaled, columns=cnv.columns, index=cnv.index)
        self.df = df.join(expr).join(mut).join(label,"PATIENT","inner").dropna()
        # due to inclusion of event-time data, should have 2 extra columns (97 vs 95)
        # print(self.df.shape)
        
    def get_tag_survlist(self, tags):
        # given a list of graph tags, attach the event/time only to those
        # that are in the dataframe, and add to a list
        tag_survlist = []
        for tag in tags:
            if tag in self.df.index:
                tag_survlist.append([tag, tuple(self.df.loc[tag,['DSS','DSS.time']])])
        return tag_survlist
    
    def get_tensor(self,tag):
        # get row at tag as a pytorch tensor
        row = self.df.loc[tag].drop(labels=["DSS","DSS.time"])
        return torch.tensor(row.values,dtype=torch.float)
    
    def get_omics_length(self):
        # remove 2 for the labels
        return self.df.shape[1] - 2

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

    def forward(self, x):
        # features = self.encoder(x)
        omics_in = x.reshape([-1,OMICS_LEN])
        ##### ABOVE IS A STOPGAP MEASURE
        features = self.encoder(omics_in)
        out = self.classifier(features)
        if self.act is not None:
            out = self.act(out)
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift
        
        return features, out

#############
# Graph Model
#############
class GNN(torch.nn.Module):
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
        ##### convert down from double?
        x = data.x.float()
        
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

############
# Fusion
############
class SimpleFusion(nn.Module):
    def init(self):
        super(SimpleFusion,self).__init__()
        # self.linear = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
    def forward(self):
        pass

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
        # o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        # o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o1 = torch.cat((o1,torch.ones(o1.shape[0],1,dtype=torch.float, device='cuda')),1)
        o2 = torch.cat((o2,torch.ones(o2.shape[0],1,dtype=torch.float, device='cuda')),1)
        
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, o1, o2), 1)
        out = self.encoder2(out)
        return out

##############################################################################
# Graph + Omic
##############################################################################
class GraphomicNet(nn.Module):
    def __init__(self, omic_input_len, act=None):
        super(GraphomicNet, self).__init__()
        self.grph_net = GNN(
            dim_features=FEATURES,
            dim_target = 32,
            layers = [64,48,32,32],
            dropout = 0.0, 
            pooling = 'mean', 
            conv='EdgeConv', 
            aggr = 'max'
            ).to(DEVICE)
        
        self.omic_net = MaxNet(
            input_dim=omic_input_len, 
            omic_dim=32, 
            #^output dim; i.e. transform input tensor width to 32
            dropout_rate=0.25, 
            act=None, 
            label_dim=1, 
            init_max=True
            ).to(DEVICE)
        
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
        # omics data is still a list of tensors
        # omic_vec, _ = self.omic_net(omics_data)
        # omic_vec, _ = self.omic_net(torch.stack(omics_data))
        omic_vec, _ = self.omic_net(data.omics_tensor)
        
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

class NetWrapper:
    def __init__(self, omics_df : OmicsWrapper) -> None:
        self.omics_df = omics_df
        self.model = GraphomicNet(self.omics_df.get_omics_length()).to(DEVICE)
        print(summary(self.model))
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
    
    def loss_fn(self,batch) -> float:
        unzipped = [j for pair in batch for j in pair]
        # batch is list of tuples of paired graph tags, this flattens list
        
        # tag_set = list(set(unzipped))
        # omics_tensors = [self.omics_df.get_tensor(tag).to(DEVICE) for tag in tag_set]
        # graphs = json_graph_load(tag_set)
        
        # batch_load = DataLoader(graphs, batch_size = len(graphs))
        # for graph_data in batch_load: 
        #     graph_data = graph_data.to(torch.device(DEVICE))
        
        # print(graph_data) ###########
        
        tag_set = list(set(unzipped))
        graphs = disk_graph_load(tag_set)
        unzipped = None
        
        batch_load = DataLoader(graphs, batch_size = len(graphs))
        
        for data in batch_load: 
            data = data.to(DEVICE)
        
        z = toTensorGPU(0)
        loss = 0
        
        self.model.train()
        self.optimizer.zero_grad()
        
        ######################### edit to use graphomicnet VVV
        # output,_ = self.model(graph_data, omics_tensors)
        output,_ = self.model(data)
        ########################### ^^^
        num_pairs = len(batch)
        for (xi,xj) in batch:
            graph_i = tag_set.index(xi)
            graph_j = tag_set.index(xj)
            # Compute loss function
            dz = output[graph_i] - output[graph_j]
            loss += torch.max(z, 1.0 - dz)
        loss = loss/num_pairs

        # print(loss)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def validation_loss_and_Cindex_eval(self,graphs,pairs) -> float:
        tot_loss = 0
        print('Number of Validation Pairs: ' + str(len(pairs)))
        predictions, e_and_t = get_predictions(self.model,graphs)
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
            time = graph[1][1]
            if time > cen_time:
                graph[1] = (0,cen_time)
            else:
                continue
        return graphs
    
    def train(self,training_data,validation_data=None,
              max_batches=500,num_logs=50,
              early_stopping = 10,return_best = False,batch_size = 10) -> float:
        # format of training_data: [TAG,(event,event_time)]
        
        return_best = return_best and validation_data is not None
        log_interval = max_batches // num_logs
        
        loss_vals = { 'train': [], 'validation': [] }
        
        concords = []
        c_best = 0.5
        best_batch = 1000
        patience = early_stopping
        
        training_indexes = pair_find(training_data)
        # produces list of tuples (i,j): indices of graphs in the training data list,
        # where time(i) > time(j)
        
        print("Number of batches used for training " + str(max_batches))
        print('Num Pairs: ' + str(len(training_indexes)))
        
        counter = 0 # To resolve list index errors with large NUM_BATCHES vals
        
        for i in tqdm(range(1,max_batches + 1)):
            if counter < len(training_indexes) - batch_size:
                batch_pairs = []
                index_pairs = training_indexes[counter:counter+batch_size]
                
                for l_index, r_index in index_pairs:
                    batch_pairs.append((training_data[l_index][0],training_data[r_index][0]))
                
                #batch pairs will consist of patient tags
                loss = self.loss_fn(batch_pairs)
                
                counter += batch_size
            else:
                counter = 0
            
            loss_vals['train'].append(loss)
            
        return loss_vals, concords, self.model
    
class Evaluator:
    def __init__(self, model, omics_df) -> None:
        self.model = model
        self.omics_df = omics_df
    
    def test_evaluation(self,testDataset):
        predictions, e_and_t = get_predictions(self.model,testDataset,self.omics_df)
        T = [x[1] for x in e_and_t]
        E = [x[0] for x in e_and_t]
        print(len(predictions))
        print(predictions)
        print(len(T))
        print(T)
        print(len(E))
        concord = cindex(T,predictions,E)
        return concord
    
    def K_M_Curves(self, graphs, split_val, mode = 'Train') -> None:
        #unused for now?
        outputs, e_and_t = get_predictions(self.model,graphs)
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

def GraphProcessing(omics_df:OmicsWrapper, tag_survlist):
    print("Pre-saving graphs.")
    for tag, label in tqdm(tag_survlist):
        G = loadfromjson(tag)
        G = MMLData(**G)
        try:
            G.y = toTensorGPU([int(label[1])], dtype=torch.long, requires_grad = False)
        except ValueError:
            continue
        W = radius_neighbors_graph(toNumpy(G.coords), 1500, mode="connectivity",include_self=False).toarray()
        g = toGeometricWW(toNumpy(G.x),W,toNumpy(G.y))
        g.coords = G.coords
        g.event = toTensor(label[0])
        g.e_time = toTensor(label[1])
        g.omics_tensor = toTensor(omics_df.get_tensor(tag))
        
        torch.save(g,PKLDIR+'/'+tag+'.g')

if __name__ == '__main__':
    ### import survival data from file: DSS event and time
    print("Preparing omics.")
    omics = OmicsWrapper()
    print("Getting tags.")
    patient_tags = get_patient_tags()
    print("Matching to omics.")
    tag_survlist = omics.get_tag_survlist(patient_tags)
    ## format is [[tag,(event,time)]]
    
    if(PROCESS_GRAPHS):
        # will bind omics tensor to graph pickle and re-save
        GraphProcessing(omics,tag_survlist)
    
    trainingDataset = tag_survlist
    folds = 5
    
    converg_vals = []
    fold_concord = []
    eval_metrics = []
    
    e_t_list = [item[1] for item in tag_survlist]
    
    # get indices for training and testing
    for train_index, vali_index in SplitOnEvent(e_t_list, folds, FRAC_VAL):
        # moved model/optimiser setup inside
        net = NetWrapper(omics)
        ### format of dataset: [TAG,(event,event_time)]
        x_train = [trainingDataset[i] for i in train_index]
        # x_train is the subset of trainingDataset at the split indices
        # Only censoring the test data
        # x_val = net.censor_data(x_val,10) 
        losses, concords, BestModel = net.train(x_train,
                                                return_best = True,
                                                max_batches = NUM_BATCHES)
        # Evaluate
        testDataset = [trainingDataset[i] for i in vali_index]
        testDataset = net.censor_data(testDataset,10)
        eval = Evaluator(BestModel,omics)
        
        concord = eval.test_evaluation(testDataset)
        print(concord)
        
        converg_vals.append(losses)
        fold_concord.append(concords)
        # fold_concoer and concords is never used?
        eval_metrics.append(concord)
        #m = max(concords)
    
    avg_c = mean(eval_metrics)
    stdev_c = stdev(eval_metrics)
    print("Performance on test data over %d folds: \n" % folds)
    print(str(avg_c)+' +/- '+str(stdev_c))
    print(f"perf on each split was: {eval_metrics}")