import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p',"--process", help="process graphs to also store omics data",
                    action="store_true")
parser.add_argument('-f','--feature_set',help="select feature set: SHUFFLE, RES, CC available",
                    default="SHUFFLE")
parser.add_argument('--fusion',help="methodology for fusion: GATED vs BILINEAR vs LINEAR vs CONCAT",
                    default="GATED")
parser.add_argument('--omics_concat', help="use concatenation to graph instead of net for omics",
                    action="store_true")

parser.add_argument('--lr', help="learning rate for optimiser",
                    default=0.00002)
parser.add_argument('--weight_decay', help="weight decay for optimiser",
                    default=0.005)
parser.add_argument('--batch_number', help="number of batches to train/test for",
                    default=2000)
parser.add_argument('--omics_len', help="length of omics data for each patient",
                    default=95)

parser.add_argument("--modelsummary", help="print structure of best model using torchinfo at end",
                    action="store_true")
parser.add_argument('--figure_dir', help="directory to save figures to",
                    default="./figures")
parser.add_argument("--noplot", help="suppress plotting of losses and KM curves",
                    action="store_true")
args = parser.parse_known_args()[0]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ujson as json
import os
import pdb
import pickle
import math
import re
import sys

from glob import glob
from copy import deepcopy
from tqdm import tqdm
from random import shuffle
from itertools import islice
from collections import OrderedDict
from statistics import mean, stdev
from pathlib import Path
from time import time

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
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test

from torchinfo import summary

print(args)
print("*****")
print(args, file=sys.stderr)
print("*****", file=sys.stderr)

#"parameters for the model"
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay
NUM_BATCHES = args.batch_number

FRAC_VAL = 0.2

#"feature parameters"
OMICS_LEN = args.omics_len

FEATURE_SET = args.feature_set
FEATURES = {
    "SHUFFLE":1024,
    "RES":2048,
    "CC":4
}[FEATURE_SET]
GRAPH_PATH = {
    "SHUFFLE":'./data/graphs_shufflenet/',
    "RES":'./data/graphs_resnet50/',
    "CC":'./data/graphs_cellcomp/'
}[FEATURE_SET]
PKL_PATH = {
    "SHUFFLE":'./data/graphs_shufflenet_update/',
    "RES":'./data/graphs_resnet50_update/',
    "CC":'./data/graphs_cellcomp_update/'
}[FEATURE_SET]

print(f"*****\nFeature set: {FEATURE_SET}\nNode features: {FEATURES}\nGraph path: {GRAPH_PATH}\nPickle path: {PKL_PATH}\n*****")

# "directories"
# BDIR = r'./graphs_json/'
# PKLDIR = r'./graphs_pkl/'
BDIR = GRAPH_PATH
PKLDIR = PKL_PATH

SURV_FILE = './data/NIHMS978596-supplement-1.xlsx'
CLINI_FILE = './data/TCGA-BRCA-DX_CLINI (8).xlsx'

#"cuda/device"
USE_CUDA = torch.cuda.is_available()
DEVICE = {
    True:'cuda:0',
    False:'cpu'
}[USE_CUDA] 

#"options"
PROCESS_GRAPHS = args.process
FUSION = args.fusion

START_TIME=int(time())

#"end of the constants declaration"

def toTensor(v,dtype = torch.float,requires_grad = True):
    return torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad)

def toTensorGPU(v,dtype = torch.float,requires_grad = True):
    if USE_CUDA: return toTensor(v,dtype,requires_grad).cuda()
    return toTensor(v,dtype,requires_grad)

def toNumpy(v):
    if type(v) is not torch.Tensor: return np.asarray(v)
    if USE_CUDA: return v.detach().cpu().numpy()
    return v.detach().numpy()

def pickleLoad(ifile):
    with open(ifile, "rb") as f: return pickle.load(f)

class MMLData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'omics_tensor': return None
        return super().__cat_dim__(key, value, *args, **kwargs)

def toGeometricWW(X,W,y,tt=0):    
    # Data -> MMLData for better batching
    return MMLData(x=toTensor(X,requires_grad = False), 
                    edge_index=(toTensor(W,requires_grad = False)>tt).nonzero().t().contiguous(),
                    y=toTensor([y],dtype=torch.long,requires_grad = False))

def pair_find(data):
    """
    from data [[tag,(event,time)]], 
    produces list of tuple pairs (i,j): 
    i, j are indices of graphs in the training data list, 
    where time(i) > time(j)
    """
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
    # event_time is a list of event-time tuples
    eventVars = [pair[0] for pair in event_time]
    x = np.zeros(len(eventVars))
    shuffleSplit = StratifiedShuffleSplit(n_splits = numSplits, test_size = testSize)
    return shuffleSplit.split(x,eventVars)

def disk_graph_load(batch):
    # takes list of patient codes, returns list of corresponding loaded
    # pkls (graph Data/MMLData objects)
    loaded_graphs = [torch.load(PKLDIR + '/' + graph + '.g') for graph in batch]
    return [MMLData(**(data.to_dict())) for data in loaded_graphs]

def get_predictions(model,graphs,device=torch.device('cuda:0')) -> list:
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
            for d in loader: d = d.to(device)
            
            z,_ = model(d)
            z = torch.flatten(z)
            z = toNumpy(z)
            
            outputs.append(z)
            e_and_t.append(temp)
    return outputs, e_and_t

def get_patient_tags(directory = BDIR):
    # returns a list of all patient tags that have json graph data
    # first 12 characters of graph json filename should be the patient barcode
    json_list = glob(os.path.join(directory, "*.json"))
    return [os.path.split(filename)[-1][:12] for filename in json_list if "DX1" in filename]
    # what does dx2 mean????

def resolve_graph_filename(tag, directory = BDIR):
    # takes a single patient tag, returns a single filename Path
    return Path(directory) / (tag + "-01Z-00-DX1.json")

def loadfromjson(tag):
    # loads a single graph from a json file as a dict.
    filename = resolve_graph_filename(tag)
    with Path(filename).open() as fptr: graph_dict = json.load(fptr)
    graph_dict = {k: torch.tensor(np.array(v)) for k, v in graph_dict.items()}
    # return graph_dict 
    # or should we cast to pyg Data? (G = Data(**graph_dict)) here?
    return Data(**graph_dict).to(DEVICE)

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

class OmicsWrapper:
    # wrapper for the dataframe containing omics data
    def __init__(self):
        # import DSS event-time data
        label = pd.read_excel(SURV_FILE).rename(columns={'bcr_patient_barcode':'PATIENT'}).set_index('PATIENT')
        # filter to event/time columns for brca only
        label = label[["DSS","DSS.time"]][label.type == 'BRCA']
        
        ### import genomic data
        df = pd.read_excel(CLINI_FILE).set_index('PATIENT')
        # print(df.shape)
        
        # one-hot for each mutation value
        mut = pd.get_dummies(df.filter(regex="_mutation$"),dtype=float)

        # log1p all expression values
        expr = df.filter(regex="_expression$").apply(lambda x: [np.log1p(item) for item in x])

        # standard scale across all of the cnv values
        cnv = df.filter(regex="_CNV$|ZNF703")
        scaler = preprocessing.StandardScaler().fit(cnv)
        scaled = scaler.fit_transform(cnv)
        
        # join parts together
        df = pd.DataFrame(scaled, columns=cnv.columns, index=cnv.index)

        # due to inclusion of event-time data, should have 2 extra columns (97 vs 95)
        self.df = df.join(expr).join(mut).join(label,"PATIENT","inner").dropna()
        # print(self.df.shape)
        
    def get_tag_survlist(self, tags):
        # given a list of graph tags, attach the event/time only to those
        # that also exist in the dataframe (have omics), and add to a list
        tag_survlist = []
        for tag in tags:
            if tag in self.df.index:
                tag_survlist.append([tag, tuple(self.df.loc[tag,['DSS','DSS.time']])])
        return tag_survlist
    
    def get_tensor(self,tag): # get row at tag as a pytorch tensor
        row = self.df.loc[tag].drop(labels=["DSS","DSS.time"])
        return torch.tensor(row.values,dtype=torch.float)
    
    def get_omics_length(self): # remove 2 for the labels
        return self.df.shape[1] - 2

from torch.nn import Parameter

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

#############
# Graph Model
#############
class GNN(torch.nn.Module):
    def __init__(
        self,
        dim_features,dim_target,
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
        # if gembed=True then learn graph embedding for final classification 
        # (classify pooled node features), otherwise pool node decision scores
        
        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            # shufflenet = 1024 features
            # features -> 0: 16, 1: 16, 2: 8, -> target
            if layer == 0:
                #first layer
                self.first_h = Sequential(
                    Linear(dim_features, out_emb_dim), 
                    BatchNorm1d(out_emb_dim),
                    GELU())
                self.linears.append(Sequential(
                    Linear(out_emb_dim, dim_target),
                    GELU()))
                
            else:
                prev_emb_dim = self.embeddings_dim[layer-1]
                self.linears.append(Linear(out_emb_dim, dim_target))          
                if conv=='GINConv':
                    self.nns.append(Sequential(
                        Linear(prev_emb_dim, out_emb_dim), 
                        BatchNorm1d(out_emb_dim))) 
                    self.convs.append(GINConv(self.nns[-1], **kwargs))  
                    # Eq. 4.2 eps=100, train_eps=False
                elif conv=='EdgeConv':
                    self.nns.append(Sequential(
                        Linear(2*prev_emb_dim, out_emb_dim), 
                        BatchNorm1d(out_emb_dim)))         
                    self.convs.append(EdgeConv(self.nns[-1],**kwargs))
                    #DynamicEdgeConv#EdgeConv  aggr='mean'
                else: raise NotImplementedError  
        
        self.linears = torch.nn.ModuleList(self.linears)  
        # has got one more layer for initial input
        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        
    def forward(self, data) -> torch.tensor:
        x, edge_index, batch, pooling = data.x, data.edge_index, data.batch, self.pooling
        out = 0
        Z_sum = 0
        ##### convert x down from double? solves issue with the input to linear layers
        ## but, loses some precision
        x = x.float() 
        
        # Z_sum is the result of linear layers
        import torch.nn.functional as F
        for layer in range(len(self.embeddings_dim)):            
            if layer == 0:
                x = self.first_h(x)
                z = self.linears[layer](x)
                Z_sum += z
                dout = F.dropout(pooling(z, batch), 
                                p=self.dropout, training=self.training)
                out += dout
            else:
                x = self.convs[layer-1](x,edge_index)
                if not self.gembed:
                    z = self.linears[layer](x)
                    Z_sum += z
                    dout = F.dropout(pooling(z, batch), 
                                    p=self.dropout, training=self.training)
                else:
                    dout = F.dropout(self.linears[layer](pooling(x, batch)), 
                                    p=self.dropout, training=self.training)
                out += dout
        return out,Z_sum,x

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

##############################################################################
# Graph + Omic
##############################################################################
class GraphomicNet(nn.Module):
    """
    Forward function takes graph data and omics data,
    passing both to respective networks and getting feature vector
    representations.
    Fuses, then returns a single value.
    Current configuration:
        Graph - Slidegraph, feature vector size 32
        Omics - Maxnet, feature vector size 32
        Fusion - Bilinear, produces fused vector size 64
        Linear 64 to 1
    Possible alternative configurations:
        - Concatenation at feature vector level, then FC (linear)
            - i.e. 32 + 95 to 1
        - Replace Maxnet with single FC (linear/bilinear) layer that transforms 95 to 32
            - then 32 + 32 to 1
        - Pre-output fusion; both nets produce an independent scalar output, we fuse here
    """
    def __init__(self, omic_input_len, act=None):
        super(GraphomicNet, self).__init__()

        graph_features = FEATURES
        omic_features = OMICS_LEN
        fusion_length = graph_features+omic_features

        graph_target = 32
        
        self.grph_net = GNN(
            dim_features=graph_features,
            dim_target = graph_target,
            layers = [64,64,32,32],
            dropout = 0.0,
            pooling = 'mean', 
            conv='EdgeConv', 
            aggr = 'max'
            ).to(DEVICE)
        
        # maxnet will output a feature representation (size omic_dim)
        # AND a classification (size label_dim)
        
        # alternatively, we could concatenate the omics vector to the
        # graph feature rep and fuse directly from there

        # concat - concatenate maxnet features to slidegraph features
        # omics_concat - concatenate omics features to slidegraph features
        if not args.omics_concat:
            self.omic_net = MaxNet(
                input_dim=omic_input_len, 
                omic_dim=32, 
                dropout_rate=0.25, 
                act=None, 
                label_dim=1,
                init_max=True
                ).to(DEVICE)
            ### need to decide what to do from here based on fusion strategy
            if FUSION == "GATED":
                self.fusion = BilinearFusion()
            elif FUSION == "BILINEAR":
                self.fusion = SimpleFusion(bilinear=True)
            elif FUSION == "LINEAR":
                self.fusion = SimpleFusion(bilinear=False)
            elif FUSION == "CONCAT":
                self.fusion = SimpleFusion(bilinear=False,cat_only=True)
            else:
                raise NotImplementedError()
        else:
            if FUSION == "GATED":
                self.fusion = BilinearFusion(dim2=OMICS_LEN)
            elif FUSION == "BILINEAR":
                self.fusion = SimpleFusion(dim2=OMICS_LEN, bilinear=True)
            elif FUSION == "LINEAR":
                self.fusion = SimpleFusion(dim2=OMICS_LEN, bilinear=False)
            elif FUSION == "CONCAT":
                self.fusion = SimpleFusion(dim2=OMICS_LEN, bilinear=False,cat_only=True)
            else:
                raise NotImplementedError()
        
        # classifier will take output fused vector (64)
        # and collapse to single value (1) by single linear layer
        if not args.omics_concat:
            self.classifier = nn.Sequential(nn.Linear(64, 1))
        else:
            print(graph_features+omic_features)
            self.classifier = nn.Sequential(nn.Linear(graph_target+omic_features, 1))
        self.act = act
        
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, data):
        grph_vec, _, _ = self.grph_net(data)
        ### additional option to not use the net? and concat directly

        # batched, so omics tensor is concatenated to single long vector. 
        # must reshape after passing to net
        # print(f"Graph shape:{grph_vec.shape}")
        if not args.omics_concat:
            omic_vec, _ = self.omic_net(data.omics_tensor)
        else:
            omic_vec = data.omics_tensor
            omic_vec = omic_vec.reshape([-1,OMICS_LEN])
        
        
        if FUSION == "GATED":
            # original logic
            features = self.fusion(grph_vec, omic_vec)
            hazard = self.classifier(features)
            
            if self.act is not None:
                hazard = self.act(hazard)
                if isinstance(self.act, nn.Sigmoid):
                    hazard = hazard * self.output_range + self.output_shift
            
            return hazard, features
        else:
            features = self.fusion(grph_vec, omic_vec)
            hazard = self.classifier(features)
            return hazard, features

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            if name in self.__dict__['_parameters']:
                return True
        if '_buffers' in self.__dict__:
            if name in self.__dict__['_buffers']:
                return True
        if '_modules' in self.__dict__:
            if name in self.__dict__['_modules']:
                return True
        return False

class NetWrapper:
    def __init__(self, omics_df : OmicsWrapper) -> None:
        self.omics_df = omics_df
        self.model = GraphomicNet(self.omics_df.get_omics_length()).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
    
    def loss_fn(self,batch) -> float:
        "batch input is a list of tuples of paired graph tags"
        # this flattens list
        unzipped = [j for pair in batch for j in pair]
        tag_set = list(set(unzipped))
        graphs = disk_graph_load(tag_set)

        # for graph in graphs:
        #     print(graph)
        
        batch_load = DataLoader(graphs, batch_size = len(graphs))
        for data in batch_load:
            data = data.to(DEVICE)
        
        z = toTensorGPU(0)
        loss = 0
        
        self.model.train()
        self.optimizer.zero_grad()
        
        output,_ = self.model(data)
        
        for (xi,xj) in batch:
            graph_i = tag_set.index(xi)
            graph_j = tag_set.index(xj)
            # Compute loss function
            dz = output[graph_i] - output[graph_j]
            loss += torch.max(z, 1.0 - dz)
        loss = loss/len(batch)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def validation_loss_and_Cindex_eval(self,graphs,pairs) -> float:
        print('Number of Validation Pairs: ' + str(len(pairs)))
        predictions, e_and_t = get_predictions(self.model,graphs)
        
        tot_loss = 0
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
    
    def censor_data(self, graphs, censor_time): 
        "given censor_time in years, censor individual times if greater than input."
        cen_time = 365 * censor_time
        for graph in graphs:
            time = graph[1][1]
            if time > cen_time: graph[1] = (0,cen_time)
        return graphs
    
    def train(self,training_data,validation_data=None,
                max_batches=500,
                num_logs=50, early_stopping = 10,return_best = False,
                batch_size = 10) -> float:
        # format of training_data: [TAG,(event,event_time)]
        
        return_best = return_best and validation_data is not None
        log_interval = max_batches // num_logs
        
        loss_vals = { 'train': [], 'validation': [] }
        
        concords = []
        c_best = 0.5
        best_batch = 1000
        patience = early_stopping
        
        training_indexes = pair_find(training_data)
        
        print("Number of batches used for training " + str(max_batches))
        print('Num Pairs: ' + str(len(training_indexes)))
        
        if validation_data is not None:
            validation_indexes = pair_find(validation_data)
        
        # To resolve list index errors with large NUM_BATCHES vals
        counter = 0 
        
        ### understand the batching logic next TODO
        for i in tqdm(range(max_batches)):
            if counter < len(training_indexes) - batch_size:
                batch_pairs = []
                index_pairs = training_indexes[counter:counter+batch_size]
                
                for l_index, r_index in index_pairs:
                    batch_pairs.append((training_data[l_index][0],training_data[r_index][0]))
                
                #batch pairs will consist of patient tags
                loss = self.loss_fn(batch_pairs)
                
                counter += batch_size
            else: counter = 0
            loss_vals['train'].append(loss)
            if validation_data is not None:
                if i % log_interval == 0:
                    val_loss, c_val = self.validation_loss_and_Cindex_eval(validation_data,validation_indexes)
                    loss_vals['validation'].append(val_loss)
                    concords.append(c_val)
                    print("Current Vali Loss Val: " + str(val_loss) + "\n")
                    print("\n" + "Current Loss Val: " + str(loss) + "\n")
                    if return_best and c_val > c_best:
                        c_best = c_val
                        #best_model = deepcopy(model)
                        best_batch = i
                    if i - best_batch > patience*log_interval:
                        print("Early Stopping")
                        #break
        return loss_vals, concords, self.model
    
class Evaluator:
    def __init__(self, model) -> None:
        self.model = model
    
    def test_evaluation(self,testDataset):
        predictions, e_and_t = get_predictions(self.model,testDataset)
        T = [x[1] for x in e_and_t]
        E = [x[0] for x in e_and_t]
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
        
        add_at_risk_counts(km_high, km_low, ax=ax)
        plt.title('Kaplan-Meier estimate')
        plt.ylabel('Survival probability')
        # plt.show()
        plt.tight_layout()
        kmpath = f"{args.figure_dir}/{FEATURE_SET}_{FUSION}_{args.omics_concat}-{int(time())}.png"
        plt.savefig(kmpath)
        print(f"KM curves plotted to {kmpath}.")
        plt.clf()
        
        results = logrank_test(T_low, T_high, E_low, E_high)
        print("p-value %s; log-rank %s" % (results.p_value, np.round(results.test_statistic, 6)))

def GraphProcessing(omics_df:OmicsWrapper, tag_survlist):
    print("Pre-saving graphs.")
    for tag, label in tqdm(tag_survlist):
        G = MMLData(**(loadfromjson(tag)))
        try:
            G.y = toTensorGPU([int(label[1])], dtype=torch.long, requires_grad = False)
        except ValueError:
            continue
        # creates new graph edges, where nodes that are 1500 apart are connected to each other
        W = radius_neighbors_graph(toNumpy(G.coords), 1500, mode="connectivity",include_self=False).toarray()
        g = toGeometricWW(toNumpy(G.x),W,toNumpy(G.y))
        # is this MMLData? or Data?
        g.coords = G.coords
        g.event = toTensor(label[0])
        g.e_time = toTensor(label[1])
        g.omics_tensor = toTensor(omics_df.get_tensor(tag))
        torch.save(g, PKLDIR + '/' + tag + '.g')

if __name__ == '__main__':
    ### import survival data from file: DSS event and time
    print("Preparing omics.")
    omics = OmicsWrapper()
    print("Getting tags.")
    patient_tags = get_patient_tags()
    print("Matching to omics.")
    tag_survlist = omics.get_tag_survlist(patient_tags)
    ## format is [[TAG,(event,time)]]
    
    # # bind omics tensor to slidegraph, pickle and store
    if(PROCESS_GRAPHS): GraphProcessing(omics,tag_survlist)
    
    trainingDataset = tag_survlist
    folds = 5
    
    converg_vals = []
    fold_concord = []
    eval_metrics = []
    
    e_t_list = [item[1] for item in tag_survlist]
    
    # get indices for training and testing
    for fold, (train_index, vali_index) in enumerate(SplitOnEvent(e_t_list, folds, FRAC_VAL)):
        print(f"NOW ON FOLD {fold+1}\n*****")
        net = NetWrapper(omics)
        # x_train is the subset of trainingDataset at the split indices
        x_train = [trainingDataset[i] for i in train_index]
        e_t_train = [e_t_list[i] for i in train_index]
        # Only censoring the test data
        # x_val = net.censor_data(x_val,10) 
        losses, concords, BestModel = net.train(x_train,
                                                return_best = True,
                                                max_batches = NUM_BATCHES)
        # Evaluate for fold
        testDataset = [trainingDataset[i] for i in vali_index]
        testDataset = net.censor_data(testDataset,10)
        eval = Evaluator(BestModel)
        concord = eval.test_evaluation(testDataset)
        print(concord)
        
        converg_vals.append(losses)
        fold_concord.append(concords)
        # fold_concord and converg_vals are never used?
        eval_metrics.append(concord)
        #m = max(concords)
        if not args.noplot:
            eval.K_M_Curves(testDataset,None)
    
    if args.modelsummary:
        print(summary(BestModel))
    
    avg_c = mean(eval_metrics)
    stdev_c = stdev(eval_metrics)
    print("Performance on test data over %d folds:" % folds)
    print(str(avg_c)+' +/- '+str(stdev_c))
    print(f"perf on each split was: {eval_metrics}")
    
    if not args.noplot:
        for i,lossvals in enumerate(converg_vals):
            figpath = f"{args.figure_dir}/{FEATURE_SET}_{FUSION}_{args.omics_concat}-{START_TIME}-fold{i+1}.png"
            f = plt.figure()
            f.set_figwidth(12)
            plt.plot(lossvals["train"])
            plt.savefig(figpath)
            print(f"Losses for fold {i+1} plotted to {figpath}.")
            plt.clf()