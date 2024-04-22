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

###########################

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

from networks.graph_net import *
from networks.omics_net import *
from networks.fusion import *

from utils.classes import *

# print args to stdout and stderr
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
# PKL_PATH = {
#     "SHUFFLE":'./data/graphs_shufflenet_update/',
#     "RES":'./data/graphs_resnet50_update/',
#     "CC":'./data/graphs_cellcomp_update/'
# }[FEATURE_SET]
PKL_PATH = {
    "SHUFFLE":'./data/graphs_shufflenet_new/',
    "RES":'./data/graphs_resnet50_new/',
    "CC":'./data/graphs_cellcomp_new/'
}[FEATURE_SET]

print(f"*****\nFeature set: {FEATURE_SET}\nNode features: {FEATURES}\nGraph path: {GRAPH_PATH}\nPickle path: {PKL_PATH}\n*****")

# "directories"
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

###########################

def toTensor(v,dtype = torch.float,requires_grad = True):
    return torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad)

def toTensorGPU(v,dtype = torch.float,requires_grad = True):
    if USE_CUDA: return toTensor(v,dtype,requires_grad).cuda()
    return toTensor(v,dtype,requires_grad)

def toNumpy(v):
    if type(v) is not torch.Tensor: return np.asarray(v)
    if USE_CUDA: return v.detach().cpu().numpy()
    return v.detach().numpy()

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
    shuffleSplit = StratifiedShuffleSplit(numSplits, test_size = testSize)
    return shuffleSplit.split(np.zeros(len(eventVars)),eventVars)
    # returns [(train_idx,vali_idx)]

def get_predictions(model,graphs,device=torch.device('cuda:0')) -> list:
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

def get_patient_tags(directory = GRAPH_PATH):
    # returns a list of all patient tags that have json graph data
    # first 12 characters of graph json filename should be the patient barcode
    json_list = glob(os.path.join(directory, "*.json"))
    return [os.path.split(filename)[-1][:12] for filename in json_list if "DX1" in filename]

def resolve_graph_filename(tag, directory = GRAPH_PATH):
    # takes a single patient tag, returns a single filename Path
    return Path(directory) / (tag + "-01Z-00-DX1.json")

def loadfromjson(tag):
    # loads a single graph from a json file as a dict.
    filename = resolve_graph_filename(tag)
    with Path(filename).open() as fptr: graph_dict = json.load(fptr)
    graph_dict = {k: torch.tensor(np.array(v)) for k, v in graph_dict.items()}
    return graph_dict

def disk_graph_load(batch):
    # takes list of patient codes, returns list of corresponding loaded
    # pkls (graph Data/MMLData objects)
    loaded_graphs = [torch.load(PKL_PATH + '/' + graph + '.g') for graph in batch]
    return [MMLData(**(data.to_dict())) for data in loaded_graphs]

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

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
        omic_features = omic_input_len
        fusion_length = graph_features+omic_features

        target_embedding = 32
        
        self.graph_net = SlideGraphGNN(
            dim_features=graph_features,
            dim_target = target_embedding,
            layers = [64,64,32,32],
            dropout = 0.0,
            pooling = 'mean', 
            conv='EdgeConv', 
            gembed = True,
            aggr = 'max'
            ).to(DEVICE)
        
        # maxnet will output a feature representation (size omic_dim)
        # alternatively, we could concatenate the omics vector to the
        # graph feature rep and fuse directly from there

        # concat - concatenate maxnet features to slidegraph features
        # omics_concat - concatenate omics features to slidegraph features
        if not args.omics_concat:
            self.omic_net = MaxNet(
                input_dim=omic_features, 
                out_dim=target_embedding, 
                dropout_rate=0.25, 
                init_max=True
                ).to(DEVICE)
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
            # final layer will take output fused vector (64)
            # and collapse to single value (1) by single linear layer
            self.post_fusion = nn.Sequential(nn.Linear(64, 1))
        else:
            if FUSION == "GATED":
                self.fusion = BilinearFusion(dim2=omic_input_len)
            elif FUSION == "BILINEAR":
                self.fusion = SimpleFusion(dim2=omic_input_len, bilinear=True)
            elif FUSION == "LINEAR":
                self.fusion = SimpleFusion(dim2=omic_input_len, bilinear=False)
            elif FUSION == "CONCAT":
                self.fusion = SimpleFusion(dim2=omic_input_len, bilinear=False,cat_only=True)
            else:
                raise NotImplementedError()
            print(fusion_length)
            self.post_fusion = nn.Sequential(nn.Linear(fusion_length, 1))
            
        self.act = act

    def forward(self, data):
        graph_vec, _, _ = self.graph_net(data)
        # print(f"Graph shape:{graph_vec.shape}")

        ### option to not use the net? and concat directly
        if not args.omics_concat:
            omic_vec = self.omic_net(data.omics_tensor)
        else:
            omic_vec = data.omics_tensor
            omic_vec = omic_vec.reshape([-1,OMICS_LEN])
        
        
        if FUSION == "GATED":
            # original logic
            features = self.fusion(graph_vec, omic_vec)
            hazard = self.post_fusion(features)
            if self.act is not None:
                hazard = self.act(hazard)
            
            return hazard, features
        else:
            features = self.fusion(graph_vec, omic_vec)
            hazard = self.post_fusion(features)
            return hazard, features

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__ and name in self.__dict__['_parameters']:
            return True
        elif '_buffers' in self.__dict__ and name in self.__dict__['_buffers']:
            return True
        elif '_modules' in self.__dict__ and name in self.__dict__['_modules']:
            return True
        else: return False

class NetWrapper:
    def __init__(self, omics_df : DFWrapper) -> None:
        self.omics_df = omics_df
        omic_len = self.omics_df.get_omics_length()
        
        self.model = GraphomicNet(omic_len).to(DEVICE)
        # adam vs adamW?
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=LEARNING_RATE,
                                     weight_decay=WEIGHT_DECAY)
    
    def loss_fn(self,batch) -> float:
        #"batch input is a list of tuples of paired graph tags"
        # following line flattens list
        unzipped = [j for pair in batch for j in pair]
        tag_set = list(set(unzipped))
        graphs = disk_graph_load(tag_set)
        
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
        # given a threshold in years, censor individual times if greater
        cen_time = 365 * censor_time
        for graph in graphs:
            if graph[1][1] > cen_time: graph[1] = (0,cen_time)
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
            else: 
                counter = 0
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

def GraphProcessing(omics_df:DFWrapper, tag_survlist):
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
        torch.save(g, PKL_PATH + '/' + tag + '.g')

if __name__ == '__main__':
    ### import survival data from file: DSS event and time
    print("Preparing omics.")
    omics = DFWrapper(SURV_FILE, CLINI_FILE)
    print("Getting tags.")
    patient_tags = get_patient_tags()
    print("Matching to omics.")
    tag_survlist = omics.get_tag_survlist(patient_tags)
    ## format is [[TAG,(event,time)]]
    
    # # bind omics tensor to slidegraph, pickle and store
    if(PROCESS_GRAPHS): 
        GraphProcessing(omics,tag_survlist)
    
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
        # Evaluate on test dataset for current fold
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