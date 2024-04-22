#############
# Graph Model
#############

import torch
from torch.nn import BatchNorm1d, Sequential, Linear, ReLU,Tanh,LeakyReLU,ELU,SELU,GELU
from torch_geometric.nn import GINConv,EdgeConv, PNAConv,DynamicEdgeConv,global_add_pool, global_mean_pool, global_max_pool

class SlideGraphGNN(torch.nn.Module):
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
        super(SlideGraphGNN, self).__init__()
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