import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops

import ot
from ot.gromov import gromov_wasserstein2
from torch_geometric_temporal.nn.recurrent import GConvGRU, GConvLSTM
from torch_geometric.nn.conv import ChebConv

from siren_pytorch import *
from layers import GIN_Conv, SAGE_Conv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from models_molsetrep.sr_gnn import *
#from metrics_molsetrep import *

import numpy as np

from torch.profiler import profile, ProfilerActivity, record_function


import torch.autograd.profiler as profiler

import time

'''
c-IGNR

Goal:
- Map each input graph to a graphon function on [0,1]^2->[0,1], parameterized SIREN
- GNN encoder map input graph to z
- SIREN decoder map z to the graphon function on [0,1]^2->[0,1],


Reference:
- Implementation for the GNN encoder is partly referenced from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py

'''



#### CoordMLP 

class CoordMLP(nn.Module):
    """ Coordinates are generated from the node representations"""

    def __init__(self, in_dim, hidden_dims = [8], dropout = 0.0):
        super().__init__()
        layers = []
        prev = in_dim

        for h in hidden_dims:
            layers += [nn.Linear(prev, h)]
            layers += [nn.BatchNorm1d(h)]
            layers += [nn.LeakyReLU()]

            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h

        layers += [nn.Linear(prev, 3)]

        self.mlp = nn.Sequential(*layers)


    def forward(self, h):
        return self.mlp(h)
    



class CoordMLP_(nn.Module):
    """ Coordinates are generated from pooled graph representations"""

    def __init__(self, in_dim, hidden_dims = [8], N = 273, dropout = 0.0):
        super().__init__()
        layers = []
        prev = in_dim

        self.N = N

        for h in hidden_dims:
            layers += [nn.Linear(prev, h)]
            layers += [nn.BatchNorm1d(h)]
            layers += [nn.LeakyReLU()]

            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h

        layers += [nn.Linear(prev, self.N* 3)]

        self.mlp = nn.Sequential(*layers)


    def forward(self, h):
        y = self.mlp(h)
        y = y.view(-1, 3)
        return y







# Main AE structure =================================================================

class cIGNR(nn.Module):
    def __init__(self, net_adj, input_card, emb_dim, latent_dim, num_layer, gnn_layers, gnn_type='gin', global_pool='mean', JK='last', drop_ratio=0.,
                device=device,flag_emb=1, N =273):
        '''
        Encode each input graph into a latent code z of dimension [latent_dim]; 
        z is used to condition the training of the MLP function f_theta (mapping R2->[0,1])

        Input:
        ---------------------------
        input_card: input cardinality (our graph doesn't come with feature, map positional encoding to a trainable embedding)
        emb_dim:   graph embedding dimension for the encoder GNN
        latent_dim:  dimension of the latent z (after mapping graph embedding to the latent code z)/ can simple use the graph embedding as latent without extra mapping

        num_layer: number of encoder gnn layers
        gnn_type:  choose from {gin, gcn, graphsage}
        global_pool: global pooling to obtain the final embedding
        JK:  {last, concate, sum, max} to aggregate node representation from each layers
        flag_emb: whether to use initial embedding
        
        '''
        
        super(cIGNR, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.JK = JK
        self.device = device
        self.flag_emb = flag_emb
        self.gnn_type = gnn_type
        self.gnn_layers = gnn_layers

        self.N = N
        
        #---initial embedding---
        # In case one wants to embed the input node features. Since we already have gnn in the encoder, I didn't use any embedding layer here. 
        # flag_emb is usually set to 0 
        if flag_emb==1:
            self.x_embedding = torch.nn.Embedding(input_card,emb_dim)
            torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        #else:
            # self.x_embedding = nn.Linear(input_card, emb_dim)
        
        
        #---encoder network---
        # 1. gnn layers
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type== "gin":
                self.gnns.append(GIN_Conv(self.gnn_layers[layer], self.gnn_layers[layer+1]))
            elif gnn_type== "gcn":
                self.gnns.append(GCNConv(emb_dim, emb_dim))
            elif gnn_type=="graphsage":
                self.gnns.append(SAGE_Conv(emb_dim, emb_dim))
            elif gnn_type == 'gconvgru':
                self.gnns.append(GConvGRU(self.gnn_layers[layer], self.gnn_layers[layer+1], K=4))
            elif gnn_type=='gconvlstm':
                self.gnns.append(GConvLSTM(self.gnn_layers[layer], self.gnn_layers[layer+1], K=4))
            elif gnn_type =='chebnet':
                self.gnns.append(ChebConv(self.gnn_layers[layer], self.gnn_layers[layer+1], K=4))
            

        if global_pool == "sum":
            self.pool = global_add_pool
        elif global_pool == "mean":
            self.pool = global_mean_pool
        elif global_pool == "max":
            self.pool = global_max_pool
        elif global_pool == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((num_layer+1)*emb_dim,1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim,1))
        else:
            raise ValueError("Invalid graph pooling type.")
            
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(1, num_layer+1):
            self.batch_norms.append(torch.nn.BatchNorm1d(self.gnn_layers[layer]))
        
        # 2. mapping to dictionary coefficients (also transform to have a sum of 1)
        if self.JK == "concat":
            self.coef_linear = nn.Linear((num_layer+1)*emb_dim, latent_dim)
        else:
            self.coef_linear = nn.Linear(latent_dim, latent_dim)
    

        #--- f_theta network set up---
        self.net_adj = net_adj.to(device)
        self.mlp_coords = CoordMLP_(in_dim = latent_dim, N=self.N)

        self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net_adj.dim_hidden,
                num_layers = net_adj.num_layers
            ).to(device)
        

    def encode(self, x, edge_index,  batch):

        if self.flag_emb==1:
            x = self.x_embedding(x[:,0].to(torch.int64).to(self.device))
        #else:
        #    x = self.x_embedding(x.to(self.device))
        h_list = [x]
        for layer in range(self.num_layer):
            if self.gnn_type == 'gconvlstm':
                h, _ = self.gnns[layer](h_list[layer], edge_index)
                h = self.batch_norms[layer](h) # batch normalization after each conv
            else:
                h = self.gnns[layer](h_list[layer], edge_index)
                h = self.batch_norms[layer](h) # batch normalization after each conv

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)
        
        # aggregate node embedding from each layers 
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == 'gru':
            node_repr = torch.stack(h_list, dim =1)
            graph_rep, _ = self.gru(node_repr) 

        graph_rep = self.pool(node_representation, batch)  # node_representation  # torch.Size([2730, 2])

    
        # normalize over node representation
        node_representation = F.normalize(node_representation, p =2.0, dim = 1)

        return graph_rep, node_representation


    def decode(self, z, C_input, M, batch):
        '''
        z: latent variable [b, latent_dim]. Obtained from graph latent embedding [b, emb_dim], from which to obtain z, [b, latent_dim]

        C_input: input graph adjacency matrices [b,N,N] N = max number of nodes in the batch, use it as ground 
                 truth to compute the reconstruction loss 

        M: the number of nodes to sample from the 2d function to get the reconstruction graph;
           if M=0, sample the reconstructed graph adj with the same size as the input graph

           (Note: can modify so that we sample M to use random grid instead of regular grid)
        '''

        loss_b = []
        C_recon_list = []
        # get reconstructed graph function

        mods = []
        for i_b in range(C_input.shape[0]):
            # get graph size for this batch
            Nb = torch.sum(batch==i_b).detach().cpu()

            # get grid from sampling M points
            if M==0:
                h_recon = ot.unif(Nb).clone().detach()
                x = (torch.arange(Nb)+(1/2))/Nb
                y = (torch.arange(Nb)+(1/2))/Nb
            else:
                h_recon = torch.tensor(ot.unif(M)).clone().detach()
                x = (torch.arange(M)+(1/2))/M
                y = (torch.arange(M)+(1/2))/M

            xx,yy = torch.meshgrid(x, y) #,indexing='ij')
            mgrid = torch.stack([xx, yy],dim=-1)
            mgrid = rearrange(mgrid, 'h w c -> (h w) c')    # [74529, 2]

            z_tmp     = z[i_b,:] # n_dict 
            mods_tmp  = self.modulator(z_tmp)
            mods.append(mods_tmp)

            C_recon_tmp = self.net_adj(mgrid.to(self.device), mods_tmp)   # sirenNet...    [74529, 1]

            tmp_M = len(x)
            C_recon_tmp = torch.squeeze(rearrange(C_recon_tmp, '(h w) c -> h w c', h = tmp_M, w = tmp_M)) # [273, 273]

            # when training only half plane
            C_recon_tmp = torch.triu(C_recon_tmp, diagonal=1)
            C_recon_tmp = C_recon_tmp+torch.transpose(C_recon_tmp, 0, 1)   # [273, 273]

            # input measure
            h_input = ot.unif(Nb).clone().detach()
            C_input = C_input.to(device)

            # compute GW distance loss
            loss_tmp = ot.sliced_wasserstein_distance(C_recon_tmp.to(torch.float32).to(device), C_input[i_b,:Nb,:Nb].to(torch.float32).to(device), h_recon.to(torch.float32).to(device), h_input.to(torch.float32).to(device))
            # If one wants to use exact GW distance (much slower), replace the above line with the following line:
            # gromov_wasserstein2(C_recon_tmp.to(device),C_input[i_b,:Nb,:Nb].to(device),h_recon.to(torch.float32).to(device),h_input.to(torch.float32).to(device), max_iter=10200.0)

            loss_b.append(loss_tmp)
            C_recon_list.append(C_recon_tmp.detach().cpu().numpy())
        
        loss_b = torch.stack(loss_b)
        loss   = torch.mean(loss_b)

        return loss, z, C_recon_list, mods
    



    def sample(self,x,edge_index,batch,M):
        z = self.encode(x, edge_index, batch)
        x = (torch.arange(M)+(1/2))/M
        y = (torch.arange(M)+(1/2))/M
        xx,yy = torch.meshgrid(x,y)#,indexing='ij')
        mgrid=torch.stack([xx,yy],dim=-1)
        mgrid=rearrange(mgrid, 'h w c -> (h w) c')  

        C_list = []
        for i_b in range(z.shape[0]):
            mods_tmp  = self.modulator(z[i_b,:]) 
            C_recon_tmp = self.net(mgrid.to(self.device), mods_tmp)
            C_recon_tmp = torch.squeeze(rearrange(C_recon_tmp, '(h w) c -> h w c', h = M, w = M))

            # if only taking half
            C_recon_tmp = torch.triu(C_recon_tmp,diagonal=1)
            C_recon_tmp = C_recon_tmp+torch.transpose(C_recon_tmp,0,1)
            C_list.append(C_recon_tmp.detach().cpu().numpy())
        return C_list

    
    def forward(self, x, edge_index, batch, C_input, M):

        # Encoding...
        z_, node_representation = self.encode(x, edge_index, batch)

        # Reconstructing the adjacency matrix
        loss, z, C_recon_list, mods = self.decode(z_.to(device), C_input.to(device), M, batch.to(device))

        # Reconstructing the coordinates
        coords_pred = self.mlp_coords(z_.to(device))

        return loss, coords_pred
