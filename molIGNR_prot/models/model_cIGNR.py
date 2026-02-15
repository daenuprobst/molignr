import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import ot
from torch_geometric.nn.conv import ChebConv

from models.siren_pytorch import *
from models.layers import GIN_Conv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# Main AE structure =================================================================
class cIGNR(nn.Module):
    def __init__(self, net_adj, latent_dim, num_layer, gnn_layers, gnn_type='gin', drop_ratio=0.,
                device=device, N =2191):
        '''
        Encode each input graph into a latent code z of dimension [latent_dim]; 
        z is used to condition the training of the MLP function f_theta (mapping R2->[0,1])

        Input:
        ---------------------------
        latent_dim:  dimension of the latent z (after mapping graph embedding to the latent code z)/ can simple use the graph embedding as latent without extra mapping

        num_layer: number of encoder gnn layers
        gnn_type:  choose from {gin, gcn, graphsage}
        '''
        
        super(cIGNR, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        # self.emb_dim = emb_dim
        self.device = device
        self.gnn_type = gnn_type
        self.gnn_layers = gnn_layers

        self.N = N
        

        #---encoder network---
        # 1. gnn layers
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GIN_Conv(self.gnn_layers[layer], self.gnn_layers[layer+1]))
            elif gnn_type == 'chebnet':
                self.gnns.append(ChebConv(self.gnn_layers[layer], self.gnn_layers[layer+1], K=4))

        self.pool = global_mean_pool

        ######## for GNNs 
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(1, num_layer+1):
            self.batch_norms.append(torch.nn.BatchNorm1d(self.gnn_layers[layer]))
        
        ## coordinate decoder
        self.mlp_coords = nn.Linear(latent_dim, N*3)
        
        #--- f_theta network set up---
        self.net_adj = net_adj.to(device)

        self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net_adj.dim_hidden,
                num_layers = net_adj.num_layers
            ).to(device)
        

    def encode(self, x, edge_index,  batch):
        h = x
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index)
            h = self.batch_norms[layer](h) # batch normalization after each conv
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
        
        graph_rep = self.pool(h, batch)

        # normalize over node representation
        node_representation = F.normalize(h, p =2.0, dim = 1)

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
            
            with torch.amp.autocast('cuda', enabled = False):
               loss_tmp = ot.sliced_wasserstein_distance(C_recon_tmp.to(torch.float32).to(device), C_input[i_b,:Nb,:Nb].to(torch.float32).to(device), h_recon.to(torch.float32).to(device), h_input.to(torch.float32).to(device))
            
            loss_b.append(loss_tmp)
        
        loss_b = torch.stack(loss_b)
        loss   = torch.mean(loss_b)

        return loss, z
    



    
    def forward(self, x, edge_index, batch, C_input, M):
        z_, node_representation = self.encode(x, edge_index, batch)

        loss, _= self.decode(z_.to(device), C_input.to(device), M, batch.to(device))

        coords_pred = self.mlp_coords(z_.to(device))
        coords_pred = coords_pred.view(-1,3)

        return loss, coords_pred
