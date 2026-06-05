import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import ot
from torch_geometric.nn.conv import ChebConv

from models.siren_pytorch import *
from models.layers import GIN_Conv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from geomloss import SamplesLoss

mmd_loss = SamplesLoss(loss="gaussian", blur=1.0)

from torch.utils.checkpoint import checkpoint
import gc

class HyperINR(nn.Module):
    """
    Hypernetwork INR: z generates ALL weights and biases of the INR.
    The INR itself only takes grid coordinates as input.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims, output_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build list of (in_features, out_features) for each INR layer
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layer_shapes = [(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        
        # Total number of parameters the INR needs
        total_params = sum(d_in * d_out + d_out for d_in, d_out in self.layer_shapes)
        
        # Hypernetwork: maps z -> all INR parameters
        self.hypernet = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8,8),
            nn.ReLU(),
            nn.Linear(8, total_params),
        )
        
        # Small init on last layer so generated weights start small
        nn.init.zeros_(self.hypernet[-1].bias)
        nn.init.normal_(self.hypernet[-1].weight, std=0.01)
        
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in hidden_dims])
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, z):
        # z: (latent_dim,) or (B, latent_dim)
        # x: (N, input_dim) or (B, N, input_dim)
        
        # Generate all INR parameters from z
        params = self.hypernet(z)  # (total_params,)
        
        # Split params into weights and biases for each layer
        weights, biases = self._split_params(params)
        
        # Forward pass through the generated INR
        for i, (W, b) in enumerate(zip(weights, biases)):
            # W: (in_features, out_features), b: (out_features,)
            if x.dim() == 2:
                # x: (N, in_features)
                x = x @ W + b
            elif x.dim() == 3:
                # x: (B, N, in_features) — batched
                x = torch.einsum('bn i, i o -> bn o', x, W) + b
            
            # Activation for all layers except last
            if i < len(weights) - 1:
                x = torch.sin(x)  # SIREN-style activation works well for INRs
                x = self.norms[i](x)
                x = self.dropout(x)
        
        return x
    
    def _split_params(self, params):
        weights = []
        biases = []
        offset = 0
        for d_in, d_out in self.layer_shapes:
            w_size = d_in * d_out
            b_size = d_out
            
            W = params[offset:offset + w_size].view(d_in, d_out)
            offset += w_size
            
            b = params[offset:offset + b_size]
            offset += b_size
            
            weights.append(W)
            biases.append(b)
        
        return weights, biases


class CoordinateDecoder(nn.Module):
    def __init__(self, latent_dim, N):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),    # first expand from the bottleneck
            nn.ReLU(),
            nn.Linear(32, 64),  # then to larger representation
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, N*3)
        )
    
    def forward(self, z):
        return self.decoder(z)

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
        self.net_adj = net_adj

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
        self.mlp_coords = CoordinateDecoder(latent_dim, N)   #nn.Linear(latent_dim, N*3)
        
        #--- f_theta network set up---
        # self.net_adj = HyperINR(2, latent_dim, hidden_dims= [12,12,12], output_dim=1) #net_adj.to(device)

        self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net_adj.dim_hidden,
                num_layers = net_adj.num_layers
            ).to(device)
        
        self.triu_indices_cache = {}

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

    def get_triu_meshgrid(self, grid_size):
        cache_key = (grid_size, str(self.device))
        if cache_key not in self.triu_indices_cache:
            i_indices, j_indices = torch.triu_indices(
                grid_size, grid_size, offset=1, device=self.device
            )

            x_coords = (i_indices.float() + 0.5) / grid_size
            y_coords = (j_indices.float() + 0.5) / grid_size
            triu_mgrid = torch.stack([x_coords, y_coords], dim=-1)

            self.triu_indices_cache[cache_key] = (triu_mgrid, i_indices, j_indices)

        return self.triu_indices_cache[cache_key]
    


    def _make_symmetric_matrix(self, values, indices, grid_size):
        i_indices, j_indices = indices
        matrix = torch.zeros(grid_size, grid_size, device=self.device)
        matrix[i_indices, j_indices] = values.float()
        return matrix + matrix.T


    def _decode_single(self, mod, triu_mgrid, i_indices, j_indices,
                        Nb, C_input_single, h_recon):
        """Process one sample. This is the function that gets checkpointed."""
        outputs = self.net_adj(triu_mgrid.to(device), mod)
        # outputs = self.decode_in_chunks(triu_mgrid, z_tmp)
        adj_logits = torch.sigmoid(outputs)

        C_recon_tmp = torch.squeeze(rearrange(adj_logits, '(h w) c -> h w c', h = Nb, w = Nb)) # [273, 273]

        # when training only half plane
        C_recon_tmp = torch.triu(C_recon_tmp, diagonal=1)
        C_recon_tmp = C_recon_tmp+torch.transpose(C_recon_tmp, 0, 1)   # [273, 273]

        # C_recon_tmp = self._make_symmetric_matrix(adj_logits, (i_indices, j_indices), Nb)

        h_input = torch.tensor(ot.unif(Nb)).clone().detach()

        with torch.amp.autocast('cuda', enabled=False):
            loss_tmp = ot.sliced_wasserstein_distance(
                C_recon_tmp.to(torch.float32).to(device),
                C_input_single[:Nb, :Nb].to(torch.float32).to(device),
                h_recon.to(torch.float32).to(device),
                h_input.to(torch.float32).to(device),
            )
        return loss_tmp
    

    def decode_in_chunks(self, coords, z, chunk_size=50000):
        outs = []
        for start in range(0, coords.shape[0], chunk_size):
            end = start + chunk_size
            coords_chunk = coords[start:end]
            out_chunk = self.net_adj(coords_chunk, z)
            outs.append(out_chunk)
        return torch.cat(outs, dim=0)
    
    def decode(self, z, C_input, M, batch):
        loss_b = []

        for i_b in range(C_input.shape[0]):
            if M == 0:
                Nb = torch.sum(batch == i_b).detach().cpu()
                h_recon = torch.tensor(ot.unif(Nb)).clone().detach()

                x = (torch.arange(Nb)+(1/2))/Nb
                y = (torch.arange(Nb)+(1/2))/Nb
            else:
                h_recon = torch.tensor(ot.unif(M)).clone().detach()
                Nb = M

                x = (torch.arange(M)+(1/2))/M
                y = (torch.arange(M)+(1/2))/M

            z_tmp = z[i_b, :]
            mods_tmp  = self.modulator(z_tmp)
            
            xx,yy = torch.meshgrid(x, y) #,indexing='ij')
            mgrid = torch.stack([xx, yy],dim=-1)
            triu_mgrid = rearrange(mgrid, 'h w c -> (h w) c')    # [74529, 2]


            # NEW (throws away intermediates, recomputes during backward):
            loss_tmp = self._decode_single(mods_tmp, triu_mgrid,0,0,Nb, C_input[i_b], h_recon)
           
            # outputs = self.net_adj(triu_mgrid.to(device), z_tmp)
            
            # outputs = self.decode_in_chunks(triu_mgrid, mods_tmp)
            # adj_logits = torch.sigmoid(outputs[:, 0])
            # C_recon_tmp = self._make_symmetric_matrix(adj_logits, (i_indices, j_indices), Nb)

            # print(f"C_recon_tmp.shape : {C_recon_tmp.shape}")

           
            loss_b.append(loss_tmp)

            del triu_mgrid, z_tmp #i_indices, j_indices, z_tmp

        loss_b = torch.stack(loss_b)
        loss = torch.mean(loss_b)
        return loss, z

    """
    def decode(self, z, C_input, M, batch):
        
        '''
        z: latent variable [b, latent_dim]. Obtained from graph latent embedding [b, emb_dim], from which to obtain z, [b, latent_dim]

        C_input: input graph adjacency matrices [b,N,N] N = max number of nodes in the batch, use it as ground 
                 truth to compute the reconstruction loss 

        M: the number of nodes to sample from the 2d function to get the reconstruction graph;
           if M=0, sample the reconstructed graph adj with the same size as the input graph

           (Note: can modify so that we sample M to use random grid instead of regular grid)
        '''
        gc.collect()
        torch.cuda.empty_cache()

        loss_b = []
        mods = []
        for i_b in range(C_input.shape[0]):
            # get graph size for this batch


            if M == 0:
                Nb = torch.sum(batch==i_b).detach().cpu()
                h_recon = ot.unif(Nb).clone().detach()
            else:
                h_recon = torch.tensor(ot.unif(M)).clone().detach()
                Nb = M


            z_tmp     = z[i_b,:] # n_dict 

            triu_mgrid, i_indices, j_indices = self.get_triu_meshgrid(Nb)
            
            # mods_tmp  = self.modulator(z_tmp)
            # mods.append(mods_tmp)
            # C_recon_tmp = self.net_adj(mgrid.to(self.device), mods_tmp)   # sirenNet...    [74529, 1]
            
            outputs = self.net_adj(triu_mgrid.to(device), z_tmp)
            adj_logits = torch.sigmoid(outputs[:, 0])
            C_recon_tmp = self._make_symmetric_matrix(adj_logits, (i_indices, j_indices), Nb)
            #  C_recon_tmp = torch.squeeze(rearrange(C_recon_tmp, '(h w) c -> h w c', h = tmp_M, w = tmp_M)) # [273, 273]

            # when training only half plane
            # C_recon_tmp = torch.triu(C_recon_tmp, diagonal=1)
            # C_recon_tmp = C_recon_tmp+torch.transpose(C_recon_tmp, 0, 1)   # [273, 273]

            # input measure
            h_input = ot.unif(Nb).clone().detach()
            C_input = C_input.to(device)
            
            with torch.amp.autocast('cuda', enabled = False):
               loss_tmp = ot.sliced_wasserstein_distance(C_recon_tmp.to(torch.float32).to(device), C_input[i_b,:Nb,:Nb].to(torch.float32).to(device), h_recon.to(torch.float32).to(device), h_input.to(torch.float32).to(device))
               #mmd_loss(C_recon_tmp.to(torch.float32).to(device), C_input[i_b,:Nb,:Nb].to(torch.float32).to(device))
               # print(f"Loss tmp : {loss_tmp.item()}")
               # ot.sliced_wasserstein_distance(C_recon_tmp.to(torch.float32).to(device), C_input[i_b,:Nb,:Nb].to(torch.float32).to(device), h_recon.to(torch.float32).to(device), h_input.to(torch.float32).to(device))
            loss_b.append(loss_tmp)
        
        loss_b = torch.stack(loss_b)
        loss   = torch.mean(loss_b)

        return loss, z
    """



    def _mem(self, tag):
        print(f"[{tag}] GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")


    def forward(self, x, edge_index, batch, C_input, M):

        # self._mem("before encode")
        z_, node_representation = self.encode(x, edge_index, batch)
        # self._mem("after encode")
        # self._mem(f"C_input shape: {C_input.shape}")

        del node_representation
        torch.cuda.empty_cache()

        loss, _= self.decode(z_.to(device), C_input.to(device), M, batch.to(device))
        # self._mem("after decode")
        coords_pred = self.mlp_coords(z_.to(device))
        # self._mem("after coords")
        coords_pred = coords_pred.view(-1,3)


        return loss, coords_pred












"""
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
"""