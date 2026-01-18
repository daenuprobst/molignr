import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from torch_geometric.nn.conv import ChebConv
from models.layers import GIN_Conv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# Main AE structure =================================================================

class cIGNR(nn.Module):
    def __init__(self, input_card, emb_dim, latent_dim, num_layer, gnn_layers, gnn_type='gin', drop_ratio=0.,
                device=device,flag_emb=1, N =273, representation = "graph"):
        '''
        Simple autoencoder WITHOUT cIGNR

        Input:
        ---------------------------
        input_card: input cardinality (our graph doesn't come with feature, map positional encoding to a trainable embedding)
        emb_dim:   graph embedding dimension for the encoder GNN
        latent_dim:  dimension of the latent z (after mapping graph embedding to the latent code z)/ can simple use the graph embedding as latent without extra mapping

        num_layer: number of encoder gnn layers
        gnn_type:  choose from {gin, gcn, graphsage}
        global_pool: global pooling to obtain the final embedding
        '''
        
        super(cIGNR, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.device = device
        self.flag_emb = flag_emb
        self.gnn_type = gnn_type
        self.gnn_layers = gnn_layers
        self.representation = representation

        self.N = N
        
        #---initial embedding---
        # In case one wants to embed the input node features. Since we already have gnn in the encoder, I didn't use any embedding layer here. 
        # flag_emb is usually set to 0 
        if flag_emb==1:
            self.x_embedding = torch.nn.Embedding(input_card,emb_dim)
            torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        
        #---encoder network---
        # 1. gnn layers
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type== "gin":
                self.gnns.append(GIN_Conv(self.gnn_layers[layer], self.gnn_layers[layer+1]))
            elif gnn_type =='chebnet':
                self.gnns.append(ChebConv(self.gnn_layers[layer], self.gnn_layers[layer+1], K=4))
            
        self.pool = global_mean_pool
        # BatchNorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(1, num_layer+1):
            self.batch_norms.append(torch.nn.BatchNorm1d(self.gnn_layers[layer]))
        
        #--- f_theta network set up ---
        if self.representation == "graph":
            print(f"Graph representation....")
            self.mlp_coords = nn.Linear(latent_dim, self.N*3)
        elif self.representation == "node":
            print(f"Node representation....")
            self.mlp_coords = nn.Linear(latent_dim, 3)
        else:
            raise Exception("Please specify the representation type in the latent space - node or graph")


        

    def encode(self, x, edge_index,  batch):
        if self.flag_emb==1:
            x = self.x_embedding(x[:,0].to(torch.int64).to(self.device))
        
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h) # batch normalization after each conv

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.leaky_relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)
        
        # aggregate node embedding from each layers 
        node_representation = h_list[-1]

        graph_rep = self.pool(node_representation, batch)  # node_representation  # torch.Size([2730, 2])
  
        # normalize over node representation
        node_representation = F.normalize(node_representation, p =2.0, dim = 1)

        return graph_rep, node_representation


    
    def forward(self, x, edge_index, batch):

        # Encoding...
        z_, node_representation = self.encode(x, edge_index, batch)

        # Reconstructing the coordinates
        if self.representation =='graph':
            coords_pred = self.mlp_coords(z_.to(device))
            coords_pred = coords_pred.view(-1,3)
        elif self.representation == "node":
            coords_pred = self.mlp_coords(node_representation.to(device))

        return coords_pred
