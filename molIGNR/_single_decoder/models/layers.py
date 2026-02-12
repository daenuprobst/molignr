import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


# Note:
# - Baseline MLP decoder
# - GNN encoder 
    
class GIN_Conv(MessagePassing):
    """
    GIN without edge feature concatenation
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        input_layer (bool): whethe the GIN conv is applied to input layer or not. (Input node labels are uniform...)
    """
    def __init__(self, emb_dim, hidden_dim, aggr = "add"):
        super(GIN_Conv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), 
                                       torch.nn.ReLU(), 
                                       torch.nn.Linear(2*emb_dim, hidden_dim))
        self.aggr = aggr

    def forward(self, x, edge_index):
        
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))[0]
        
        return self.propagate(edge_index, aggr=self.aggr, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return self.mlp(aggr_out.float())
