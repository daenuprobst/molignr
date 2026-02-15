import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import ot
from ot.gromov import (
    gromov_wasserstein2,
    entropic_gromov_wasserstein2,
    fused_gromov_wasserstein2,
    entropic_fused_gromov_wasserstein2,
)

from torch_geometric.nn.models import GIN
from torch_geometric.nn import GlobalAttention, GINConv, GINEConv
from diffusion_distance import GraphDiffusionDistance, PermutationInvariantGDD
from networks import GaborWithModulation, SirenWithModulation, WIREWithModulation


class AtomGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_atom_types, num_layers=2, edge_feat_dim=0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.edge_feat_dim = edge_feat_dim

        # First layer
        nn_node = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        if edge_feat_dim > 0:
            nn_edge = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn_node, edge_dim=edge_feat_dim))
        else:
            self.convs.append(GINEConv(nn_node))  # no edge features

        # Additional layers
        for _ in range(num_layers - 1):
            nn_node = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if edge_feat_dim > 0:
                nn_edge = nn.Sequential(
                    nn.Linear(edge_feat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINEConv(nn_node, edge_dim=edge_feat_dim))
            else:
                self.convs.append(GINEConv(nn_node))

        self.fc_out = nn.Linear(hidden_dim, n_atom_types)

    def forward(self, x, edge_index, edge_attr=None, edge_weights=None):
        h = x
        for conv in self.convs:
            # 1. Scale attributes by weights if both exist
            current_edge_attr = edge_attr
            if edge_attr is not None and edge_weights is not None:
                # We unsqueeze weights to [E, 1] so they broadcast across [E, D] attributes
                current_edge_attr = edge_attr * edge_weights.unsqueeze(-1)

            # 2. Pass to GINEConv
            if self.edge_feat_dim > 0 and current_edge_attr is not None:
                h = conv(h, edge_index, current_edge_attr)
            else:
                h = conv(h, edge_index)

            h = F.relu(h)

        return self.fc_out(h)


class GINEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        emb_dim=64,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        assert num_layers >= 1

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(self._make_gin_conv(in_dim, emb_dim))
        self.norms.append(nn.LayerNorm(emb_dim))

        for _ in range(num_layers - 1):
            self.convs.append(self._make_gin_conv(emb_dim, emb_dim))
            self.norms.append(nn.LayerNorm(emb_dim))

        gate_nn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.SiLU(),
            nn.Linear(emb_dim // 2, 1),
        )
        self.pool = GlobalAttention(gate_nn)

        self.dropout = dropout

        self.reset_parameters()

    @staticmethod
    def _make_gin_conv(in_dim, out_dim):
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        return GINConv(mlp, eps=0.0, train_eps=True)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pool(x, batch)


class cIGNR(nn.Module):
    def __init__(
        self,
        n_attr,
        emb_dim,
        latent_dim,
        num_layer,
        hidden_dims=[128, 128, 64],
        n_atom_types=14,
        n_bond_types=3,
        valences=None,
        conditional_scalar=False,
        network_type="gabor",
        loss_type="diff",
        predict_property=False,
        prediction_task="regression",
        mmd=False,
        device="cpu",
    ):
        super().__init__()

        self.num_layer = num_layer
        self.n_attr = n_attr
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.device = device
        self.n_atom_types = n_atom_types
        self.n_bond_types = n_bond_types
        self.conditional_scalar = conditional_scalar
        self.network_type = network_type
        self.loss_type = loss_type
        self.predict_property = predict_property
        self.prediction_task = prediction_task
        self.mmd = mmd

        if valences is not None:
            self.valences = torch.tensor(valences, device=self.device)
        else:
            self.valences = None

        # GNN encoder
        self.gnn = GINEncoder(
            in_dim=n_attr,
            emb_dim=emb_dim,
            num_layers=num_layer,
            dropout=0.1,
        )

        self.degree_to_atom_embedding = nn.Embedding(10, 128)

        self.atom_gnn = AtomGNN(
            in_dim=latent_dim + 1,
            hidden_dim=128,
            n_atom_types=n_atom_types,
            num_layers=3,
            edge_feat_dim=self.n_bond_types,
        )

        self.fc_latent = nn.Linear(emb_dim, latent_dim)

        if self.network_type == "gabor":
            self.net = GaborWithModulation(
                input_dim=2,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                output_dim=1 + n_bond_types,
            )
        elif self.network_type == "siren":
            self.net = SirenWithModulation(
                input_dim=2,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                output_dim=1 + n_bond_types,
            )
        else:
            self.net = WIREWithModulation(
                in_features=2,
                latent_dim=latent_dim,
                hidden_features=hidden_dims[0],
                hidden_layers=len(hidden_dims),
                out_features=1 + n_bond_types,
                scale=20,
                first_omega_0=10,
                hidden_omega_0=10,
            )

        if conditional_scalar:
            self.scalar_encoder = nn.Sequential(
                nn.Linear(1, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            )

        if self.predict_property:
            if prediction_task == "classification":
                self.property_head = nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, latent_dim),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(latent_dim, latent_dim // 2),
                    nn.SiLU(),
                    nn.Linear(latent_dim // 2, 1),
                )
            else:
                self.property_head = nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(latent_dim, 1),
                )

        # self.gddist = GraphDiffusionDistance(self.device)
        self.gddist = GraphDiffusionDistance(self.device)
        self.triu_indices_cache = {}

        print(f"Initialized cIGNR:")
        print(f"  - Hidden dims: {hidden_dims}")
        print(f"  - Latent dim: {latent_dim}")

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
        matrix[i_indices, j_indices] = values
        return matrix + matrix.T

    def _make_symmetric_tensor(self, values, indices, grid_size, depth):
        i_indices, j_indices = indices
        tensor = torch.zeros(grid_size, grid_size, depth, device=self.device)
        tensor[i_indices, j_indices] = values
        # Average both directions for symmetry
        return (tensor + tensor.permute(1, 0, 2)) / 2.0

    def count_triangles_soft(self, C_recon):
        a_2 = torch.mm(C_recon, C_recon)
        a_3 = torch.mm(a_2, C_recon)

        num_triangles_soft = torch.trace(a_3) / 6.0

        return num_triangles_soft

    def chemical_validity_loss(self, C_recon, atom_logits, bond_type_matrix):
        losses = {}

        # --- Existing valence / sparsity code ---
        atom_probs = F.softmax(atom_logits, dim=-1)
        atom_types = torch.argmax(atom_probs, dim=-1)

        bond_probs = F.softmax(bond_type_matrix, dim=-1)
        bond_orders_range = torch.arange(
            1,
            self.n_bond_types + 1,
            device=bond_type_matrix.device,
            dtype=torch.float32,
        )
        bond_orders = (bond_probs * bond_orders_range.view(1, 1, -1)).sum(dim=-1)
        bond_orders = bond_orders * (
            1 - torch.eye(bond_orders.size(0), device=bond_orders.device)
        )

        valences = torch.sum(C_recon * bond_orders, dim=-1)

        if self.valences is not None:
            valence_tensor = torch.as_tensor(
                self.valences, device=C_recon.device, dtype=C_recon.dtype
            )
            safe_atom_types = torch.clamp(atom_types, 0, len(self.valences) - 1)
            max_valences = valence_tensor[safe_atom_types]
            violations = F.relu(valences - max_valences)
            violation_mask = violations > 0
            if violation_mask.any():
                losses["valence"] = violations[violation_mask].mean() * 0.5

        losses["sparsity"] = torch.mean(C_recon) * 0.1

        N = C_recon.size(0)
        if N >= 3:
            C2 = torch.matmul(C_recon, C_recon)
            num_triangles = torch.trace(torch.matmul(C2, C_recon)) / 6.0

            max_triangles = (N * (N - 1) * (N - 2)) / 6.0
            triangle_density = num_triangles / max(max_triangles, 1.0)

            losses["three_ring_penalty"] = triangle_density
        else:
            losses["three_ring_penalty"] = torch.tensor(0.0, device=C_recon.device)

        return losses

    def _compute_losses(self, C_recon, bond_type_matrix, atom_logits, targets, i_b, Nb):
        C_gt = targets["adj"][i_b, :Nb, :Nb].to(self.device)
        C_recon_slice = C_recon[:Nb, :Nb]
        C_gt_norm = C_gt / (C_gt.max() + 1e-8)
        C_recon_norm = C_recon_slice / (C_recon_slice.max() + 1e-8)

        # Adjacency loss
        adjacency_loss = 0.0
        if self.loss_type == "diff":
            # adjacency_loss = self.gddist.compute_gdd_vectorized(
            #     C_gt_norm, C_recon_norm, num_samples=20
            # )
            adjacency_loss = F.mse_loss(C_gt_norm, C_recon_norm)

        else:
            h_input = torch.from_numpy(ot.unif(Nb)).to(self.device)
            h_recon = torch.from_numpy(ot.unif(Nb)).to(self.device)
            adjacency_loss = gromov_wasserstein2(
                C_recon_norm, C_gt_norm, h_recon, h_input
            ).clone()

        # # Bond type loss
        # bond_types_gt = targets["bond_types"][i_b, :Nb, :Nb].to(self.device)
        # edge_exists = C_gt > 0.5
        # bond_type_loss = torch.tensor(0.0, device=self.device)

        if edge_exists.sum() > 0:
            masked_bond_logits = bond_type_matrix[:Nb, :Nb][edge_exists]
            masked_bond_gt = bond_types_gt[edge_exists]
            bond_type_loss = nn.functional.cross_entropy(
                masked_bond_logits, masked_bond_gt, reduction="sum"
            )

            bond_type_loss = bond_type_loss / edge_exists.sum()

        # Atom type loss
        # atom_types_gt = targets["atom_types"][i_b, :Nb].to(self.device)
        # atom_type_loss = nn.functional.cross_entropy(
        #     atom_logits[:Nb], atom_types_gt, reduction="sum"
        # )
        # atom_type_loss = atom_type_loss / Nb

        # print(adjacency_loss.item(), bond_type_loss.item(), atom_type_loss.item())
        return (
            # adjacency_loss + bond_type_loss + atom_type_loss,
            adjacency_loss,
            adjacency_loss,
            bond_type_loss,
            atom_type_loss,
        )

    def encode(self, x, edge_index, batch, scalar=None):
        graph_embedding = self.gnn(x.float(), edge_index, batch)
        z = self.fc_latent(graph_embedding)

        if scalar is not None and self.conditional_scalar:
            s = scalar.view(-1, 1).float()
            z_scalar = self.scalar_encoder(s)
            z = z + z_scalar

        return z

    def compute_node_features(self, z_tmp, C_recon, Nb):
        features = []

        # Basic features
        soft_degrees = C_recon.sum(dim=-1, keepdim=True)
        features.append(soft_degrees)
        features.append(torch.sqrt(soft_degrees + 1e-8))  # sqrt instead of square

        # Graph features
        C2 = torch.matmul(C_recon, C_recon)
        C3_diag = (C2 * C_recon).sum(dim=-1, keepdim=True)
        possible_triangles = soft_degrees * (soft_degrees - 1) + 1e-8
        clustering = C3_diag / possible_triangles
        features.append(clustering)

        neighbor_degrees = torch.matmul(C_recon, soft_degrees)
        avg_neighbor_degree = neighbor_degrees / (soft_degrees + 1e-8)
        features.append(avg_neighbor_degree)

        degree_2hop = C2.sum(dim=-1, keepdim=True)
        features.append(degree_2hop)

        degree_centrality = soft_degrees / (Nb - 1 + 1e-8)
        features.append(degree_centrality)

        # Simplified eigenvector centrality
        ones = torch.ones(Nb, 1, device=C_recon.device)
        eig_approx = torch.matmul(C_recon, ones)
        eig_centrality = eig_approx / (eig_approx.max() + 1e-8)
        features.append(eig_centrality)

        # Graph latent
        z_expanded = z_tmp.unsqueeze(0).expand(Nb, -1)
        features.append(z_expanded)

        # Concatenate and normalize
        node_features = torch.cat(features, dim=-1)

        return node_features

    def spectral_loss(self, C_recon_list, targets):
        losses = []

        for C_recon, C_gt, s in zip(C_recon_list, targets["adj"], targets["sizes"]):
            C_gt = C_gt[:s, :s].to(self.device)
            eps = 1e-6
            C_gt_stable = C_gt + eps * torch.eye(C_gt.size(0), device=C_gt.device)
            C_recon_stable = C_recon + eps * torch.eye(
                C_recon.size(0), device=C_recon.device
            )

            eig_gt = torch.linalg.eigvalsh(C_gt_stable)
            eig_recon = torch.linalg.eigvalsh(C_recon_stable)
            losses.append(F.mse_loss(eig_recon, eig_gt))
        return torch.stack(losses).mean()

    def degree_loss(self, C_recon_list, targets):
        losses = []

        for C_recon, C_gt, s in zip(C_recon_list, targets["adj"], targets["sizes"]):
            C_gt = C_gt[:s, :s].to(self.device)

            deg_recon = C_recon.sum(dim=-1)
            deg_gt = C_gt.sum(dim=-1)

            losses.append(F.mse_loss(deg_recon, deg_gt))

        return torch.stack(losses).mean()

    # def decode(self, z, M, batch=None, targets=None, tf_ratio=1.0):
    #     losses = [] if targets is not None else None
    #     adj_losses = [] if targets is not None else None
    #     bond_losses = [] if targets is not None else None
    #     atom_losses = [] if targets is not None else None

    #     C_recon_list = []
    #     bond_type_logits_list = []
    #     atom_type_logits_list = []

    #     B = z.shape[0]

    #     for i_b in range(B):
    #         z_tmp = z[i_b]

    #         if targets is not None and M == 0:
    #             Nb = targets["sizes"][i_b]
    #         else:
    #             Nb = M

    #         triu_mgrid, i_indices, j_indices = self.get_triu_meshgrid(Nb)

    #         outputs = self.net(triu_mgrid, z_tmp)
    #         adj_logits = outputs[:, 0]
    #         bond_logits = outputs[:, 1:]

    #         triu_adj_values = torch.sigmoid(adj_logits)
    #         C_recon = self._make_symmetric_matrix(
    #             triu_adj_values, (i_indices, j_indices), Nb
    #         )
    #         bond_type_matrix = self._make_symmetric_tensor(
    #             bond_logits, (i_indices, j_indices), Nb, self.n_bond_types
    #         )

    #         if targets is not None:
    #             C_gt = targets["adj"][i_b, :Nb, :Nb].to(self.device).float()
    #             C_tf = tf_ratio * C_gt + (1 - tf_ratio) * C_recon
    #         else:
    #             C_tf = C_recon

    #         degrees = C_tf.sum(dim=-1).clamp(0, 9).long()
    #         degree_embeddings = self.degree_to_atom_embedding(degrees)

    #         z_expanded = z_tmp.expand(degree_embeddings.size(0), -1)

    #         node_features = torch.cat([degree_embeddings, z_expanded], dim=-1)

    #         rows, cols = torch.meshgrid(
    #             torch.arange(Nb), torch.arange(Nb), indexing="ij"
    #         )

    #         edge_index = torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=0).to(
    #             self.device
    #         )

    #         edge_weights = C_tf.reshape(-1)
    #         edge_attr = bond_type_matrix.reshape(-1, bond_type_matrix.size(-1))

    #         soft_degrees = C_recon.sum(dim=-1, keepdim=True)
    #         node_features = torch.cat([soft_degrees, z_expanded], dim=-1)

    #         atom_logits = self.atom_gnn(
    #             node_features,
    #             edge_index,
    #             edge_attr=edge_attr,
    #             edge_weights=edge_weights,
    #         )

    #         # Compute loss if targets provided
    #         if targets is not None:
    #             loss, adj_loss, bond_loss, atom_loss = self._compute_losses(
    #                 C_recon, bond_type_matrix, atom_logits, targets, i_b, Nb
    #             )

    #             losses.append(loss)
    #             adj_losses.append(adj_loss)
    #             bond_losses.append(bond_loss)
    #             atom_losses.append(atom_loss)

    #         # Collect outputs
    #         C_recon_list.append(C_recon)
    #         bond_type_logits_list.append(bond_type_matrix)
    #         atom_type_logits_list.append(atom_logits)

    #     mean_loss = torch.stack(losses).mean() if losses else None
    #     mean_adj_loss = torch.stack(adj_losses).mean() if adj_losses else None
    #     mean_bond_loss = torch.stack(bond_losses).mean() if bond_losses else None
    #     mean_atom_loss = torch.stack(atom_losses).mean() if atom_losses else None

    #     return (
    #         mean_loss,
    #         mean_adj_loss,
    #         mean_bond_loss,
    #         mean_atom_loss,
    #         z,
    #         C_recon_list,
    #         bond_type_logits_list,
    #         atom_type_logits_list,
    #     )

    def decode(self, z, M, batch=None, targets=None, tf_ratio=1.0):
        losses = [] if targets is not None else None
        adj_losses = [] if targets is not None else None
        bond_losses = [] if targets is not None else None
        atom_losses = [] if targets is not None else None

        C_recon_list = []
        bond_type_logits_list = []
        atom_type_logits_list = []

        B = z.shape[0]

        for i_b in range(B):
            z_tmp = z[i_b]

            if targets is not None and M == 0:
                Nb = targets["sizes"][i_b]
            else:
                Nb = M

            triu_mgrid, i_indices, j_indices = self.get_triu_meshgrid(Nb)

            outputs = self.net(triu_mgrid, z_tmp)
            adj_logits = outputs[:, 0]

            triu_adj_values = torch.sigmoid(adj_logits)
            C_recon = self._make_symmetric_matrix(
                triu_adj_values, (i_indices, j_indices), Nb
            )

            if targets is not None:
                C_gt = targets["adj"][i_b, :Nb, :Nb].to(self.device)
                C_recon_slice = C_recon[:Nb, :Nb]
                C_gt_norm = C_gt / (C_gt.max() + 1e-8)
                C_recon_norm = C_recon_slice / (C_recon_slice.max() + 1e-8)

                if self.loss_type == "diff":
                    # adjacency_loss = self.gddist.compute_gdd_vectorized(
                    #     C_gt, C_recon_norm, num_samples=20
                    # )
                    adjacency_loss = F.mse_loss(C_gt, C_recon_norm)

                else:
                    h_input = torch.from_numpy(ot.unif(Nb)).to(self.device)
                    h_recon = torch.from_numpy(ot.unif(Nb)).to(self.device)
                    adjacency_loss = gromov_wasserstein2(
                        C_recon_norm, C_gt_norm, h_recon, h_input
                    ).clone()

                losses.append(adjacency_loss)

            # Collect outputs
            C_recon_list.append(C_recon)

        mean_loss = torch.stack(losses).mean() if losses else None

        return (
            mean_loss,
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            z,
            C_recon_list,
            bond_type_logits_list,
            atom_type_logits_list,
        )

    # def forward(self, x, edge_index, batch, targets, M, epoch, scalar=None):
    #     z = self.encode(x, edge_index, batch, scalar)

    #     tf_start_epoch = 0
    #     tf_end_epoch = 1000

    #     if epoch < tf_start_epoch:
    #         tf_ratio = 1.0
    #     elif epoch >= tf_end_epoch:
    #         tf_ratio = 0.0
    #     else:
    #         tf_ratio = 1.0 - (epoch - tf_start_epoch) / (tf_end_epoch - tf_start_epoch)

    #     start = time.perf_counter()
    #     (
    #         recon_loss,
    #         adj_recon_loss,
    #         bond_recon_loss,
    #         atom_recon_loss,
    #         z,
    #         C_recon_list,
    #         bond_type_logits_list,
    #         atom_type_logits_list,
    #     ) = self.decode(z, M, batch, targets, tf_ratio=tf_ratio)

    #     end = time.perf_counter()
    #     # print(f"Reconstruct time: {(end - start) * 1000:.2f}ms")

    #     loss = recon_loss

    #     spec_loss = torch.tensor(0.0, device=self.device)
    #     deg_loss = torch.tensor(0.0, device=self.device)

    #     if targets is not None:
    #         # Compute raw losses
    #         spec_loss_raw = self.spectral_loss(C_recon_list, targets)
    #         deg_loss_raw = self.degree_loss(C_recon_list, targets)

    #         # Compute adaptive weights
    #         with torch.no_grad():
    #             recon_scale = recon_loss.detach()
    #             spec_scale = spec_loss_raw.detach() + 1e-8
    #             deg_scale = deg_loss_raw.detach() + 1e-8

    #         lambda_spectral = 5.0 * (recon_scale / spec_scale)
    #         lambda_degree = 5.0 * (recon_scale / deg_scale)

    #         # Apply weights for gradient updates
    #         spec_loss_weighted = spec_loss_raw * lambda_spectral
    #         deg_loss_weighted = deg_loss_raw * lambda_degree

    #         loss = loss + spec_loss_weighted + deg_loss_weighted

    #     loss_components = {
    #         "reconstruction": recon_loss,
    #         "spectral_loss": (
    #             spec_loss_raw if targets is not None else torch.tensor(0.0)
    #         ),
    #         "deg_loss": (deg_loss_raw if targets is not None else torch.tensor(0.0)),
    #         "spectral_loss_weighted": (
    #             spec_loss_weighted if targets is not None else torch.tensor(0.0)
    #         ),
    #         "deg_loss_weighted": (
    #             deg_loss_weighted if targets is not None else torch.tensor(0.0)
    #         ),
    #         "adj_reconstruction": adj_recon_loss,
    #         "bond_reconstruction": bond_recon_loss,
    #         "atom_reconstruction": atom_recon_loss,
    #         "chemistry": torch.tensor(0.0),
    #         "property": torch.tensor(0.0),
    #     }

    #     if targets is not None:
    #         start = time.perf_counter()
    #         if self.valences is not None:
    #             chem_loss = 0.0

    #             for i, (C_recon, atom_logits, bond_matrix) in enumerate(
    #                 zip(C_recon_list, atom_type_logits_list, bond_type_logits_list)
    #             ):
    #                 chem_losses = self.chemical_validity_loss(
    #                     C_recon, atom_logits, bond_matrix
    #                 )
    #                 for key, val in chem_losses.items():
    #                     chem_loss = chem_loss + val / len(C_recon_list)

    #             chem_loss = chem_loss * 1000
    #             loss += chem_loss
    #             loss_components["chemistry"] = chem_loss

    #         end = time.perf_counter()
    #     # print(f"Chemloss time: {(end - start) * 1000:.2f}ms")

    #     if self.predict_property and "property" in targets:
    #         y_pred = self.property_head(z)
    #         y_true = targets["property"].to(y_pred.device)

    #         if y_true.dim() == 1:
    #             y_true = y_true.unsqueeze(-1)

    #         if self.prediction_task == "classification":
    #             if "pos_weight" in targets:
    #                 prop_loss = F.binary_cross_entropy_with_logits(
    #                     y_pred, y_true, pos_weight=targets["pos_weight"]
    #                 )
    #             else:
    #                 prop_loss = F.binary_cross_entropy_with_logits(
    #                     y_pred, y_true.float()
    #                 )
    #         else:
    #             prop_loss = F.mse_loss(y_pred, y_true)

    #         # loss = loss + prop_loss

    #         prop_warmup = 50
    #         prop_max_weight = 1.0

    #         prop_weight = min(1.0, epoch / prop_warmup) * prop_max_weight
    #         loss = loss + prop_weight * prop_loss

    #         loss_components["property"] = prop_loss

    #     reconstructions = {
    #         "adj": C_recon_list,
    #         "bond_types": bond_type_logits_list,
    #         "atom_types": atom_type_logits_list,
    #     }

    #     return loss, z, reconstructions, loss_components

    def graphon_constraints(self, z, num_samples=100):
        if z.dim() == 1:
            z = z.unsqueeze(0)

        losses = []
        for z_tmp in z:
            # Sample random points
            u = torch.rand(num_samples, 1, device=self.device)
            v = torch.rand(num_samples, 1, device=self.device)

            coords_uv = torch.cat([u, v], dim=1)
            coords_vu = torch.cat([v, u], dim=1)  # Swapped

            # Evaluate graphon
            out_uv = torch.sigmoid(self.net(coords_uv, z_tmp)[:, 0])
            out_vu = torch.sigmoid(self.net(coords_vu, z_tmp)[:, 0])

            # Symmetry loss
            symmetry_loss = F.mse_loss(out_uv, out_vu)

            # Boundedness is automatic with sigmoid, but penalize saturation
            saturation_loss = (
                F.relu(out_uv - 0.95).mean()  # Penalize values > 0.95
                + F.relu(0.05 - out_uv).mean()  # Penalize values < 0.05
            )

            losses.append(symmetry_loss + 0.1 * saturation_loss)

        return torch.stack(losses).mean()

    def contrastive_loss(self, z, targets, temperature=0.1):
        # Compute graph-level features
        degrees = torch.stack(
            [
                targets["adj"][i, :s, :s].to(self.device).sum(dim=-1).mean()
                for i, s in enumerate(targets["sizes"])
            ]
        )

        # Normalize
        z_norm = F.normalize(z, dim=-1)

        # Similarity matrix
        similarity = torch.mm(z_norm, z_norm.T) / temperature

        # Degree similarity as target
        degree_sim = 1.0 / (
            1.0 + torch.abs(degrees.unsqueeze(1) - degrees.unsqueeze(0))
        )

        # InfoNCE-style loss
        loss = F.mse_loss(torch.sigmoid(similarity), degree_sim)

        return loss

    def forward(self, x, edge_index, batch, targets, M, epoch, scalar=None):
        z = self.encode(x, edge_index, batch, scalar)

        tf_start_epoch = 0
        tf_end_epoch = 1000

        if epoch < tf_start_epoch:
            tf_ratio = 1.0
        elif epoch >= tf_end_epoch:
            tf_ratio = 0.0
        else:
            tf_ratio = 1.0 - (epoch - tf_start_epoch) / (tf_end_epoch - tf_start_epoch)

        start = time.perf_counter()
        (
            recon_loss,
            adj_recon_loss,
            bond_recon_loss,
            atom_recon_loss,
            z,
            C_recon_list,
            bond_type_logits_list,
            atom_type_logits_list,
        ) = self.decode(z, M, batch, targets, tf_ratio=tf_ratio)

        end = time.perf_counter()
        # print(f"Reconstruct time: {(end - start) * 1000:.2f}ms")

        loss = recon_loss

        graphon_loss = self.graphon_constraints(z, num_samples=50)
        loss = loss + 0.5 * graphon_loss

        # In forward (only if batch size > 1):
        # if z.shape[0] > 1:
        #     contrast_loss = self.contrastive_loss(z, targets)
        #     loss = loss + 0.5 * contrast_loss

        # spec_loss = torch.tensor(0.0, device=self.device)
        # deg_loss = torch.tensor(0.0, device=self.device)

        # if targets is not None:
        #     # Compute raw losses
        #     spec_loss_raw = self.spectral_loss(C_recon_list, targets)
        #     deg_loss_raw = self.degree_loss(C_recon_list, targets)

        #     # # Compute adaptive weights
        #     # with torch.no_grad():
        #     #     recon_scale = recon_loss.detach()
        #     #     spec_scale = spec_loss_raw.detach() + 1e-8
        #     #     deg_scale = deg_loss_raw.detach() + 1e-8

        #     # lambda_spectral = 1.0 * (recon_scale / spec_scale)
        #     # lambda_degree = 1.0 * (recon_scale / deg_scale)

        #     lambda_spectral = 10.0
        #     lambda_degree = 10.0

        #     # Apply weights for gradient updates
        #     spec_loss_weighted = spec_loss_raw * lambda_spectral
        #     deg_loss_weighted = deg_loss_raw * lambda_degree

        #     loss = loss + spec_loss_weighted + deg_loss_weighted

        loss_components = {
            "reconstruction": recon_loss,
            # "contrast_loss": contrast_loss,
            # "spectral_loss": (
            #     spec_loss_raw if targets is not None else torch.tensor(0.0)
            # ),
            # "deg_loss": (deg_loss_raw if targets is not None else torch.tensor(0.0)),
            # "spectral_loss_weighted": (
            #     spec_loss_weighted if targets is not None else torch.tensor(0.0)
            # ),
            # "deg_loss_weighted": (
            #     deg_loss_weighted if targets is not None else torch.tensor(0.0)
            # ),
            "adj_reconstruction": adj_recon_loss,
            "bond_reconstruction": bond_recon_loss,
            "atom_reconstruction": atom_recon_loss,
            "chemistry": torch.tensor(0.0),
            "property": torch.tensor(0.0),
        }

        if self.predict_property and "property" in targets:
            y_pred = self.property_head(z)
            y_true = targets["property"].to(y_pred.device)

            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(-1)

            if self.prediction_task == "classification":
                if "pos_weight" in targets:
                    prop_loss = F.binary_cross_entropy_with_logits(
                        y_pred, y_true, pos_weight=targets["pos_weight"]
                    )
                else:
                    prop_loss = F.binary_cross_entropy_with_logits(
                        y_pred, y_true.float()
                    )
            else:
                prop_loss = F.mse_loss(y_pred, y_true)

            # loss = loss + prop_loss

            prop_warmup = 50
            prop_max_weight = 1.0

            prop_weight = min(1.0, epoch / prop_warmup) * prop_max_weight
            loss = loss + prop_weight * prop_loss

            loss_components["property"] = prop_loss

        reconstructions = {
            "adj": C_recon_list,
            "bond_types": [],
            "atom_types": {},
        }

        return loss, z, reconstructions, loss_components

    def sample(
        self,
        x=None,
        edge_index=None,
        batch=None,
        M=0,
        latents=None,
        scalar=None,
        return_property=False,
    ):
        # --- Use provided latents OR encode ---
        if latents is not None:
            z = latents
            if not isinstance(z, torch.Tensor):
                z = torch.tensor(z, device=self.device)
            z = z.to(self.device)
            if z.dim() == 1:
                z = z.unsqueeze(0)
        else:
            assert (
                x is not None and edge_index is not None and batch is not None
            ), "x, edge_index, batch must be provided if latents is None"
            z = self.encode(x, edge_index, batch, scalar)

        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=self.device)

        # loss, _, C_recon_list, bond_type_logits_list, atom_type_logits_list = (
        #     self.decode(z, M, batch, targets=None)
        # )

        (
            recon_loss,
            adj_recon_loss,
            bond_recon_loss,
            atom_recon_loss,
            z,
            C_recon_list,
            bond_type_logits_list,
            atom_type_logits_list,
        ) = self.decode(z, M, batch)

        loss = recon_loss

        reconstructions = {
            "adj": C_recon_list,
            "bond_types": bond_type_logits_list,
            "atom_types": atom_type_logits_list,
        }

        if self.predict_property and return_property:
            y_pred = self.property_head(z)
            return loss, z, reconstructions, y_pred

        return loss, z, reconstructions
