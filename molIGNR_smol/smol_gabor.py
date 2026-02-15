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
from torch_geometric.nn import GlobalAttention, GINEConv
from diffusion_distance import GraphDiffusionDistance
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

        if valences is not None:
            self.valences = torch.tensor(valences, device=self.device)
        else:
            self.valences = None

        # GNN encoder
        self.gnn = GIN(n_attr, 128, num_layer, emb_dim)

        gate_nn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2), nn.ReLU(), nn.Linear(emb_dim // 2, 1)
        )
        self.pooling = GlobalAttention(gate_nn)

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

        graph_feature_dim = 5

        self.graph_feature_encoder = nn.Sequential(
            nn.Linear(graph_feature_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )

        self.atom_type_predictor = nn.Sequential(
            nn.Linear(32 + latent_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], n_atom_types),
        )

        if conditional_scalar:
            self.scalar_encoder = nn.Sequential(
                nn.Linear(1, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            )

        if self.predict_property:
            self.property_head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(latent_dim, 1),
            )

        self.gddist = GraphDiffusionDistance(self.device)
        self.triu_indices_cache = {}

        print(f"Initialized cIGNR with gabor network:")
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

    # def chemical_validity_loss(self, C_recon, atom_logits, bond_type_matrix):
    #     losses = {}

    #     atom_probs = F.softmax(atom_logits, dim=-1)
    #     atom_types = torch.argmax(atom_probs, dim=-1)

    #     bond_orders = torch.argmax(bond_type_matrix, dim=-1) + 1

    #     valences = torch.sum(C_recon * bond_orders.float(), dim=-1)

    #     if self.valences is not None:
    #         valence_tensor = torch.as_tensor(
    #             self.valences, device=C_recon.device, dtype=C_recon.dtype
    #         )

    #         num_classes = len(self.valences)
    #         safe_atom_types = torch.clamp(atom_types, 0, num_classes - 1)

    #         max_valences = valence_tensor[safe_atom_types]
    #         violations = F.relu(valences - max_valences)

    #         violation_mask = violations > 0
    #         if violation_mask.any():
    #             losses["valence"] = violations[violation_mask].mean() * 0.5

    #     losses["sparsity"] = torch.mean(C_recon) * 0.1

    #     C2 = torch.matmul(C_recon, C_recon)
    #     num_triangles = torch.trace(torch.matmul(C2, C_recon)) / 6.0
    #     losses["three_ring_penalty"] = num_triangles * 1.0

    #     return losses

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

        # --- Normalized triangle penalty ---
        N = C_recon.size(0)
        if N >= 3:
            C2 = torch.matmul(C_recon, C_recon)
            num_triangles = torch.trace(torch.matmul(C2, C_recon)) / 6.0

            # Total possible triangles = N choose 3
            max_triangles = (N * (N - 1) * (N - 2)) / 6.0
            triangle_density = num_triangles / max(
                max_triangles, 1.0
            )  # avoid div by zero

            # Add to losses
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
        if self.loss_type == "diff":
            loss = self.gddist.compute_gdd_vectorized(
                C_gt_norm, C_recon_norm, num_samples=20
            )
        elif self.loss_type == "sw":
            loss = self.sliced_wasserstein(
                C_recon_norm,
                C_gt_norm,
            )
        else:
            h_input = torch.from_numpy(ot.unif(Nb)).to(self.device)
            h_recon = torch.from_numpy(ot.unif(Nb)).to(self.device)

            loss = gromov_wasserstein2(
                C_recon_norm,
                C_gt_norm,
                h_recon,
                h_input,
            ).clone()

        # Bond type loss
        bond_types_gt = targets["bond_types"][i_b, :Nb, :Nb].to(self.device)

        edge_exists = C_gt > 0.5
        if edge_exists.sum() > 0:
            masked_bond_logits = bond_type_matrix[:Nb, :Nb][edge_exists]
            masked_bond_gt = bond_types_gt[edge_exists]
            loss += nn.functional.cross_entropy(masked_bond_logits, masked_bond_gt)

        # Atom type loss
        atom_types_gt = targets["atom_types"][i_b, :Nb].to(self.device)
        loss += nn.functional.cross_entropy(atom_logits[:Nb], atom_types_gt)

        return loss

    def encode(self, x, edge_index, batch, scalar=None):
        node_embeddings = self.gnn(x.float(), edge_index)
        graph_embedding = self.pooling(node_embeddings, batch)
        z = self.fc_latent(graph_embedding)

        if scalar is not None and self.conditional_scalar:
            s = scalar.view(-1, 1).float()
            z_scalar = self.scalar_encoder(s)
            z = z + z_scalar

        return z

    def compute_node_graph_features(self, C_recon, bond_type_matrix):
        N = C_recon.size(0)

        # Feature 1: Node degree (normalized) - OK, differentiable
        degrees = C_recon.sum(dim=-1, keepdim=True)
        max_degree = max(degrees.max().item(), 1.0)
        normalized_degrees = degrees / max_degree

        # Feature 2: Local clustering coefficient - OK, differentiable
        C2 = torch.matmul(C_recon, C_recon)
        possible_triangles = degrees * (degrees - 1) / 2.0
        possible_triangles = torch.clamp(possible_triangles, min=1.0)
        triangles = torch.diagonal(C2).unsqueeze(-1) / 2.0
        clustering = triangles / possible_triangles
        clustering = torch.nan_to_num(clustering, 0.0)

        # Feature 3: Average bond type - USE SOFT VERSION
        # Instead of argmax, use softmax probabilities
        bond_probs = F.softmax(bond_type_matrix, dim=-1)  # [N, N, n_bond_types]

        # Expected bond order (differentiable)
        bond_orders = torch.arange(
            1, self.n_bond_types + 1, device=C_recon.device, dtype=torch.float32
        )
        expected_bond_order = (bond_probs * bond_orders.view(1, 1, -1)).sum(
            dim=-1
        )  # [N, N]

        bond_sum = (C_recon * expected_bond_order).sum(dim=-1, keepdim=True)
        avg_bond_type = bond_sum / torch.clamp(degrees, min=1.0)
        avg_bond_type = torch.nan_to_num(avg_bond_type, 0.0)
        avg_bond_type = avg_bond_type / self.n_bond_types

        # Feature 4: Eigenvector centrality - OK, differentiable
        x = degrees / (degrees.sum() + 1e-8)
        for _ in range(3):
            x = torch.matmul(C_recon, x)
            x = x / (torch.norm(x) + 1e-8)
        centrality = x

        # Feature 5: Average degree of neighbors - OK, differentiable
        neighbor_degrees = torch.matmul(C_recon, degrees)
        avg_neighbor_degree = neighbor_degrees / torch.clamp(degrees, min=1.0)
        avg_neighbor_degree = torch.nan_to_num(avg_neighbor_degree, 0.0)
        avg_neighbor_degree = avg_neighbor_degree / max_degree

        features = torch.cat(
            [
                normalized_degrees,
                clustering,
                avg_bond_type,
                centrality,
                avg_neighbor_degree,
            ],
            dim=-1,
        )

        return features

    # def decode(self, z, M, batch, targets=None):
    #     losses = [] if targets is not None else None
    #     C_recon_list = []
    #     bond_type_logits_list = []
    #     atom_type_logits_list = []

    #     batch_sizes = torch.bincount(batch, minlength=z.shape[0])

    #     for i_b in range(z.shape[0]):
    #         Nb = batch_sizes[i_b].item()
    #         grid_size = M if M != 0 else Nb

    #         # Get upper triangular meshgrid
    #         triu_mgrid, i_indices, j_indices = self.get_triu_meshgrid(grid_size)
    #         z_tmp = z[i_b, :]

    #         outputs = self.net(triu_mgrid, z_tmp)
    #         adj_logits = outputs[:, 0]
    #         bond_logits = outputs[:, 1:]

    #         # Construct symmetric matrices
    #         triu_adj_values = torch.sigmoid(adj_logits)
    #         C_recon = self._make_symmetric_matrix(
    #             triu_adj_values, (i_indices, j_indices), grid_size
    #         )
    #         bond_type_matrix = self._make_symmetric_tensor(
    #             bond_logits, (i_indices, j_indices), grid_size, self.n_bond_types
    #         )

    #         # Generate node features
    #         _, atom_logits = self._generate_vertex_features(grid_size, z_tmp)

    #         # Compute losses if targets provided
    #         if targets is not None:
    #             loss = self._compute_losses(
    #                 C_recon, bond_type_matrix, atom_logits, targets, i_b, Nb
    #             )
    #             losses.append(loss)

    #         C_recon_list.append(C_recon)
    #         bond_type_logits_list.append(bond_type_matrix)
    #         atom_type_logits_list.append(atom_logits)

    #     mean_loss = torch.stack(losses).mean() if losses else None
    #     return mean_loss, z, C_recon_list, bond_type_logits_list, atom_type_logits_list

    def decode(self, z, M, batch, targets=None, tf_ratio=0.0):

        losses = [] if targets is not None else None
        C_recon_list = []
        bond_type_logits_list = []
        atom_type_logits_list = []

        batch_sizes = torch.bincount(batch, minlength=z.shape[0])

        for i_b in range(z.shape[0]):
            Nb = batch_sizes[i_b].item()
            grid_size = M if M != 0 else Nb

            # Get upper triangular meshgrid (OFF-DIAGONAL ONLY)
            triu_mgrid, i_indices, j_indices = self.get_triu_meshgrid(grid_size)
            z_tmp = z[i_b, :]

            # Predict edges and bonds from graphon
            outputs = self.net(triu_mgrid, z_tmp)
            adj_logits = outputs[:, 0]
            bond_logits = outputs[:, 1:]

            # Construct symmetric matrices
            triu_adj_values = torch.sigmoid(adj_logits)
            C_recon = self._make_symmetric_matrix(
                triu_adj_values, (i_indices, j_indices), grid_size
            )
            bond_type_matrix = self._make_symmetric_tensor(
                bond_logits, (i_indices, j_indices), grid_size, self.n_bond_types
            )

            # Compute graph features for each node
            # graph_features_raw = self.compute_node_graph_features(
            #     C_recon, bond_type_matrix
            # )
            # graph_features = self.graph_feature_encoder(graph_features_raw)

            # # Predict atom types conditioned on graph features + latent
            # z_expanded = z_tmp.unsqueeze(0).expand(grid_size, -1)
            # atom_input = torch.cat([graph_features, z_expanded], dim=-1)
            # atom_logits = self.atom_type_predictor(atom_input)
            # atom_logits = atom_logits[:Nb]

            if targets is not None:
                C_gt = targets["adj"][i_b, :Nb, :Nb].to(self.device).float()
                C_tf = tf_ratio * C_gt + (1 - tf_ratio) * C_recon
            else:
                C_tf = C_recon

            degrees = C_tf.sum(dim=-1).clamp(0, 9).long()
            degree_embeddings = self.degree_to_atom_embedding(degrees)

            z_expanded = z_tmp.expand(degree_embeddings.size(0), -1)

            node_features = torch.cat([degree_embeddings, z_expanded], dim=-1)

            rows, cols = torch.meshgrid(
                torch.arange(Nb), torch.arange(Nb), indexing="ij"
            )

            edge_index = torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=0).to(
                self.device
            )

            edge_weights = C_tf.reshape(-1)
            edge_attr = bond_type_matrix.reshape(-1, bond_type_matrix.size(-1))

            soft_degrees = C_recon.sum(dim=-1, keepdim=True)
            node_features = torch.cat([soft_degrees, z_expanded], dim=-1)

            atom_logits = self.atom_gnn(
                node_features,
                edge_index,
                edge_attr=edge_attr,
                edge_weights=edge_weights,
            )

            # Compute losses if targets provided
            if targets is not None:
                loss = self._compute_losses(
                    C_recon, bond_type_matrix, atom_logits, targets, i_b, Nb
                )
                losses.append(loss)

            C_recon_list.append(C_recon)
            bond_type_logits_list.append(bond_type_matrix)
            atom_type_logits_list.append(atom_logits)

        mean_loss = torch.stack(losses).mean() if losses else None
        return mean_loss, z, C_recon_list, bond_type_logits_list, atom_type_logits_list

    def sliced_wasserstein(self, X, Y, n_proj=16):
        d = X.shape[1]
        loss = 0.0

        for _ in range(n_proj):
            theta = torch.randn(d, device=X.device)
            theta = theta / torch.norm(theta)

            proj_X = X @ theta
            proj_Y = Y @ theta

            proj_X, _ = torch.sort(proj_X)
            proj_Y, _ = torch.sort(proj_Y)

            loss += torch.mean((proj_X - proj_Y) ** 2)

        return loss / n_proj

    def forward(self, x, edge_index, batch, targets, M, epoch, scalar=None):
        loss_components = {}

        start = time.perf_counter()
        z = self.encode(x, edge_index, batch, scalar)
        end = time.perf_counter()
        # print(f"Encode time: {(end - start) * 1000:.2f}ms")

        start = time.perf_counter()
        loss, z, C_recon_list, bond_type_logits_list, atom_type_logits_list = (
            self.decode(z, M, batch, targets)
        )
        end = time.perf_counter()
        # print(f"Decode time: {(end - start) * 1000:.2f}ms")

        loss_components = {"reconstruction": loss}

        # if targets is not None:
        #     start = time.perf_counter()
        #     if self.valences is not None:
        #         chem_loss = 0.0

        #         for i, (C_recon, atom_logits, bond_matrix) in enumerate(
        #             zip(C_recon_list, atom_type_logits_list, bond_type_logits_list)
        #         ):
        #             chem_losses = self.chemical_validity_loss(
        #                 C_recon, atom_logits, bond_matrix
        #             )
        #             for key, val in chem_losses.items():
        #                 chem_loss = chem_loss + val / len(C_recon_list)

        #         loss += chem_loss
        #         loss_components["chemistry"] = chem_loss

        #     end = time.perf_counter()
        # print(f"Chemloss time: {(end - start) * 1000:.2f}ms")

        if (
            self.predict_property
            and "property" in targets
            and epoch > 20
            and epoch < 200
        ):
            # z_prop = z.detach()
            y_pred = self.property_head(z)
            y_true = targets["property"].to(y_pred.device)

            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(-1)

            prop_loss = F.mse_loss(y_pred, y_true)
            loss = loss + prop_loss

            loss_components["property"] = prop_loss

        reconstructions = {
            "adj": C_recon_list,
            "bond_types": bond_type_logits_list,
            "atom_types": atom_type_logits_list,
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
        if latents is not None:
            z = (
                latents
                if isinstance(latents, torch.Tensor)
                else torch.tensor(latents, device=self.device)
            )
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z = z.to(self.device)
        else:
            z = self.encode(x, edge_index, batch, scalar)

        loss, z, C_recon_list, bond_type_logits_list, atom_type_logits_list = (
            self.decode(
                z,
                M,
                (
                    batch
                    if batch is not None
                    else torch.zeros(1, dtype=torch.long, device=self.device)
                ),
                targets=None,
            )
        )

        reconstructions = {
            "adj": C_recon_list,
            "bond_types": bond_type_logits_list,
            "atom_types": atom_type_logits_list,
        }

        if self.predict_property and return_property:
            y_pred = self.property_head(z)
            return loss, z, reconstructions, y_pred

        return loss, z, reconstructions
