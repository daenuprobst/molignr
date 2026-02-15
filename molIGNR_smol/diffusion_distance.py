import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class GraphDiffusionDistance:
    def __init__(self, device: Optional[str] = None, temperature: float = 10.0):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.temperature = temperature

    def compute_graph_laplacian(
        self, adj_matrix: torch.Tensor, normalized: bool = True
    ) -> torch.Tensor:
        adj_matrix = adj_matrix.to(self.device)

        # Ensure adjacency is symmetric and non-negative
        adj_matrix = torch.clamp(adj_matrix, min=0.0)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2

        # Compute degree - add epsilon BEFORE any operations
        degree = adj_matrix.sum(dim=1)
        eps = 1e-8
        degree = degree + eps  # Prevent division by zero

        if normalized:
            # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
            deg_inv_sqrt = torch.pow(degree, -0.5)

            # Safety check - should not have inf/nan now, but double-check
            deg_inv_sqrt = torch.clamp(deg_inv_sqrt, max=1e6)
            deg_inv_sqrt[torch.isnan(deg_inv_sqrt) | torch.isinf(deg_inv_sqrt)] = 0.0

            deg_inv_sqrt_mat = torch.diag(deg_inv_sqrt)
            laplacian = (
                torch.eye(adj_matrix.shape[0], device=self.device)
                - deg_inv_sqrt_mat @ adj_matrix @ deg_inv_sqrt_mat
            )
        else:
            # Unnormalized Laplacian: L = D - A
            degree_matrix = torch.diag(degree)
            laplacian = degree_matrix - adj_matrix

        # Ensure Laplacian is symmetric (fix numerical errors)
        laplacian = (laplacian + laplacian.T) / 2

        # Clamp eigenvalues to prevent extreme values in matrix exponential
        # This doesn't change the Laplacian much but prevents numerical overflow
        laplacian = torch.clamp(laplacian, min=-1e6, max=1e6)

        return laplacian

    def laplacian_exponential_kernel_vectorized(
        self, laplacian: torch.Tensor, t_values: torch.Tensor, max_t: float = 10.0
    ) -> torch.Tensor:
        # Ensure symmetry
        laplacian = (laplacian + laplacian.T) / 2

        # Scale down large t values to prevent overflow
        # The matrix exponential of -t*L can overflow if eigenvalues of L are large
        t_values = torch.clamp(t_values, max=max_t)

        # Compute -t * L for all t values
        L_batch = -t_values.unsqueeze(1).unsqueeze(2) * laplacian.unsqueeze(0)

        # Clamp to prevent extreme values before exponentiation
        L_batch = torch.clamp(L_batch, min=-50, max=50)

        # Compute matrix exponentials
        try:
            kernels = torch.matrix_exp(L_batch)
        except RuntimeError as e:
            print(
                f"Warning: matrix_exp failed, using eigenvalue decomposition fallback"
            )
            # Fallback: use eigenvalue decomposition
            kernels = self._safe_matrix_exp_batch(L_batch)

        # # Check for NaNs/Infs and replace with identity if needed
        # if torch.any(torch.isnan(kernels)) or torch.any(torch.isinf(kernels)):
        #     print("Warning: NaN or Inf in kernel matrices, using identity fallback")
        #     N = laplacian.shape[0]
        #     eye = torch.eye(N, device=self.device)
        #     kernels = eye.unsqueeze(0).expand(len(t_values), -1, -1).clone()

        return kernels

    def _safe_matrix_exp_batch(self, L_batch: torch.Tensor) -> torch.Tensor:
        results = []
        for L in L_batch:
            try:
                # Ensure symmetry for eigendecomposition
                L_sym = (L + L.T) / 2
                eigvals, eigvecs = torch.linalg.eigh(L_sym)

                # Clamp eigenvalues to prevent overflow in exp
                eigvals = torch.clamp(eigvals, min=-50, max=50)

                # Compute exp(L) = V @ diag(exp(λ)) @ V^T
                exp_eigvals = torch.exp(eigvals)
                result = eigvecs @ torch.diag(exp_eigvals) @ eigvecs.T
                results.append(result)
            except RuntimeError:
                # Ultimate fallback: identity matrix
                N = L.shape[0]
                results.append(torch.eye(N, device=L.device))

        return torch.stack(results)

    def compute_gdd_vectorized(
        self,
        adj1: torch.Tensor,
        adj2: torch.Tensor,
        t_min: float = 0.01,
        t_max: float = 10.0,
        num_samples: int = 20,
        normalized: bool = True,
    ) -> torch.Tensor:
        # Ensure matrices are valid
        # if torch.any(torch.isnan(adj1)) or torch.any(torch.isnan(adj2)):
        #     print("Warning: Input adjacency matrices contain NaN")
        #     return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

        # if torch.any(torch.isinf(adj1)) or torch.any(torch.isinf(adj2)):
        #     print("Warning: Input adjacency matrices contain Inf")
        #     return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

        # Compute Laplacians with stability fixes
        L1 = self.compute_graph_laplacian(adj1, normalized)
        L2 = self.compute_graph_laplacian(adj2, normalized)

        # # Check for NaN/Inf in Laplacians
        # if torch.any(torch.isnan(L1)) or torch.any(torch.isnan(L2)):
        #     print("Warning: Laplacian contains NaN")
        #     return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

        # if torch.any(torch.isinf(L1)) or torch.any(torch.isinf(L2)):
        #     print("Warning: Laplacian contains Inf")
        #     return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

        # Sample time points
        t_values = torch.logspace(
            torch.log10(torch.tensor(t_min, device=self.device)),
            torch.log10(torch.tensor(t_max, device=self.device)),
            num_samples,
            device=self.device,
        )

        # Compute all kernel matrices with stability
        K1_batch = self.laplacian_exponential_kernel_vectorized(
            L1, t_values, max_t=t_max
        )
        K2_batch = self.laplacian_exponential_kernel_vectorized(
            L2, t_values, max_t=t_max
        )

        # Compute all Frobenius norms
        diff_batch = K1_batch - K2_batch
        gdd_values = torch.sqrt(torch.sum(diff_batch * diff_batch, dim=(1, 2)) + 1e-8)

        # # Check for NaN in GDD values
        # if torch.any(torch.isnan(gdd_values)):
        #     print("Warning: GDD values contain NaN")
        #     return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

        weights = F.softmax(gdd_values * self.temperature, dim=0)
        max_gdd = torch.sum(weights * gdd_values)

        return max_gdd


class PermutationInvariantGDD:
    def __init__(self, device: Optional[str] = None, temperature: float = 10.0):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.temperature = temperature

    def compute_graph_laplacian_eigenvalues(
        self, adj_matrix: torch.Tensor, normalized: bool = True
    ) -> torch.Tensor:
        adj_matrix = adj_matrix.to(self.device)

        adj_matrix = torch.clamp(adj_matrix, min=0.0)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2

        degree = adj_matrix.sum(dim=1)
        eps = 1e-8

        if normalized:
            deg_inv_sqrt = torch.pow(degree + eps, -0.5)
            deg_inv_sqrt_mat = torch.diag(deg_inv_sqrt)
            laplacian = (
                torch.eye(adj_matrix.shape[0], device=self.device)
                - deg_inv_sqrt_mat @ adj_matrix @ deg_inv_sqrt_mat
            )
        else:
            laplacian = torch.diag(degree) - adj_matrix

        laplacian = (laplacian + laplacian.T) / 2

        eigvals = torch.linalg.eigvalsh(laplacian)

        return torch.clamp(eigvals, min=0.0)

    def compute_gdd_vectorized(
        self,
        adj1: torch.Tensor,
        adj2: torch.Tensor,
        t_min: float = 0.01,
        t_max: float = 10.0,
        num_samples: int = 20,
        normalized: bool = True,
    ) -> torch.Tensor:
        evals1 = self.compute_graph_laplacian_eigenvalues(adj1, normalized)
        evals2 = self.compute_graph_laplacian_eigenvalues(adj2, normalized)

        if evals1.shape[0] != evals2.shape[0]:
            n1, n2 = evals1.shape[0], evals2.shape[0]
            max_n = max(n1, n2)
            if n1 < max_n:
                evals1 = torch.cat(
                    [evals1, torch.full((max_n - n1,), 1e6, device=self.device)]
                )
            else:
                evals2 = torch.cat(
                    [evals2, torch.full((max_n - n2,), 1e6, device=self.device)]
                )

        t_values = torch.logspace(
            torch.log10(torch.tensor(t_min, device=self.device)),
            torch.log10(torch.tensor(t_max, device=self.device)),
            num_samples,
            device=self.device,
        )

        evals1_exp = torch.exp(-t_values.unsqueeze(1) * evals1.unsqueeze(0))
        evals2_exp = torch.exp(-t_values.unsqueeze(1) * evals2.unsqueeze(0))

        diff = evals1_exp - evals2_exp
        gdd_values = torch.sqrt(torch.sum(diff**2, dim=1) + 1e-8)

        weights = F.softmax(gdd_values * self.temperature, dim=0)
        max_gdd = torch.sum(weights * gdd_values)

        return max_gdd
