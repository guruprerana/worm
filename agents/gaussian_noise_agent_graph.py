"""
AgentGraph subclass with straight-line graph and correlated Gaussian noise losses.

This is a synthetic test environment for evaluating quantile estimation algorithms
under controlled correlation structures.
"""
from typing import List, Tuple, Optional, Dict
import numpy as np

from agents.agent_graph import AgentGraph


class GaussianNoiseAgentGraph(AgentGraph):
    """
    AgentGraph with straight-line topology and correlated Gaussian noise losses.

    Graph structure:
    - Straight line: 0 -> 1 -> 2 -> ... -> (graph_length - 1)
    - graph_length vertices, (graph_length - 1) edges

    Noise structure:
    - Each edge produces losses that are Gaussian noise
    - Noise = rho * shared_noise + sqrt(1 - rho^2) * individual_noise
    - shared_noise ~ N(mean, std): same across all edges for each sample
    - individual_noise ~ N(mean, std): unique per edge
    - rho ∈ [0, 1] controls correlation:
      * rho = 1: fully correlated (100% correlation across edges)
      * rho = 0: fully independent (0% correlation)
      * rho = 0.5: moderate correlation

    The coefficient sqrt(1 - rho^2) ensures constant variance:
    Var(total) = rho^2 * Var(shared) + (1 - rho^2) * Var(individual) = std^2
    """

    def __init__(
        self,
        graph_length: int,
        rho: float = 0.5,
        mean: float = 0.0,
        std: float = 1.0,
        noise_seed: int = 42,
        cache_save_file: Optional[str] = None,
    ):
        """
        Initialize GaussianNoiseAgentGraph.

        Args:
            graph_length: Number of vertices in the straight-line graph.
                         Results in (graph_length - 1) edges.
            rho: Correlation coefficient (0 ≤ ρ ≤ 1).
                 Controls correlation between edge losses in the same sample.
            mean: Mean of the Gaussian noise distribution
            std: Standard deviation of the Gaussian noise distribution
            noise_seed: Random seed for noise generation (for reproducibility)
            cache_save_file: Path to cache file for samples (optional)
        """
        # Create straight-line graph adjacency list
        # 0 -> 1 -> 2 -> ... -> (graph_length - 1)
        adj_lists = [[i + 1] if i < graph_length - 1 else [] for i in range(graph_length)]

        # Initialize parent class
        super().__init__(adj_lists=adj_lists, cache_save_file=cache_save_file)

        # Validate parameters
        if not 0 <= rho <= 1:
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        if graph_length < 2:
            raise ValueError(f"graph_length must be at least 2, got {graph_length}")

        self.graph_length = graph_length
        self.rho = rho
        self.mean = mean
        self.std = std
        self.noise_seed = noise_seed
        self.rng = np.random.RandomState(noise_seed)

        # Cache for shared noise per sample batch
        # Key: n_samples, Value: shared noise array of shape (n_samples,)
        self.shared_noise_cache: Dict[int, np.ndarray] = {}

        print(f"Initialized GaussianNoiseAgentGraph:")
        print(f"  Graph: straight line with {graph_length} vertices, {graph_length - 1} edges")
        print(f"  Noise: N({mean}, {std}²)")
        print(f"  ρ={rho} (correlation coefficient)")
        print(f"  Noise = {rho:.3f}*shared + {np.sqrt(1 - rho**2):.3f}*individual")
        if cache_save_file:
            print(f"  Cache file: {cache_save_file}")

    def _generate_shared_noise(self, n_samples: int) -> np.ndarray:
        """
        Generate shared Gaussian noise component for a batch of samples.

        This noise is the same across all edges for each sample, but different
        across samples. Uses a dedicated RNG seeded consistently.

        Args:
            n_samples: Number of samples

        Returns:
            Array of shape (n_samples,) with Gaussian noise ~ N(mean, std)
        """
        # Use cached shared noise if available
        if n_samples in self.shared_noise_cache:
            return self.shared_noise_cache[n_samples]

        # Generate new shared noise
        shared_rng = np.random.RandomState(self.noise_seed)
        shared_noise = shared_rng.normal(self.mean, self.std, size=n_samples)

        # Cache for reuse
        self.shared_noise_cache[n_samples] = shared_noise
        return shared_noise

    def _generate_edge_noise(
        self,
        path: List[int],
        target_vertex: int,
        n_samples: int
    ) -> np.ndarray:
        """
        Generate correlated Gaussian noise for an edge.

        Combines shared noise (correlated across edges) and individual noise
        (uncorrelated) using the formula:

        noise = rho * shared + sqrt(1 - rho^2) * individual

        This maintains constant variance: Var(noise) = std^2

        Args:
            path: The path up to (but not including) the target vertex
            target_vertex: The target vertex
            n_samples: Number of samples

        Returns:
            Array of shape (n_samples,) with correlated Gaussian noise
        """
        # Generate shared noise component (same for all edges in this batch)
        shared_noise = self._generate_shared_noise(n_samples)

        # Generate individual noise component (unique to this edge)
        # Use edge-specific seed for reproducibility
        edge_seed = self.noise_seed + hash((tuple(path), target_vertex)) % (2**31)
        edge_rng = np.random.RandomState(edge_seed)
        individual_noise = edge_rng.normal(self.mean, self.std, size=n_samples)

        # Combine with correlation coefficient
        # The sqrt(1 - rho^2) factor ensures constant variance
        total_noise = (
            self.rho * shared_noise +
            np.sqrt(1 - self.rho**2) * individual_noise
        )

        return total_noise

    def sample(
        self,
        target_vertex: int,
        n_samples: int,
        path: List[int],
        path_samples: list,
    ) -> Tuple[list, List[float]]:
        """
        Sample losses for an edge.

        Since this is a synthetic graph, path_samples are not used
        (there's no actual state to track). We just generate correlated
        Gaussian noise as the losses.

        Args:
            target_vertex: Target vertex to reach
            n_samples: Number of samples to draw
            path: Path taken so far
            path_samples: Samples from previous edge (unused for synthetic graph)

        Returns:
            Tuple of (next_path_samples, losses)
            - next_path_samples: list of None (no actual state)
            - losses: list of float with Gaussian noise
        """
        # Validate path
        if len(path) == 0:
            raise ValueError("Path must contain at least the starting vertex")
        if path[-1] >= self.graph_length:
            raise ValueError(f"Path vertex {path[-1]} exceeds graph_length {self.graph_length}")
        if target_vertex >= self.graph_length:
            raise ValueError(f"Target vertex {target_vertex} exceeds graph_length {self.graph_length}")

        # Generate correlated Gaussian noise
        noise = self._generate_edge_noise(path, target_vertex, n_samples)

        # Convert to list of losses
        losses = noise.tolist()

        # No actual state samples in this synthetic graph
        next_path_samples = [None] * n_samples

        return next_path_samples, losses

    def get_all_paths(self) -> List[List[int]]:
        """
        Get all paths from source (vertex 0) to sink (last vertex).

        For a straight-line graph, there's only one path: [0, 1, 2, ..., graph_length-1]

        Returns:
            List containing the single path through the graph
        """
        return [list(range(self.graph_length))]

    def get_noise_info(self) -> Dict:
        """
        Get information about noise configuration.

        Returns:
            Dictionary with noise parameters
        """
        return {
            "graph_length": self.graph_length,
            "n_edges": self.graph_length - 1,
            "rho": self.rho,
            "mean": self.mean,
            "std": self.std,
            "noise_seed": self.noise_seed,
            "noise_type": "gaussian_correlated",
            "description": f"noise = {self.rho:.3f}*shared + {np.sqrt(1 - self.rho**2):.3f}*individual",
            "theoretical_variance": self.std**2,
            "theoretical_correlation": self.rho,
        }
