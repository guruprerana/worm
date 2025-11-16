"""
RLAgentGraph subclass that adds correlated uniform noise to losses.

Noise is a mixture of shared (correlated) and individual (uncorrelated) components,
controlled by correlation coefficient rho.
"""
from typing import List, Optional, Dict, Tuple
import numpy as np

from agents.rl_agent_graph import RLAgentGraph


class RLAgentGraphCorrelatedNoise(RLAgentGraph):
    """
    RLAgentGraph with correlated uniform noise added to losses.

    Noise structure:
    - Total noise = rho * shared_noise + (1-rho) * individual_noise
    - shared_noise ~ Uniform(noise_min, noise_max): same across all edges in path
    - individual_noise ~ Uniform(noise_min, noise_max): unique per edge
    - rho ∈ [0, 1] controls correlation:
      * rho = 1: fully shared (100% correlation across edges)
      * rho = 0: fully individual (0% correlation)
      * rho = 0.5: equal mix of shared and individual
    """

    def __init__(
        self,
        spec_graph: List[Dict[int, str]],
        env_name: str,
        rho: float = 0.5,
        noise_min: float = -3.0,
        noise_max: float = 3.0,
        noise_seed: int = 42,
        env_kwargs: Optional[dict] = None,
        eval_env_kwargs: Optional[dict] = None,
        cache_save_file: str = None,
        true_samples_cache_file: str = None,
    ):
        """
        Initialize RLAgentGraph with correlated uniform noise.

        Args:
            spec_graph: Task graph specification
            env_name: Gymnasium environment name
            rho: Correlation coefficient (0 ≤ ρ ≤ 1).
                 Controls mix of shared vs individual noise.
                 ρ=1: fully shared, ρ=0: fully individual
            noise_min: Minimum noise value (lower bound of range)
            noise_max: Maximum noise value (upper bound of range)
            noise_seed: Random seed for noise generation (for reproducibility)
            env_kwargs: Keyword arguments for environment creation
            eval_env_kwargs: Keyword arguments for evaluation environment
            cache_save_file: Path to cache file for noisy samples (optional, usually not needed)
            true_samples_cache_file: Path to cache file for true samples WITHOUT noise.
                                    This should be shared across all noise configurations
                                    to avoid redundant environment sampling.
        """
        # Use true_samples_cache_file for the parent class caching
        # This way all noise configurations share the same true sample cache
        super().__init__(
            spec_graph=spec_graph,
            env_name=env_name,
            env_kwargs=env_kwargs,
            eval_env_kwargs=eval_env_kwargs,
            cache_save_file=true_samples_cache_file,  # Use separate cache for true samples
        )

        if not 0 <= rho <= 1:
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        if noise_min >= noise_max:
            raise ValueError(f"noise_min must be < noise_max, got [{noise_min}, {noise_max}]")

        self.rho = rho
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.noise_seed = noise_seed
        self.rng = np.random.RandomState(noise_seed)

        # Optional separate cache for noisy samples (usually not needed since noise is deterministic)
        self.noisy_samples_cache: Dict[str, Tuple[list, List[float]]] = {}
        self.noisy_cache_save_file = cache_save_file
        if cache_save_file:
            self._load_noisy_cache(cache_save_file)

        print(f"Initialized RLAgentGraphCorrelatedNoise with:")
        print(f"  Uniform({noise_min}, {noise_max})")
        print(f"  ρ={rho} (correlation coefficient)")
        print(f"  Noise = {rho}*shared + {1-rho}*individual")
        print(f"True samples cache: {true_samples_cache_file}")
        print(f"Noisy samples cache: {cache_save_file if cache_save_file else 'disabled (noise regenerated on-the-fly)'}")

    def _generate_shared_noise(self, n_samples: int) -> np.ndarray:
        """
        Generate shared uniform noise component. Same across all edges for each sample.

        Uses a dedicated RNG seeded consistently so that the shared noise is truly
        the same for all edges (i.e., sample #0 gets the same shared noise on all edges).

        Args:
            n_samples: Number of samples

        Returns:
            Array of shape (n_samples,) with uniform noise in [noise_min, noise_max]
        """
        # Use dedicated RNG for shared noise, seeded consistently
        shared_rng = np.random.RandomState(self.noise_seed)
        return shared_rng.uniform(self.noise_min, self.noise_max, size=n_samples)

    def _generate_edge_noise(
        self,
        path: List[int],
        target_vertex: int,
        n_samples: int
    ) -> np.ndarray:
        """
        Generate correlated uniform noise for an edge.

        Combines shared noise (correlated across edges) and individual noise
        (uncorrelated) according to: noise = rho * shared + (1-rho) * individual

        Args:
            path: The path up to (but not including) the target vertex
            target_vertex: The target vertex
            n_samples: Number of samples

        Returns:
            Array of shape (n_samples,) with uniform noise in [noise_min, noise_max]
        """
        # Generate shared noise component (same for all edges in path)
        shared_noise = self._generate_shared_noise(n_samples)

        # Generate individual noise component (unique to this edge)
        # Use edge-specific seed for reproducibility
        edge_seed = self.noise_seed + hash((tuple(path), target_vertex)) % (2**31)
        edge_rng = np.random.RandomState(edge_seed)
        individual_noise = edge_rng.uniform(self.noise_min, self.noise_max, size=n_samples)

        # Combine: rho * shared + (1-rho) * individual
        # This is a convex combination, so result stays in [noise_min, noise_max]
        total_noise = self.rho * shared_noise + (1 - self.rho) * individual_noise

        return total_noise

    def _load_noisy_cache(self, file: str):
        """Load noisy samples cache from file."""
        try:
            import pickle
            with open(file, "rb") as f:
                self.noisy_samples_cache = pickle.load(f)
            print(f"Loaded noisy samples cache with {len(self.noisy_samples_cache)} entries")
        except Exception as e:
            print(f"Could not load noisy cache from {file}: {e}")
            self.noisy_samples_cache = {}

    def _save_noisy_cache(self):
        """Save noisy samples cache to file."""
        if not self.noisy_cache_save_file:
            return
        try:
            import pickle
            with open(self.noisy_cache_save_file, "wb") as f:
                pickle.dump(self.noisy_samples_cache, f)
        except Exception as e:
            print(f"Could not save noisy cache: {e}")

    def _get_noisy_cache_key(self, path: List[int], target_vertex: int, n_samples: int) -> str:
        """
        Generate cache key that includes noise parameters.

        This ensures different noise configurations don't collide in the cache.
        """
        return (f"{list(path)}_{target_vertex}_{n_samples}_"
                f"rho{self.rho}_min{self.noise_min}_max{self.noise_max}_seed{self.noise_seed}")

    def sample_cached(self, target_vertex, n_samples, path, path_samples):
        """
        Sample with caching using two-level strategy.

        This method ensures proper cache separation:
        1. True samples (without noise) are cached in parent's cache (shared across noise configs)
        2. Noise is added on-the-fly (deterministic, so doesn't need caching)
        3. Optionally, noisy samples can be cached separately per noise configuration

        Args:
            target_vertex: Target vertex to reach
            n_samples: Number of samples to draw
            path: Path taken so far
            path_samples: Samples (states) from previous edge

        Returns:
            Tuple of (next_path_samples, noisy_losses)
        """
        # Check noisy sample cache first if enabled
        if self.noisy_cache_save_file:
            cache_key = self._get_noisy_cache_key(path, target_vertex, n_samples)
            if cache_key in self.noisy_samples_cache:
                return self.noisy_samples_cache[cache_key]

        # Get TRUE samples from parent's cache (this is the expensive operation)
        # CRITICAL: We call super().sample_cached() which uses the parent's cache
        # This cache contains only true samples (no noise), shared across all noise configs
        next_path_samples, true_losses = super().sample_cached(
            target_vertex, n_samples, path, path_samples
        )

        # Generate correlated noise for this edge (cheap, deterministic)
        noise = self._generate_edge_noise(path, target_vertex, n_samples)

        # Add noise to true losses
        noisy_losses = [
            true_loss + noise_val
            for true_loss, noise_val in zip(true_losses, noise)
        ]

        # Optionally cache the noisy results separately
        if self.noisy_cache_save_file:
            cache_key = self._get_noisy_cache_key(path, target_vertex, n_samples)
            self.noisy_samples_cache[cache_key] = (next_path_samples, noisy_losses)
            self._save_noisy_cache()

        return next_path_samples, noisy_losses

    def sample_full_path_cached(self, path: List[int], n_samples: int):
        """
        Sample full path with caching using two-level strategy.

        Similar to sample_cached, this ensures:
        1. True path samples are cached in parent's cache (shared across noise configs)
        2. Noise is added on-the-fly per noise configuration with proper correlation

        Args:
            path: Complete path to sample along
            n_samples: Number of trajectory samples

        Returns:
            List of trajectory samples, where each trajectory is a list of losses
        """
        # Get TRUE path samples from parent's cache
        true_trajectories = super().sample_full_path_cached(path, n_samples)

        # Generate noise for all edges in the path for all samples at once
        # This maintains the correlation structure correctly
        # noise_per_edge[edge_idx][sample_idx] = noise for that edge and sample
        noise_per_edge = []
        for edge_idx in range(len(path) - 1):
            # Generate noise for all n_samples for this edge
            noise = self._generate_edge_noise(
                path[:edge_idx + 1],  # Path up to (but not including) target
                path[edge_idx + 1],   # Target vertex
                n_samples             # All samples at once
            )
            noise_per_edge.append(noise)

        # Add noise to each trajectory
        noisy_trajectories = []
        for sample_idx in range(n_samples):
            true_trajectory = true_trajectories[sample_idx]
            noisy_trajectory = [
                true_trajectory[edge_idx] + noise_per_edge[edge_idx][sample_idx]
                for edge_idx in range(len(true_trajectory))
            ]
            noisy_trajectories.append(noisy_trajectory)

        return noisy_trajectories

    def get_noise_info(self) -> Dict:
        """
        Get information about noise configuration.

        Returns:
            Dictionary with noise parameters
        """
        return {
            "rho": self.rho,
            "noise_min": self.noise_min,
            "noise_max": self.noise_max,
            "noise_seed": self.noise_seed,
            "noise_type": "uniform_correlated",
            "description": f"noise = {self.rho}*shared + {1-self.rho}*individual",
        }
