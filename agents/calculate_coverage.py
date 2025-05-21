from typing import List
from conformal.nonconformity_score_graph import NonConformityScoreGraph


def calculate_coverage(
        score_graph: NonConformityScoreGraph, 
        path: List[int], 
        path_score_bounds: List[float],
        n_samples: int,
    ) -> float:
    """
    Calculates empirical coverage of conformal score bounds given by path_score_bounds
    on samples drawn along path from the score_graph
    """
    # assert len(path_score_bounds) == (len(path) - 1s)

    trajectories_scores = score_graph.sample_full_path_cached(path, n_samples)
    conforming_trajectories = 0
    
    for traj_scores in trajectories_scores:
        if all((traj_scores[i] <= path_score_bounds[i]) 
                for i in range(len(path_score_bounds))):
            conforming_trajectories += 1

    return (conforming_trajectories/n_samples)

