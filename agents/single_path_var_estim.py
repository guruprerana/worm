from typing import List
from conformal.nonconformity_score_graph import NonConformityScoreGraph
from conformal.utils import get_conformal_quantile_index, get_dkw_quantile_index


def single_path_var_estim(
    score_graph: NonConformityScoreGraph, 
    path: List[int],
    e: float, 
    n_samples: int, 
    delta: float=0.05,
    quantile_eval: str="normal",
) -> float:
    prev_samples = [None for _ in range(n_samples)]
    trajectories_scores = [[] for _ in range(n_samples)]

    for i in range(1, len(path)):
        # starting from 1 because we sample on edges of the path
        prev_samples, scores = score_graph.sample_cached(path[i], n_samples, path[:i], prev_samples)
        for j in range(n_samples):
            trajectories_scores[j].append(scores[j])

    trajectories_scores = sorted(max(scores) for scores in trajectories_scores)
    if quantile_eval == "normal":
        quantile_index = get_conformal_quantile_index(n_samples, e)
    else:
        quantile_index = get_dkw_quantile_index(n_samples, e, delta)

    return trajectories_scores[quantile_index]
