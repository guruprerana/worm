from typing import List, Tuple
import numpy as np

from agents.agent_graph import AgentGraph
from agents.utils import get_quantile_index, get_dkw_quantile_index


def baseline_var_estim(
    score_graph: AgentGraph, 
    e: float, 
    n_samples: int, 
    delta: float=0.05,
    quantile_eval: str="normal",
) -> Tuple[Tuple[int], List[float]]:
    """
    Naive var estimation algorithm on the agent graph
    that first builds n_samples traces over all paths in the graph
    to reach the terminal vertex and then runs quantile estimation 
    on each path.

    Inputs:
        score_graph : AgentGraph
        e : float (non-coverage rate)
        n_samples : int (number of sample traces to estimate quantile from along each path)
        delta : float (confidence-level)

    Outputs:
        min_path : Tuple[int] (the path that achieves the minimum bound on the losses)
        min_path_scores : List[float] (the ~(1-e)th quantile of the minimum scores of the samples along this path)
    """
    path_samples: dict[Tuple[int], list] = dict()
    path_scores: dict[Tuple[int], List[List[float]]] = dict()
    path_samples[(0,)] = [None for _ in range(n_samples)]
    path_scores[(0,)] = []

    delta_bar = delta/score_graph.n_paths

    stack: List[Tuple[int]] = [(0,)]

    while stack:
        path = stack.pop()
        if len(score_graph.adj_lists[path[-1]]) == 0:
            continue
        for succ in score_graph.adj_lists[path[-1]]:
            if succ == path[-1]:
                continue
            samples, scores = score_graph.sample_cached(
                succ, n_samples, path, path_samples[path]
            )
            next_path = path + (succ,)
            path_samples[next_path] = samples
            path_scores[next_path] = [scores for scores in path_scores[path]] + [scores]
            stack.append(next_path)

        del path_samples[path], path_scores[path]

    min_path = None
    min_path_quantile = np.inf
    min_path_scores = None
    for path in path_scores:
        score_maxes: List[Tuple[float, List[float]]] = list()
        for i in range(n_samples):
            sample_path_scores: List[float] = []
            for j in range(len(path)-1):
                sample_path_scores.append(path_scores[path][j][i])
            score_maxes.append((max(sample_path_scores), sample_path_scores))

        score_maxes = sorted(score_maxes, key=lambda t: t[0])
        if quantile_eval == "normal":
            quantile_index = get_quantile_index(n_samples, e)
        else:
            quantile_index = get_dkw_quantile_index(n_samples, e, delta_bar)
        max_score, scores = score_maxes[quantile_index]

        if max_score <= min_path_quantile:
            min_path = path
            min_path_quantile = max_score
            min_path_scores = scores

    return min_path, min_path_scores


def baseline_cvar_estim(
    score_graph: AgentGraph,
    e: float,
    n_samples: int,
    delta: float=0.05,
    quantile_eval: str="normal",
) -> Tuple[Tuple[int], List[float], float]:
    """
    Naive CVaR (Conditional Value at Risk) estimation algorithm on the agent graph
    that first builds n_samples traces over all paths in the graph to reach the
    terminal vertex and then runs CVaR estimation on each path.

    CVaR is the expected value of losses in the tail beyond the VaR threshold.
    For each path:
    1. Compute VaR as the (1-e)th quantile of maximum losses along the path
    2. Compute CVaR as the mean of all losses >= VaR

    Inputs:
        score_graph : AgentGraph
        e : float (non-coverage rate, e.g., 0.1 for 90% confidence)
        n_samples : int (number of sample traces to estimate quantile from along each path)
        delta : float (confidence-level for conformal prediction)
        quantile_eval : str ("normal" or "conformal")

    Outputs:
        min_path : Tuple[int] (the path that achieves the minimum CVaR)
        min_path_scores : List[float] (the scores along this path at the VaR threshold)
        min_cvar : float (the CVaR value for the minimum path)
    """
    path_samples: dict[Tuple[int], list] = dict()
    path_scores: dict[Tuple[int], List[List[float]]] = dict()
    path_samples[(0,)] = [None for _ in range(n_samples)]
    path_scores[(0,)] = []

    delta_bar = delta/score_graph.n_paths

    stack: List[Tuple[int]] = [(0,)]

    # Sample all paths
    while stack:
        path = stack.pop()
        if len(score_graph.adj_lists[path[-1]]) == 0:
            continue
        for succ in score_graph.adj_lists[path[-1]]:
            if succ == path[-1]:
                continue
            samples, scores = score_graph.sample_cached(
                succ, n_samples, path, path_samples[path]
            )
            next_path = path + (succ,)
            path_samples[next_path] = samples
            path_scores[next_path] = [scores for scores in path_scores[path]] + [scores]
            stack.append(next_path)

        del path_samples[path], path_scores[path]

    # Find path with minimum CVaR
    min_path = None
    min_cvar = np.inf
    min_path_scores = None

    for path in path_scores:
        # Compute max score for each sample along this path
        score_maxes: List[Tuple[float, List[float]]] = list()
        for i in range(n_samples):
            sample_path_scores: List[float] = []
            for j in range(len(path)-1):
                sample_path_scores.append(path_scores[path][j][i])
            score_maxes.append((max(sample_path_scores), sample_path_scores))

        # Sort by maximum score
        score_maxes = sorted(score_maxes, key=lambda t: t[0])

        # Compute VaR (quantile threshold)
        if quantile_eval == "normal":
            quantile_index = get_quantile_index(n_samples, e)
        else:
            quantile_index = get_dkw_quantile_index(n_samples, e, delta_bar)

        var_threshold = score_maxes[quantile_index][0]
        var_scores = score_maxes[quantile_index][1]

        # Compute CVaR as mean of all samples >= VaR
        # Include all samples from quantile_index onwards (the tail)
        tail_scores = [score_maxes[i][0] for i in range(quantile_index, n_samples)]

        if len(tail_scores) > 0:
            path_cvar = np.mean(tail_scores)
        else:
            # Edge case: if tail is empty, use VaR as CVaR
            path_cvar = var_threshold

        # Track path with minimum CVaR
        if path_cvar <= min_cvar:
            min_path = path
            min_cvar = path_cvar
            min_path_scores = var_scores

    return min_path, min_path_scores, min_cvar
