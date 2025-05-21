from typing import Dict, List, Tuple
import numpy as np

from conformal.nonconformity_score_graph import NonConformityScoreGraph
from conformal.utils import get_conformal_quantile_index, get_dkw_quantile_index


class VertexBucket:
    """
    Object to store the best path to reach a vertex for a specific bucket.

    path: List[int] (the best path as the sequence of vertices)
    path_buckets: List[int] (the buckets assigned to each of the edges along the path)
    path_score_quantiles: List[float] (the quantile scores corresponding to the buckets of each edge in path)
    path_samples: list (score samples at the current vertex and bucket)
    """

    def __init__(
        self,
        vertex: int,
        bucket: int,
        path: List[int] = None,
        path_buckets: List[int] = None,
        path_score_quantiles: List[float] = None,
        path_samples: list = None,
    ) -> None:
        self.vertex = vertex
        self.bucket = bucket
        self.path = path
        self.path_buckets = path_buckets
        self.path_score_quantiles = path_score_quantiles
        self.path_samples = path_samples


class VertexBuckets:
    """
    The DP table object that stores the vertex bucket objects for each vertex and bucket.
    The buckets dictionary is constructed by the bucketed_conformal_pred algorithm iteratively.
    """

    def __init__(
        self, n_vertices: int, e: float, total_buckets: int, n_samples: int
    ) -> None:
        self.n_vertices = n_vertices
        self.e = e
        self.total_buckets = total_buckets
        self.n_samples = n_samples
        self.buckets: Dict[Tuple[int, int], VertexBucket] = dict()


def bucketed_conformal_pred(
    score_graph: NonConformityScoreGraph, 
    e: float, total_buckets: int, 
    n_samples: int, 
    delta: float=0.95,
    quantile_eval: str="normal",
) -> VertexBuckets:
    """
    DP implementation of the bucketed conformal prediction algorithm.

    Params :
    score_graph : NonConformityScoreGraph
    e : float (the error/confidence parameter)
    total_buckets : int (total number of buckets for dividing the error parameter)
    n_samples : int (number of samples to draw along each edge to estimate the quantile)

    Returns :
    vbs : (VertexBuckets)
    """
    n_estimations = min(score_graph.n_vertices*score_graph.compute_n_paths(), score_graph.n_vertices*score_graph.n_vertices*total_buckets*total_buckets)
    delta_bar = delta/n_estimations
    vbs = VertexBuckets(len(score_graph.adj_lists), e, total_buckets, n_samples)
    for i in range(total_buckets + 1):
        vbs.buckets[(0, i)] = VertexBucket(
            0, i, [0], [], [], [None for _ in range(n_samples)]
        )
    for layer in score_graph.dag_layers[1:]:
        # skip first layer which is just [0]
        for v in layer:
            for bucket in range(total_buckets + 1):
                vb = VertexBucket(v, bucket)
                vbs.buckets[(v, bucket)] = vb
                min_quantile = np.inf
                for pred in score_graph.rev_adj_lists[v]:
                    bucket_preds = range(bucket + 1) if pred != 0 else [0]
                    # to get to vertex 0, we do not want to use any of the error param
                    for bucket_pred in bucket_preds:
                        pred_vb = vbs.buckets[(pred, bucket_pred)]
                        path_samples, scores = score_graph.sample_cached(
                            v,
                            n_samples,
                            pred_vb.path,
                            pred_vb.path_samples,
                        )
                        scores = sorted(scores)
                        rem_e = (bucket - bucket_pred) * (e / total_buckets)
                        if quantile_eval == "normal":
                            quantile_index = get_conformal_quantile_index(n_samples, rem_e)
                        else:
                            quantile_index = get_dkw_quantile_index(n_samples, rem_e, delta_bar)

                        quantile = scores[quantile_index]  # compute quantile
                        pred_vb_max_quantile = (
                            max(max(pred_vb.path_score_quantiles), quantile)
                            if len(pred_vb.path_score_quantiles) != 0
                            else quantile
                        )
                        if pred_vb_max_quantile <= min_quantile:
                            min_quantile = pred_vb_max_quantile
                            vb.path = [i for i in pred_vb.path] + [v]
                            vb.path_buckets = [i for i in pred_vb.path_buckets] + [
                                bucket - bucket_pred
                            ]
                            vb.path_score_quantiles = [
                                i for i in pred_vb.path_score_quantiles
                            ] + [quantile]
                            vb.path_samples = path_samples
    return vbs
