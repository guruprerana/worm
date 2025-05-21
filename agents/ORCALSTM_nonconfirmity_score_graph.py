import math
from typing import List
from conformal.nonconformity_score_graph import NonConformityScoreGraph


class ORCALSTMNonConformityScoreGraph(NonConformityScoreGraph):
    """
    Non-conformity score graph for the ORCA pedestrian simulator
    with LSTM network dataset
    """

    def __init__(self, adj_lists: List[List[int]], data: dict):
        super().__init__(adj_lists)
        self.data = data

    def sample(self, target_vertex, n_samples, path, path_samples):
        x = [self.data["x"][i][target_vertex-1] for i in range(n_samples)]
        y = [self.data["y"][i][target_vertex-1] for i in range(n_samples)]
        xh = [self.data["xh"][i][target_vertex-1] for i in range(n_samples)]
        yh = [self.data["yh"][i][target_vertex-1] for i in range(n_samples)]

        scores = [math.sqrt(((x[i] - xh[i])**2) + ((y[i] - yh[i])**2)) for i in range(n_samples)]
        return None, scores
