from heapq import heappop, heappush
from typing import Dict, List, Tuple
import pickle


class NonConformityScoreGraph:
    """
    Base class representing the non-conformity score graph.
    Allows sampling from distributions and evaluating the
    non-conformity scores.
    """

    def __init__(self, adj_lists: List[List[int]], cache_save_file: str=None) -> None:
        # adjacency lists of the DAG
        remove_loops(adj_lists)
        self.adj_lists = adj_lists
        self.rev_adj_lists = reverse_adj_list(self.adj_lists)
        self.dag_layers = dag_layers(self.adj_lists, self.rev_adj_lists)
        self.sample_cache: Dict[str, Tuple[list, List[float]]] = dict()
        self.samples_full_path_cache: Dict[str, List[List[float]]] = dict()

        self.n_vertices = len(adj_lists)
        self.n_paths = self.compute_n_paths()

        self.load_caches(cache_save_file)
        self.cache_save_file = cache_save_file

    def sample(
        self,
        target_vertex: int,
        n_samples: int,
        path: List[int],
        path_samples: list,
    ) -> Tuple[list, List[float]]:
        """
        Samples from the distribution induced on an edge by a path, i.e.,
        extends the path samples until the target_vertex.

        Also evaluates the non-conformity score on the samples.
        """
        raise NotImplementedError
    
    def sample_cached(
        self,
        target_vertex: int,
        n_samples: int,
        path: List[int],
        path_samples: list,
    ) -> Tuple[list, List[float]]:
        cache_string = str(list(path)) + str(target_vertex) + str(n_samples)
        if cache_string in self.sample_cache:
            return self.sample_cache[cache_string]
        
        # retrieve from cache if we have already drawn enough samples before
        cache_start_string = str(list(path)) + str(target_vertex)
        for key in self.sample_cache.keys():
            if key.startswith(cache_start_string) and len(self.sample_cache[key][0]) >= n_samples:
                return self.sample_cache[key][0][:n_samples], self.sample_cache[key][1][:n_samples]
            
        sample = self.sample(target_vertex, n_samples, path, path_samples)
        self.sample_cache[cache_string] = sample
        self.save_caches(self.cache_save_file)
        return sample
    
    def sample_full_path(
        self,
        path: List[int],
        n_samples: int,
    ) -> List[List[float]]:
        """
        Samples n_samples trajectories of non-conformity scores along specified path
        """
        trajectories_scores = [[] for _ in range(n_samples)]
        prev_samples = [None for _ in range(n_samples)]

        for i in range(1, len(path)):
            # starting from 1 because we sample on edges of the path
            prev_samples, scores = self.sample(path[i], n_samples, path[:i], prev_samples)
            for j in range(n_samples):
                trajectories_scores[j].append(scores[j])

        return trajectories_scores
    
    def sample_full_path_cached(
        self,
        path: List[int],
        n_samples: int,
    ) -> List[List[float]]:
        cache_string = str(list(path)) + str(n_samples)
        if cache_string in self.samples_full_path_cache:
            return self.samples_full_path_cache[cache_string]
        
        cache_start_string = str(list(path))
        for key in self.samples_full_path_cache.keys():
            if key.startswith(cache_start_string) and len(self.samples_full_path_cache[key]) >= n_samples:
                return self.samples_full_path_cache[key]
            
        sample = self.sample_full_path(path, n_samples)
        self.samples_full_path_cache[cache_string] = sample
        self.save_caches(self.cache_save_file)
        return sample
    
    def compute_n_paths(self) -> int:
        stack = [(0,)]
        paths = 0

        while stack:
            path = stack.pop()
            if len(self.adj_lists[path[-1]]) == 0:
                paths += 1
            
            for succ in self.adj_lists[path[-1]]:
                next_path = path + (succ,)
                stack.append(next_path)

        return paths
    
    def save_caches(self, file: str):
        if not file:
            return
        with open(file, "wb") as f:
            pickle.dump((self.sample_cache, self.samples_full_path_cache), f)

    def load_caches(self, file: str):
        if not file:
            return
        try:
            with open(file, "rb") as f:
                self.sample_cache, self.samples_full_path_cache = pickle.load(f)
        except:
            return
    

def remove_loops(adj_list: List[List[int]]) -> None:
    """
    Removes self loops on vertices from the adjacency lists
    """
    for i in range(len(adj_list)):
        if i in adj_list[i]:
            adj_list[i].remove(i)


def reverse_adj_list(adj_list: List[List[int]]) -> List[List[int]]:
    """
    Compute adjancency list of graph with reversed edges
    """
    reversed = [[] for _ in range(len(adj_list))]
    for i in range(len(adj_list)):
        for j in adj_list[i]:
            reversed[j].append(i)

    return reversed


def dag_layers(
    adj_list: List[List[int]], rev_adj_list: List[List[int]]
) -> List[List[int]]:
    """
    Partitions the vertices of a DAG into layers by the induced partial order
    """
    layers = [[0]]
    explored = set()
    queue = []
    heappush(queue, 0)

    while len(queue) > 0:
        v1 = heappop(queue)
        explored.add(v1)
        layer = []
        for v2 in adj_list[v1]:
            if v2 == v1:
                continue
            if all((pred in explored) for pred in rev_adj_list[v2]):
                layer.append(v2)
                heappush(queue, v2)
        if len(layer) > 0:
            layers.append(layer)

    return layers
