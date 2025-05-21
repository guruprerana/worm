from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from agents.nonconformity_score_graph import NonConformityScoreGraph
from dirl.spectrl.hierarchy.path_policies import PathPolicy
from dirl.spectrl.hierarchy.reachability import ReachabilityEnv
from dirl.spectrl.util.rl import get_rollout


class DIRLNonConformityScoreGraph(NonConformityScoreGraph):
    """
    Non-conformity score graphs for DIRL policies
    """

    def __init__(self, adj_lists: List[List[int]], path_policies: PathPolicy) -> None:
        super().__init__(adj_lists)
        self.path_policies = path_policies

    def sample(
        self,
        target_vertex: int,
        n_samples: int,
        path: List[int],
        path_samples: list,
    ) -> Tuple[list]:
        # assert target_vertex in self.adj_lists[path[-1]]
        assert len(path_samples) == n_samples

        pp = self.path_policies.get_vertex_path_policy(path)
        scores: List[float] = []
        next_path_samples = []
        print(f"Drawing samples for {path} -> {target_vertex}")
        for init_state in tqdm(path_samples):
            sarss = get_rollout(
                pp.reach_envs[target_vertex],
                pp.policies[target_vertex],
                False,
                init_state=init_state,
            )
            scores.append(self.compute_score(sarss, pp.reach_envs[target_vertex]))
            next_path_samples.append(pp.reach_envs[target_vertex].get_state())

        return path_samples, scores

    def compute_score(self, sarss: list, env: ReachabilityEnv) -> float:
        """
        Computes the non-conformity score on a given (s, a, r, s') trace.
        """
        raise NotImplementedError


class DIRLTimeTakenScoreGraph(DIRLNonConformityScoreGraph):
    """
    Non-conformity scores corresponding to time taken to complete reach objective.
    """

    def __init__(self, adj_lists: List[List[int]], path_policies: PathPolicy) -> None:
        super().__init__(adj_lists, path_policies)

    def compute_score(self, sarss: List, env: ReachabilityEnv) -> float:
        states = np.array([state for state, _, _, _ in sarss] + [sarss[-1][-1]])
        if env.cum_reward(states) <= 0:
            return np.inf
        return len(sarss)
    

class DIRLCumRewardScoreGraph(DIRLNonConformityScoreGraph):
    """
    Non-conformity scores corresponding to cumulative reward achieved.
    """

    def __init__(self, adj_lists: List[List[int]], path_policies: PathPolicy, cum_reward_type="normal") -> None:
        super().__init__(adj_lists, path_policies)
        self.cum_reward_type = cum_reward_type

    def compute_score(self, sarss: List, env: ReachabilityEnv) -> float:
        states = np.array([state for state, _, _, _ in sarss] + [sarss[-1][-1]])
        if self.cum_reward_type == "cum_safety_reward":
            return -env.cum_safety_reward(states)
        elif self.cum_reward_type == "cum_safety_reach_reward":
            return -env.cum_safety_reach_reward(states)
        return -env.cum_reward(states)
    

class DIRLRepeatedScoreGraph(DIRLCumRewardScoreGraph):
    def __init__(self, path_policies: PathPolicy, path: List[int], n_repeats: int, cum_reward_type="normal"):
        self.path = path
        self.n_repeats = n_repeats
        super().__init__(self.compute_repeated_adj_lists(), path_policies, cum_reward_type=cum_reward_type)

    def compute_repeated_adj_lists(self) -> List[List[int]]:
        return [[i+1] for i in range(self.n_repeats * (len(self.path)-1))] + [[]]
    
    def sample(self, target_vertex, n_samples, path, path_samples):
        print(f"Drawing samples for {path} -> {target_vertex}")

        target_vertex = target_vertex % (len(self.path)-1)
        if target_vertex == 1:
            path_samples = [None for _ in range(len(path_samples))]

        if target_vertex == 0:
            path = self.path[:-1]
            target_vertex = self.path[-1]
        else:
            path = self.path[:target_vertex]
            target_vertex = self.path[target_vertex]

        return super().sample(target_vertex, n_samples, path, path_samples)
