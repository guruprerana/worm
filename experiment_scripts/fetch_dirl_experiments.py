import os
import sys
from pathlib import Path

# Add parent directory to Python path for agents module
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, f"{str(Path(__file__).parent.parent)}/dirl")

num_iters = 4000
spec_num = 5
use_gpu = True

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from agents.baseline_var_estim import baseline_var_estim, baseline_cvar_estim
from agents.bucketed_var import bucketed_var
from agents.calculate_coverage import calculate_coverage
from agents.dirl_agent_graphs import DIRLCumRewardAgentGraph
import numpy as np
import json
import dill as pickle
from numpy import linalg as LA

with open("experiments_data/fetch-policies/path_policies.pkl", "rb") as f:
    path_policies = pickle.load(f)

with open("experiments_data/fetch-policies/adj_list.pkl", "rb") as f:
    adj_list = pickle.load(f)

with open("experiments_data/fetch-policies/terminal_vertices.pkl", "rb") as f:
    terminal_vertices = pickle.load(f)

# cum_reward_score_graph = DIRLCumRewardAgentGraph(adj_list, path_policies)
with open("experiments_data/fetch-policies/fetch-cum-rew-scoregraph.pkl", "rb") as f:
    cum_reward_score_graph = pickle.load(f)
n_samples = 10000
n_samples_coverage = 10000
es = [0.2, 0.1, 0.05]
total_buckets = [30]

data_cum_reward = dict()
data_cum_reward["metadata"] = {"es": es, "total_buckets": total_buckets, "scores": "cum-reward", "env": "fetch", "spec": spec_num, "n_samples": n_samples}

for e in es:
    e_data = dict()

    # VaR baseline
    min_path, min_path_scores = baseline_var_estim(cum_reward_score_graph, e, n_samples)
    all_paths_coverage = calculate_coverage(
        cum_reward_score_graph,
        min_path,
        [max(min_path_scores) for _ in range(len(min_path)-1)],
        n_samples_coverage,
    )

    # CVaR baseline
    cvar_path, _, cvar_value = baseline_cvar_estim(cum_reward_score_graph, e, n_samples)
    cvar_coverage = calculate_coverage(
        cum_reward_score_graph,
        cvar_path,
        [cvar_value for _ in range(len(cvar_path)-1)],
        n_samples_coverage,
    )

    for buckets in total_buckets:
        bucket_data = dict()
        vbs = bucketed_var(cum_reward_score_graph, e, buckets, n_samples)
        min_terminal_vertex_index = None
        min_terminal_vertex_score = np.inf
        for vertex in terminal_vertices:
            vb = vbs.buckets[(vertex, buckets)]
            if max(vb.path_score_quantiles) < min_terminal_vertex_score:
                min_terminal_vertex_score = max(vb.path_score_quantiles)
                min_terminal_vertex_index = vertex

        vb = vbs.buckets[(min_terminal_vertex_index, buckets)]

        # Compute bucketed CVaR: average of risk bounds across all bucket levels
        final_vertex = min_terminal_vertex_index
        bucketed_cvar = np.mean([
            max(vbs.buckets[(final_vertex, b)].path_score_quantiles)
            for b in range(1, buckets + 1)
        ])

        # Calculate CVaR error metrics
        cvar_absolute_error = cvar_value - bucketed_cvar
        cvar_relative_error = (cvar_value - bucketed_cvar) / cvar_value if cvar_value != 0 else 0.0

        bucket_data["bucketed"] = {"path": vb.path,
                                   "path_buckets": vb.path_buckets,
                                   "path_score_quantiles": vb.path_score_quantiles,
                                   "max_path_score_quantile": max(vb.path_score_quantiles),
                                   "bucketed_cvar": bucketed_cvar}
        bucket_data["all-paths"] = {"path": min_path, "min_path_scores": min_path_scores, "max_min_path_scores": max(min_path_scores)}
        bucket_data["cvar-baseline"] = {"path": cvar_path, "cvar_value": cvar_value}
        bucket_data["cvar-error"] = {"cvar_absolute_error": cvar_absolute_error, "cvar_relative_error": cvar_relative_error}

        bucket_data["bucketed-coverage"] = calculate_coverage(
            cum_reward_score_graph, min_path,
            vb.path_score_quantiles if min_path == tuple(vb.path) else [max(vb.path_score_quantiles) for _ in range(len(min_path) - 1)],
            n_samples_coverage
        )
        bucket_data["bucketed-cvar-coverage"] = calculate_coverage(
            cum_reward_score_graph, vb.path, [bucketed_cvar for _ in range(len(vb.path)-1)], n_samples_coverage
        )
        bucket_data["all-paths-coverage"] = all_paths_coverage
        bucket_data["cvar-coverage"] = cvar_coverage
        e_data[buckets] = bucket_data
    data_cum_reward[str(e)] = e_data

# Convert the Python object to a JSON string
json_data = json.dumps(data_cum_reward, indent=2)

# Store the JSON string in a file
with open("experiments_data/fetch-spec6-cum-reward.json", "w") as json_file:
    json_file.write(json_data)

with open("experiments_data/fetch-policies/fetch-cum-rew-scoregraph.pkl", "wb") as f:
    pickle.dump(cum_reward_score_graph, f)
