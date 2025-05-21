from conformal.all_paths_conformal_pred import all_paths_conformal_pred
from conformal.bucketed_conformal_pred import bucketed_conformal_pred
from conformal.calculate_coverage import calculate_coverage
from conformal.dirl_score_graphs import DIRLCumRewardScoreGraph
from numpy import linalg as LA
import numpy as np

import json

import dill as pickle

with open("conformal_experiments_data/fetch-policies/path_policies.pkl", "rb") as f:
    path_policies = pickle.load(f)

with open("conformal_experiments_data/fetch-policies/adj_list.pkl", "rb") as f:
    adj_list = pickle.load(f)

with open("conformal_experiments_data/fetch-policies/terminal_vertices.pkl", "rb") as f:
    terminal_vertices = pickle.load(f)

# cum_reward_score_graph = DIRLCumRewardScoreGraph(adj_list, path_policies)
with open("conformal_experiments_data/fetch-policies/fetch-cum-rew-scoregraph.pkl", "rb") as f:
    score_graph = pickle.load(f)

# cum_reward_score_graph = DIRLCumRewardScoreGraph(adj_list, path_policies)
n_samples = [i*500 for i in range(1, 21)]
n_samples_coverage = 10000
e = 0.1
total_buckets = 100

min_path, min_path_scores = all_paths_conformal_pred(score_graph, e, 10000)

data = {}
for n in n_samples:
    vbs = bucketed_conformal_pred(score_graph, e, total_buckets, n)
    min_terminal_vertex_index = None
    min_terminal_vertex_score = np.inf
    for vertex in terminal_vertices:
        vb = vbs.buckets[(vertex, total_buckets)]
        if max(vb.path_score_quantiles) < min_terminal_vertex_score:
            min_terminal_vertex_score = max(vb.path_score_quantiles)
            min_terminal_vertex_index = vertex

    vb = vbs.buckets[(min_terminal_vertex_index, total_buckets)]
    coverage = calculate_coverage(
        score_graph, 
        min_path, 
        vb.path_score_quantiles if min_path == tuple(vb.path) else [max(vb.path_score_quantiles) for _ in range(len(min_path) - 1)], 
        n_samples_coverage
    )
    data[n] = {
        "var-estim": max(vb.path_score_quantiles),
        "coverage": coverage,
    }

# Convert the Python object to a JSON string
json_data = json.dumps(data, indent=2)

# Store the JSON string in a file
with open("conformal_experiments_data/fetch-sample-size.json", "w") as json_file:
    json_file.write(json_data)

n_samples = 10000
n_samples_coverage = 10000
e = 0.1
total_buckets = [5, 10, 20, 30, 40, 50, 60, 70, 100]

min_path, min_path_scores = all_paths_conformal_pred(score_graph, e, 10000)

data = {}
for buckets in total_buckets:
    vbs = bucketed_conformal_pred(score_graph, e, buckets, n_samples)
    min_terminal_vertex_index = None
    min_terminal_vertex_score = np.inf
    for vertex in terminal_vertices:
        vb = vbs.buckets[(vertex, buckets)]
        if max(vb.path_score_quantiles) < min_terminal_vertex_score:
            min_terminal_vertex_score = max(vb.path_score_quantiles)
            min_terminal_vertex_index = vertex

    vb = vbs.buckets[(min_terminal_vertex_index, buckets)]
    coverage = calculate_coverage(
        score_graph, 
        min_path, 
        vb.path_score_quantiles if min_path == tuple(vb.path) else [max(vb.path_score_quantiles) for _ in range(len(min_path) - 1)], 
        n_samples_coverage
    )
    data[buckets] = {
        "var-estim": max(vb.path_score_quantiles),
        "coverage": coverage,
    }

# Convert the Python object to a JSON string
json_data = json.dumps(data, indent=2)

# Store the JSON string in a file
with open("conformal_experiments_data/fetch-buckets.json", "w") as json_file:
    json_file.write(json_data)
