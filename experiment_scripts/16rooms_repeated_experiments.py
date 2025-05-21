from conformal.single_path_var_estim import single_path_var_estim
from conformal.all_paths_conformal_pred import all_paths_conformal_pred
from conformal.bucketed_conformal_pred import bucketed_conformal_pred
from conformal.calculate_coverage import calculate_coverage
from conformal.dirl_score_graphs import DIRLCumRewardScoreGraph, DIRLRepeatedScoreGraph


import json

import dill as pickle

with open("conformal_experiments_data/16rooms-dirl-policies/path_policies.pkl", "rb") as f:
    path_policies = pickle.load(f)

with open("conformal_experiments_data/16rooms-dirl-policies/adj_list.pkl", "rb") as f:
    adj_list = pickle.load(f)

with open("conformal_experiments_data/16rooms-dirl-policies/terminal_vertices.pkl", "rb") as f:
    terminal_vertices = pickle.load(f)

with open("conformal_experiments_data/16rooms-repeated/16rooms-spec13-cum-rew-scoregraph.pkl", "rb") as f:
    cum_reward_score_graph = pickle.load(f)

path = [0, 1, 3, 4, 6, 8, 9, 10, 12]
full_repeated_path = [v for v in range(5*(len(path) - 1))]
full_repeated_path.append(full_repeated_path[-1] + 1)
# cum_reward_score_graph = DIRLRepeatedScoreGraph(path_policies, path, 5)
n_samples = 10000
n_samples_coverage = 10000
e = 0.1
total_buckets = [25, 50, 75, 100, 125, 150, 175, 200]

data = dict()
for buckets in total_buckets:
    data[buckets] = dict()
    vbs = bucketed_conformal_pred(cum_reward_score_graph, e, buckets, n_samples)
    for i in range(1, 6):
        partial_path = [v for v in range(i*(len(path)-1))]
        partial_path.append(partial_path[-1] + 1)
        data[buckets][i] = dict()

        vb = vbs.buckets[(partial_path[-1], buckets)]
        data[buckets][i] = {
            "path_buckets": vb.path_buckets,
            "path_score_quantiles": vb.path_score_quantiles, 
            "max_path_score_quantile": max(vb.path_score_quantiles)
        }
        data[buckets][i]["bucketed-coverage"] = calculate_coverage(
            cum_reward_score_graph, full_repeated_path, vb.path_score_quantiles, n_samples_coverage
        )

data = {"data_bucketed": data}
data["data_baseline"] = dict()
for i in range(1, 6):
    partial_path = [v for v in range(i*(len(path)-1))]
    partial_path.append(partial_path[-1] + 1)
    var_estim = single_path_var_estim(cum_reward_score_graph, partial_path, e, n_samples)
    coverage = calculate_coverage(
        cum_reward_score_graph, full_repeated_path, 
        [var_estim for _ in range(len(partial_path) - 1)], n_samples_coverage
    )
    data["data_baseline"][i] = {
        "quantile": var_estim,
        "coverage": coverage
    }

# Convert the Python object to a JSON string
json_data = json.dumps(data, indent=2)

# Store the JSON string in a file
with open("conformal_experiments_data/16rooms-repeated-data-additional.json", "w") as json_file:
    json_file.write(json_data)

with open("conformal_experiments_data/16rooms-repeated/16rooms-spec13-cum-rew-scoregraph.pkl", "wb") as f:
    pickle.dump(cum_reward_score_graph, f)
