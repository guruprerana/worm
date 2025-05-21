spec_num = 13

from conformal.all_paths_conformal_pred import all_paths_conformal_pred
from conformal.bucketed_conformal_pred import bucketed_conformal_pred
from conformal.calculate_coverage import calculate_coverage
from conformal.dirl_score_graphs import DIRLCumRewardScoreGraph


import json

import dill as pickle

with open("conformal_experiments_data/16rooms-dirl-policies/path_policies.pkl", "rb") as f:
    path_policies = pickle.load(f)

with open("conformal_experiments_data/16rooms-dirl-policies/adj_list.pkl", "rb") as f:
    adj_list = pickle.load(f)

with open("conformal_experiments_data/16rooms-dirl-policies/terminal_vertices.pkl", "rb") as f:
    terminal_vertices = pickle.load(f)

with open("conformal_experiments_data/16rooms-dirl-policies/16rooms-spec13-cum-rew-scoregraph.pkl", "rb") as f:
    cum_reward_score_graph = pickle.load(f)

# cum_reward_score_graph = DIRLCumRewardScoreGraph(adj_list, path_policies)
n_samples = 10000
n_samples_coverage = 10000
es = [0.2, 0.1, 0.05]
total_buckets = [10, 20, 40, 50, 100]

data_cum_reward = dict()
data_cum_reward["metadata"] = {"es": es, "total_buckets": total_buckets, "scores": "cum-reward", "env": "16-rooms-3", "spec": spec_num, "n_samples": n_samples}

for e in es:
    e_data = dict()
    min_path, min_path_scores = all_paths_conformal_pred(cum_reward_score_graph, e, n_samples, quantile_eval="conformal")
    all_paths_coverage = calculate_coverage(
        cum_reward_score_graph, 
        min_path, 
        [max(min_path_scores) for _ in range(len(min_path)-1)], 
        n_samples_coverage,
    )
    for buckets in total_buckets:
        bucket_data = dict()
        vbs = bucketed_conformal_pred(cum_reward_score_graph, e, buckets, n_samples, quantile_eval="conformal")
        vb = vbs.buckets[(terminal_vertices[0], buckets)]

        bucket_data["bucketed"] = {"path": vb.path, 
                                   "path_buckets": vb.path_buckets, 
                                   "path_score_quantiles": vb.path_score_quantiles, 
                                   "max_path_score_quantile": max(vb.path_score_quantiles)}
        bucket_data["all-paths"] = {"path": min_path, "min_path_scores": min_path_scores, "max_min_path_scores": max(min_path_scores)}

        bucket_data["bucketed-coverage"] = calculate_coverage(
            cum_reward_score_graph, vb.path, vb.path_score_quantiles, n_samples_coverage
        )
        bucket_data["all-paths-coverage"] = all_paths_coverage
        e_data[buckets] = bucket_data
    data_cum_reward[str(e)] = e_data

# Convert the Python object to a JSON string
json_data = json.dumps(data_cum_reward, indent=2)

# Store the JSON string in a file
with open("conformal_experiments_data/16rooms-spec13-dirl-cum-safety-reach-reward.json", "w") as json_file:
    json_file.write(json_data)

with open("conformal_experiments_data/16rooms-dirl-policies/16rooms-spec13-cum-rew-scoregraph.pkl", "wb") as f:
    pickle.dump(cum_reward_score_graph, f)
