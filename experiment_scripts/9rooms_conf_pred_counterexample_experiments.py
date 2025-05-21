num_iters = 4000
env_num = 5
spec_num = 0
use_gpu = True

from agents.baseline_var_estim import baseline_var_estim
from agents.bucketed_var import bucketed_var
from agents.dirl_agent_graphs import DIRLCumRewardAgentGraph



from agents.baseline_var_estim import baseline_var_estim
from agents.bucketed_var import bucketed_var
from agents.calculate_coverage import calculate_coverage
from agents.dirl_agent_graphs import DIRLCumRewardAgentGraph

import dill as pickle
import json

with open("experiments_data/9rooms-counterexample-policies/path_policies.pkl", "rb") as f:
    path_policies = pickle.load(f)

with open("experiments_data/9rooms-counterexample-policies/adj_list.pkl", "rb") as f:
    adj_list = pickle.load(f)

with open("experiments_data/9rooms-counterexample-policies/terminal_vertices.pkl", "rb") as f:
    terminal_vertices = pickle.load(f)

cum_reward_score_graph = DIRLCumRewardAgentGraph(adj_list, path_policies, cum_reward_type="cum_safety_reach_reward")
n_samples = 10000
n_samples_coverage = 10000
es = [0.2, 0.1, 0.05]
total_buckets = [5, 10, 15, 20]

data_cum_reward = dict()
data_cum_reward["metadata"] = {"es": es, "total_buckets": total_buckets, "scores": "cum-reward", "env": "9-rooms", "spec": spec_num, "n_samples": n_samples}

for e in es:
    e_data = dict()
    for buckets in total_buckets:
        bucket_data = dict()
        vbs = bucketed_var(cum_reward_score_graph, e, buckets, n_samples)
        min_path, min_path_scores = baseline_var_estim(cum_reward_score_graph, e, n_samples)
        vb = vbs.buckets[(terminal_vertices[0], buckets)]

        bucket_data["bucketed"] = {"path": vb.path, 
                                   "path_buckets": vb.path_buckets, 
                                   "path_score_quantiles": vb.path_score_quantiles, 
                                   "max_path_score_quantile": max(vb.path_score_quantiles)}
        bucket_data["all-paths"] = {"path": min_path, "min_path_scores": min_path_scores, "max_min_path_scores": max(min_path_scores)}

        bucket_data["bucketed-coverage"] = calculate_coverage(
            cum_reward_score_graph, vb.path, vb.path_score_quantiles, n_samples_coverage
        )
        bucket_data["all-paths-coverage"] = calculate_coverage(
            cum_reward_score_graph, 
            min_path, 
            [max(min_path_scores) for _ in range(len(min_path)-1)], 
            n_samples_coverage,
        )
        e_data[buckets] = bucket_data
    data_cum_reward[str(e)] = e_data

# Convert the Python object to a JSON string
json_data = json.dumps(data_cum_reward, indent=2)

# Store the JSON string in a file
with open("experiments_data/9rooms-counterexample-cum-reward.json", "w") as json_file:
    json_file.write(json_data)

with open("experiments_data/9rooms-counterexample-policies/cum-rew-score-graph.pkl", "wb") as f:
    pickle.dump(cum_reward_score_graph, f)
