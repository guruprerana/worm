import json
from conformal.all_paths_conformal_pred import all_paths_conformal_pred
from conformal.bucketed_conformal_pred import bucketed_conformal_pred
from conformal.calculate_coverage import calculate_coverage
import conformal.miniworld
import gymnasium as gym
import numpy as np
from conformal.miniworld.boxrelay import spec_graph, BoxRelay
from conformal.rl_task_graph import RLTaskGraph
from PIL import Image

wandb_project_name = "boxrelayenv-agentview"
env_kwargs = {"view": "agent"}
cache_save_file = "logs/boxrelayenv-agentview/sample_caches.pkl"
task_graph = RLTaskGraph(spec_graph, "BoxRelay-v0", env_kwargs=env_kwargs, eval_env_kwargs=env_kwargs, cache_save_file=cache_save_file)

def train():
    # task_graph.train_all_edges(wandb_project_name, training_iters=500_000, final_policy_recordings=3, n_envs=1)
    task_graph.train_all_paths(wandb_project_name=wandb_project_name, n_samples=1000, training_iters=500_000, final_policy_recordings=3, n_envs=1)

def risk_min():
    task_graph.load_path_policies(subfolder=wandb_project_name)

    n_samples = 10000
    n_samples_coverage = 10000
    es = [0.2, 0.1, 0.05]
    total_buckets = [10, 20, 40, 50, 100]

    data = dict()
    data["metadata"] = {"es": es, "total_buckets": total_buckets, "scores": "cum-reward", "env": "boxrelay", "n_samples": n_samples}

    for e in es:
        e_data = dict()
        min_path, min_path_scores = all_paths_conformal_pred(task_graph, e, n_samples, quantile_eval="conformal")
        all_paths_coverage = calculate_coverage(
            task_graph, 
            min_path, 
            [max(min_path_scores) for _ in range(len(min_path)-1)], 
            n_samples_coverage,
        )
        for buckets in total_buckets:
            bucket_data = dict()
            vbs = bucketed_conformal_pred(task_graph, e, buckets, n_samples, quantile_eval="conformal")
            vb = vbs.buckets[(5, buckets)]

            bucket_data["bucketed"] = {"path": vb.path, 
                                    "path_buckets": vb.path_buckets, 
                                    "path_score_quantiles": vb.path_score_quantiles, 
                                    "max_path_score_quantile": max(vb.path_score_quantiles)}
            bucket_data["all-paths"] = {"path": min_path, "min_path_scores": min_path_scores, "max_min_path_scores": max(min_path_scores)}

            bucket_data["bucketed-coverage"] = calculate_coverage(
                task_graph, vb.path, vb.path_score_quantiles, n_samples_coverage
            )
            bucket_data["all-paths-coverage"] = all_paths_coverage
            e_data[buckets] = bucket_data
        data[str(e)] = e_data

    # Convert the Python object to a JSON string
    json_data = json.dumps(data, indent=2)

    # Store the JSON string in a file
    with open("conformal_experiments_data/boxrelay-time-taken.json", "w") as json_file:
        json_file.write(json_data)

def sample_size_buckets_experiment():
    task_graph.load_path_policies(subfolder=wandb_project_name)

    n_samples = [i*500 for i in range(1, 21)]
    n_samples_coverage = 10000
    e = 0.1
    total_buckets = 100

    min_path, min_path_scores = all_paths_conformal_pred(task_graph, e, 10000)

    data = {}
    for n in n_samples:
        vbs = bucketed_conformal_pred(task_graph, e, total_buckets, n)
        vb = vbs.buckets[(5, total_buckets)]
        coverage = calculate_coverage(
            task_graph, 
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
    with open("conformal_experiments_data/boxrelay-sample-size.json", "w") as json_file:
        json_file.write(json_data)

    n_samples = 10000
    n_samples_coverage = 10000
    e = 0.1
    total_buckets = [5, 10, 20, 30, 40, 50, 60, 70, 100]

    min_path, min_path_scores = all_paths_conformal_pred(task_graph, e, 10000)

    data = {}
    for buckets in total_buckets:
        vbs = bucketed_conformal_pred(task_graph, e, buckets, n_samples)
        vb = vbs.buckets[(5, buckets)]
        coverage = calculate_coverage(
            task_graph, 
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
    with open("conformal_experiments_data/boxrelay-buckets.json", "w") as json_file:
        json_file.write(json_data)


def generate_screenshots():
    env = gym.make("BoxRelay-v0", render_mode="rgb_array", view="top", task_str=BoxRelay.Tasks.GOTO_LEFT_HALL_TARGET)
    env.reset()

    frame = env.render()
    img = Image.fromarray(frame)
    img.save("conformal_experiments_data/boxrelay/boxrelay_topview.png")

if __name__ == "__main__":
    sample_size_buckets_experiment()
