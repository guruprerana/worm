import os
import sys
from pathlib import Path

# Add parent directory to Python path for agents module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment for headless rendering
os.environ['PYGLET_HEADLESS'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'

import json
import argparse
from agents.baseline_var_estim import baseline_var_estim
from agents.bucketed_var import bucketed_var
from agents.calculate_coverage import calculate_coverage
import agents.miniworld
import gymnasium as gym
import numpy as np
from agents.miniworld.boxrelay import spec_graph, BoxRelay
from agents.rl_agent_graph import RLAgentGraph
from agents.rl_agent_graph_correlated_noise import RLAgentGraphCorrelatedNoise
from PIL import Image

wandb_project_name = "boxrelayenv-agentview"
env_kwargs = {"view": "agent"}
cache_save_file = "logs/boxrelayenv-agentview/sample_caches.pkl"
task_graph = RLAgentGraph(spec_graph, "BoxRelay-v0", env_kwargs=env_kwargs, eval_env_kwargs=env_kwargs, cache_save_file=cache_save_file)

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
        min_path, min_path_scores = baseline_var_estim(task_graph, e, n_samples, quantile_eval="conformal")
        all_paths_coverage = calculate_coverage(
            task_graph, 
            min_path, 
            [max(min_path_scores) for _ in range(len(min_path)-1)], 
            n_samples_coverage,
        )
        for buckets in total_buckets:
            bucket_data = dict()
            vbs = bucketed_var(task_graph, e, buckets, n_samples, quantile_eval="conformal")
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
    with open("experiments_data/boxrelay-time-taken.json", "w") as json_file:
        json_file.write(json_data)

def sample_size_buckets_experiment():
    task_graph.load_path_policies(subfolder=wandb_project_name)

    n_samples = [i*500 for i in range(1, 21)]
    n_samples_coverage = 10000
    e = 0.1
    total_buckets = 100

    min_path, min_path_scores = baseline_var_estim(task_graph, e, 10000)

    data = {}
    for n in n_samples:
        vbs = bucketed_var(task_graph, e, total_buckets, n)
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
    with open("experiments_data/boxrelay-sample-size.json", "w") as json_file:
        json_file.write(json_data)

    n_samples = 10000
    n_samples_coverage = 10000
    e = 0.1
    total_buckets = [5, 10, 20, 30, 40, 50, 60, 70, 100]

    min_path, min_path_scores = baseline_var_estim(task_graph, e, 10000)

    data = {}
    for buckets in total_buckets:
        vbs = bucketed_var(task_graph, e, buckets, n_samples)
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
    with open("experiments_data/boxrelay-buckets.json", "w") as json_file:
        json_file.write(json_data)


def correlated_noise_experiment():
    """
    Test bucketed variance estimation with correlated Beta-distributed noise.

    Evaluates how the algorithm performs under different correlation levels (rho)
    and noise magnitudes.

    Uses efficient caching strategy: all noise configurations share the same
    true samples cache, avoiding redundant environment sampling.
    """
    # Load trained policies once
    task_graph.load_path_policies(subfolder=wandb_project_name)

    # Experiment parameters
    n_samples = 5000
    n_samples_coverage = 5000
    e = 0.1
    total_buckets = 50

    # Test different correlation levels
    rho_values = [0.0, 0.3, 0.5, 0.7, 0.9]

    # Test different noise ranges
    noise_configs = [
        {"noise_min": 0.0, "noise_max": 10.0, "name": "max-10"},
        {"noise_min": 0.0, "noise_max": 50.0, "name": "max-50"},
        {"noise_min": 0.0, "noise_max": 100.0, "name": "max-100"},
    ]

    data = {
        "metadata": {
            "e": e,
            "total_buckets": total_buckets,
            "n_samples": n_samples,
            "n_samples_coverage": n_samples_coverage,
            "env": "boxrelay",
            "noise_distribution": "Beta(2, 2)",
        },
        "noise_experiments": {}
    }

    # SHARED cache file for true samples (without noise)
    # All noise configurations will reuse this cache, so environment sampling happens only once
    true_samples_cache_file = "logs/boxrelayenv-agentview/sample_caches_true_samples.pkl"

    print(f"\n{'='*60}")
    print(f"Using shared true samples cache: {true_samples_cache_file}")
    print(f"Environment sampling will be reused across all noise configurations")
    print(f"{'='*60}")

    # Test with different noise configurations
    for noise_config in noise_configs:
        noise_name = noise_config["name"]
        noise_min = noise_config["noise_min"]
        noise_max = noise_config["noise_max"]

        print(f"\n{'='*60}")
        print(f"Testing noise range: [{noise_min}, {noise_max}] ({noise_name})")
        print(f"{'='*60}")

        noise_range_data = {}

        for rho in rho_values:
            print(f"\nTesting ρ={rho}...")

            # Create task graph with correlated noise
            # Use two-level caching:
            # 1. true_samples_cache_file: SHARED across all noise configs (expensive environment sampling)
            # 2. cache_save_file: PER noise config (cheap but still worth caching for re-runs)
            noisy_samples_cache_file = f"logs/boxrelayenv-agentview/noisy_samples_{noise_name}_rho{rho}.pkl"

            task_graph_noise = RLAgentGraphCorrelatedNoise(
                spec_graph=spec_graph,
                env_name="BoxRelay-v0",
                rho=rho,
                noise_min=noise_min,
                noise_max=noise_max,
                noise_seed=42,
                env_kwargs=env_kwargs,
                eval_env_kwargs=env_kwargs,
                true_samples_cache_file=true_samples_cache_file,  # SHARED across all configs
                # cache_save_file=noisy_samples_cache_file,  # PER config (for re-runs)
            )

            # Load the same trained policies
            task_graph_noise.load_path_policies(subfolder=wandb_project_name)

            # Run bucketed variance estimation with noise
            vbs_noise = bucketed_var(task_graph_noise, e, total_buckets, n_samples, quantile_eval="normal")
            vb_noise = vbs_noise.buckets[(5, total_buckets)]

            # Calculate coverage using the noisy task graph
            coverage_noise = calculate_coverage(
                task_graph_noise,
                vb_noise.path,
                vb_noise.path_score_quantiles,
                n_samples_coverage,
            )

            noise_range_data[str(rho)] = {
                "path": vb_noise.path,
                "path_buckets": vb_noise.path_buckets,
                "path_score_quantiles": vb_noise.path_score_quantiles,
                "max_path_score_quantile": max(vb_noise.path_score_quantiles),
                "coverage": coverage_noise,
                "noise_config": task_graph_noise.get_noise_info(),
            }

            print(f"  Path: {vb_noise.path}")
            print(f"  Max quantile: {max(vb_noise.path_score_quantiles):.4f}")
            print(f"  Coverage: {coverage_noise:.4f}")

        data["noise_experiments"][noise_name] = noise_range_data

        # Save results to JSON
        json_data = json.dumps(data, indent=2)
        output_file = "experiments_data/boxrelay-correlated-noise.json"
        with open(output_file, "w") as json_file:
            json_file.write(json_data)

        print(f"\n{'='*60}")
        print(f"Results saved to {output_file}")
        print(f"{'='*60}")


def generate_screenshots():
    env = gym.make("BoxRelay-v0", render_mode="rgb_array", view="top", task_str=BoxRelay.Tasks.GOTO_LEFT_HALL_TARGET)
    env.reset()

    frame = env.render()
    img = Image.fromarray(frame)
    img.save("experiments_data/boxrelay/boxrelay_topview.png")

def parse_args():
    parser = argparse.ArgumentParser(description='Run BoxRelay benchmarks and experiments')
    parser.add_argument('function', type=str, choices=['train', 'risk_min', 'sample_size_buckets_experiment', 'correlated_noise_experiment', 'generate_screenshots'],
                        help='Function to run: train, risk_min, sample_size_buckets_experiment, correlated_noise_experiment, or generate_screenshots')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.function == 'train':
        train()
    elif args.function == 'risk_min':
        risk_min()
    elif args.function == 'sample_size_buckets_experiment':
        sample_size_buckets_experiment()
    elif args.function == 'correlated_noise_experiment':
        correlated_noise_experiment()
    elif args.function == 'generate_screenshots':
        generate_screenshots()
