"""
Experiment script for testing bucketed_var with GaussianNoiseAgentGraph.

Tests performance of the bucketed variance estimation algorithm under
controlled synthetic conditions with varying:
- Correlation levels (rho)
- Path lengths (number of edges)

Fixed parameters:
- Sample size: 5000
- Number of buckets: 50
"""
import os
import sys
from pathlib import Path

# Add parent directory to Python path for agents module
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
from agents.gaussian_noise_agent_graph import GaussianNoiseAgentGraph
from agents.bucketed_var import bucketed_var
from agents.baseline_var_estim import baseline_var_estim
from agents.calculate_coverage import calculate_coverage


def correlation_path_length_experiment():
    """
    Test how correlation and path length affect bucketed_var performance.

    This is the main experiment that varies both:
    - Correlation coefficient (rho)
    - Path length (graph_length)

    For each combination, evaluates:
    - Risk bound estimates
    - Coverage guarantees
    - Comparison with baseline
    """
    print("=" * 80)
    print("CORRELATION & PATH LENGTH EXPERIMENT")
    print("=" * 80)

    # Fixed parameters
    mean = 0.0
    std = 2.0
    n_samples = 5000
    n_samples_coverage = 10000
    e = 0.1
    total_buckets = 50

    # Varying parameters
    rho_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    graph_lengths = [3, 5, 10, 15, 20, 30]

    data = {
        "metadata": {
            "experiment": "correlation_path_length",
            "mean": mean,
            "std": std,
            "e": e,
            "total_buckets": total_buckets,
            "n_samples": n_samples,
            "n_samples_coverage": n_samples_coverage,
            "rho_values": rho_values,
            "graph_lengths": graph_lengths,
        },
        "results": {}
    }

    for graph_length in graph_lengths:
        print(f"\n{'='*60}")
        print(f"Graph length: {graph_length} ({graph_length - 1} edges)")
        print(f"{'='*60}")

        length_data = {}

        for rho in rho_values:
            print(f"\n  ρ = {rho}")
            print("  " + "-" * 40)

            # Create graph
            graph = GaussianNoiseAgentGraph(
                graph_length=graph_length,
                rho=rho,
                mean=mean,
                std=std,
                noise_seed=42,
            )

            # Run bucketed variance estimation
            vbs = bucketed_var(graph, e, total_buckets, n_samples)

            # Get the final vertex bucket (graph_length - 1 is the final vertex)
            final_vertex = graph_length - 1
            vb = vbs.buckets[(final_vertex, total_buckets)]

            # Extract results
            selected_path = vb.path
            budgets = vb.path_score_quantiles
            risk_bound = max(vb.path_score_quantiles)

            # Calculate coverage
            coverage = calculate_coverage(
                graph,
                selected_path,
                budgets,
                n_samples_coverage,
            )

            # Also compute baseline for comparison
            baseline_path, baseline_scores = baseline_var_estim(graph, e, n_samples)
            baseline_coverage = calculate_coverage(
                graph,
                baseline_path,
                [max(baseline_scores)] * (graph_length - 1),
                n_samples_coverage,
            )

            length_data[str(rho)] = {
                "risk_bound": risk_bound,
                "budgets": budgets,
                "coverage": coverage,
                "baseline_risk_bound": max(baseline_scores),
                "baseline_coverage": baseline_coverage,
                "improvement": max(baseline_scores) - risk_bound,
            }

            print(f"    Bucketed risk bound: {risk_bound:.4f}")
            print(f"    Baseline risk bound: {max(baseline_scores):.4f}")
            print(f"    Improvement: {max(baseline_scores) - risk_bound:.4f}")
            print(f"    Coverage: {coverage:.4f}")

        data["results"][str(graph_length)] = length_data

        # Save results
        output_file = "experiments_data/gaussian-correlation-path-length.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print("\n" + "=" * 80)
        print(f"Results saved to {output_file}")
        print("=" * 80)


if __name__ == "__main__":
    correlation_path_length_experiment()
