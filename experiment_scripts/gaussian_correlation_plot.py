import json
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import beta

# Define file paths
BASE_DATA_DIR = "experiments_data"
INPUT_FILE = "gaussian-correlation-path-length.json"
OUTPUT_IMAGE1 = f"{BASE_DATA_DIR}/plots/gaussian_correlation_coverage.png"
OUTPUT_IMAGE2 = f"{BASE_DATA_DIR}/plots/gaussian_correlation_coverage_error.png"

# Constants for CI calculation
N_SAMPLES_FOR_CI = 10000.0
ALPHA_CI = 0.05

# Construct full path
input_path = os.path.join(BASE_DATA_DIR, INPUT_FILE)

def calculate_ci_bounds(coverage_val, n_samples):
    """Calculate 95% Clopper-Pearson confidence intervals for coverage."""
    if not (0 <= coverage_val <= 1):
        print(f"Warning: Coverage value {coverage_val} is outside [0,1]. Clamping to [0,1] for CI.")
        coverage_val = min(max(coverage_val, 0.0), 1.0)

    k = round(coverage_val * n_samples)
    n_ci = n_samples

    if k < 0: k = 0
    if k > n_ci: k = n_ci

    lower_bound = 0.0
    if k > 0:
        lower_bound = beta.ppf(ALPHA_CI / 2, k, n_ci - k + 1)

    upper_bound = 1.0
    if k < n_ci:
        upper_bound = beta.ppf(1 - ALPHA_CI / 2, k + 1, n_ci - k)

    lower_bound_clamped = max(0.0, lower_bound)
    upper_bound_clamped = min(1.0, upper_bound)

    return lower_bound_clamped, upper_bound_clamped

def load_gaussian_data(json_filepath):
    """Load and process gaussian correlation path length data."""
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)

        metadata = data.get('metadata', {})
        results = data.get('results', {})

        # Get graph lengths from metadata or results
        graph_lengths = metadata.get('graph_lengths', [])
        if not graph_lengths:
            graph_lengths = sorted([int(k) for k in results.keys()])

        # Extract desired coverage from metadata
        desired_coverage = 1.0 - metadata.get('e', 0.1)
        n_samples_coverage = metadata.get('n_samples_coverage', N_SAMPLES_FOR_CI)

        # Process data for each graph length
        data_by_length = {}

        for graph_length in graph_lengths:
            length_key = str(graph_length)
            if length_key not in results:
                print(f"Warning: No data for graph length {graph_length}")
                continue

            length_data = results[length_key]

            # Lists to store data for this graph length
            rho_values = []
            coverages = []
            coverage_lowers = []
            coverage_uppers = []
            bucketed_cvar_coverages = []
            bucketed_cvar_coverage_lowers = []
            bucketed_cvar_coverage_uppers = []
            baseline_cvar_coverages = []
            baseline_cvar_coverage_lowers = []
            baseline_cvar_coverage_uppers = []
            coverage_errors = []

            # Sort by rho value
            sorted_rhos = sorted([(float(rho), rho) for rho in length_data.keys()])

            for rho_float, rho_key in sorted_rhos:
                rho_data = length_data[rho_key]

                # Get bucketed coverage (regular)
                coverage = rho_data.get('coverage')
                if coverage is not None:
                    lower, upper = calculate_ci_bounds(coverage, n_samples_coverage)
                    rho_values.append(rho_float)
                    coverages.append(coverage)
                    coverage_lowers.append(lower)
                    coverage_uppers.append(upper)

                    # Get bucketed CVaR coverage
                    bucketed_cvar_cov = rho_data.get('bucketed_cvar_coverage')
                    if bucketed_cvar_cov is not None:
                        lower_bc, upper_bc = calculate_ci_bounds(bucketed_cvar_cov, n_samples_coverage)
                        bucketed_cvar_coverages.append(bucketed_cvar_cov)
                        bucketed_cvar_coverage_lowers.append(lower_bc)
                        bucketed_cvar_coverage_uppers.append(upper_bc)
                    else:
                        bucketed_cvar_coverages.append(None)
                        bucketed_cvar_coverage_lowers.append(None)
                        bucketed_cvar_coverage_uppers.append(None)

                    # Get baseline CVaR coverage
                    baseline_cvar_cov = rho_data.get('baseline_cvar_coverage')
                    if baseline_cvar_cov is not None:
                        lower_bl, upper_bl = calculate_ci_bounds(baseline_cvar_cov, n_samples_coverage)
                        baseline_cvar_coverages.append(baseline_cvar_cov)
                        baseline_cvar_coverage_lowers.append(lower_bl)
                        baseline_cvar_coverage_uppers.append(upper_bl)

                        # Calculate coverage error
                        if bucketed_cvar_cov is not None:
                            coverage_errors.append(abs(bucketed_cvar_cov - baseline_cvar_cov))
                        else:
                            coverage_errors.append(None)
                    else:
                        baseline_cvar_coverages.append(None)
                        baseline_cvar_coverage_lowers.append(None)
                        baseline_cvar_coverage_uppers.append(None)
                        coverage_errors.append(None)

            data_by_length[graph_length] = {
                'rho_values': rho_values,
                'coverages': coverages,
                'coverage_lowers': coverage_lowers,
                'coverage_uppers': coverage_uppers,
                'bucketed_cvar_coverages': bucketed_cvar_coverages,
                'bucketed_cvar_coverage_lowers': bucketed_cvar_coverage_lowers,
                'bucketed_cvar_coverage_uppers': bucketed_cvar_coverage_uppers,
                'baseline_cvar_coverages': baseline_cvar_coverages,
                'baseline_cvar_coverage_lowers': baseline_cvar_coverage_lowers,
                'baseline_cvar_coverage_uppers': baseline_cvar_coverage_uppers,
                'coverage_errors': coverage_errors,
            }

        return data_by_length, desired_coverage, graph_lengths

    except FileNotFoundError:
        print(f"Error: File not found at {json_filepath}")
        return None, None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while processing {json_filepath}: {e}")
        return None, None, None

def plot_coverage(data_by_length, desired_coverage, graph_lengths, output_path):
    """Generate coverage vs correlation plot using bucketed var coverage."""
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 20,
        'legend.fontsize': 20,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot data for each graph length
    for graph_length in sorted(graph_lengths):
        if graph_length not in data_by_length:
            continue

        data = data_by_length[graph_length]
        rho_values = data['rho_values']
        # Use regular coverage (bucketed var coverage)
        coverages = data['coverages']
        lowers = data['coverage_lowers']
        uppers = data['coverage_uppers']

        # Filter out None values
        valid_data = [(r, c, l, u) for r, c, l, u in zip(rho_values, coverages, lowers, uppers) if c is not None]
        if not valid_data:
            print(f"Warning: No valid coverage data for graph length {graph_length}")
            continue

        rho_vals, cov_vals, lower_vals, upper_vals = zip(*valid_data)

        label = f"{graph_length - 1} edges"
        line, = ax.plot(rho_vals, cov_vals, linestyle='-', marker='o', label=label)
        ax.fill_between(rho_vals, lower_vals, upper_vals, alpha=0.2, color=line.get_color())

    # Add desired coverage line
    ax.axhline(y=desired_coverage, color="k", linestyle=':', label="Desired coverage")

    ax.set_xlabel("Correlation level (ρ)")
    ax.set_ylabel("Empirical quantile coverage")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300)
        print(f"Coverage plot saved to {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error saving coverage plot: {e}")

    plt.close()

def plot_coverage_error(data_by_length, graph_lengths, output_path):
    """Generate coverage error vs correlation plot."""
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 20,
        'legend.fontsize': 20,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot data for each graph length
    for graph_length in sorted(graph_lengths):
        if graph_length not in data_by_length:
            continue

        data = data_by_length[graph_length]
        rho_values = data['rho_values']
        coverage_errors = data['coverage_errors']

        # Filter out None values and convert to percentage
        valid_data = [(r, e * 100) for r, e in zip(rho_values, coverage_errors) if e is not None]
        if not valid_data:
            print(f"Warning: No valid coverage error data for graph length {graph_length}")
            continue

        rho_vals, error_vals = zip(*valid_data)

        label = f"{graph_length - 1} edges"
        ax.plot(rho_vals, error_vals, linestyle='-', marker='o', label=label)

    ax.set_xlabel("Correlation level (ρ)")
    ax.set_ylabel("Absolute coverage error (%)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300)
        print(f"Coverage error plot saved to {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error saving coverage error plot: {e}")

    plt.close()

def main():
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.join(BASE_DATA_DIR, "plots"), exist_ok=True)

    # Load data
    data_by_length, desired_coverage, graph_lengths = load_gaussian_data(input_path)

    if data_by_length is None or not data_by_length:
        print("Could not generate plots due to data loading errors or empty datasets.")
        return

    # Generate both plots
    plot_coverage(data_by_length, desired_coverage, graph_lengths, OUTPUT_IMAGE1)
    plot_coverage_error(data_by_length, graph_lengths, OUTPUT_IMAGE2)

if __name__ == '__main__':
    main()
