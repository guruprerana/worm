import json
import matplotlib.pyplot as plt
import os
from scipy.stats import beta # Added import
import numpy as np # Added import for numpy

# Function to calculate Clopper-Pearson confidence interval
def clopper_pearson_interval(observed_successes, total_trials, alpha=0.05):
    """
    Calculate Clopper-Pearson confidence interval for a binomial proportion.
    Handles cases where observed_successes might be None (e.g. missing data)
    or when observed_successes is derived from proportion * total_trials (float).
    """
    if observed_successes is None or total_trials is None or total_trials == 0:
        return (None, None)

    # Ensure k is an integer and clamped between 0 and n
    k = int(round(observed_successes))
    n = int(total_trials)

    # Clamp k to be within [0, n] to handle potential float precision issues
    k = max(0, min(k, n))

    if k == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, k, n - k + 1)
    
    if k == n:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, k + 1, n - k)
    
    return (lower, upper)

def create_plot():
    json_file_path = 'conformal_experiments_data/16rooms-repeated-data-additional.json'
    plot_output_path = 'conformal_experiments_data/plots/16rooms_repeated.png'
    
    # User-defined: Total number of trials for baseline coverage.
    # This value is crucial for calculating meaningful confidence intervals for the baseline.
    # Please set this to the actual number of samples used for baseline coverage calculation.
    n_baseline = 10000 # Example: if baseline coverage was on 1000 samples.
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    data_bucketed = data.get("data_bucketed", {})
    data_baseline = data.get("data_baseline", {})

    # X-axis: number of repeats
    # Assuming repeats are always "1", "2", "3", "4", "5"
    # We'll convert them to integers for plotting
    repeats_str = sorted(data_baseline.keys(), key=int) # Ensures order if keys are not sorted
    repeats_int = [int(r)*8 for r in repeats_str]

    # Update font sizes for better readability
    plt.rcParams.update({
        'font.size': 16,          # Default font size (e.g., for ticks)
        'axes.labelsize': 20,     # X and Y labels
        # 'axes.titlesize': 14,     # Plot title
        'legend.fontsize': 20,    # Legend
        # 'xtick.labelsize': 10,    # X-axis tick labels
        # 'ytick.labelsize': 10     # Y-axis tick labels
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot for data_bucketed
    bucket_size_to_plot = "100"
    if bucket_size_to_plot in data_bucketed:
        y_values = []
        lower_bounds = []
        upper_bounds = []
        # n_bucket_trials = int(bucket_size) # Reverted: This line was part of the change to be undone for total_trials

        for r_str in repeats_str:
            if r_str in data_bucketed[bucket_size_to_plot]:
                coverage = data_bucketed[bucket_size_to_plot][r_str]["bucketed-coverage"]
                # k_bucket is observed_successes, can be float initially
                # Reverted: Use n_baseline for total_trials for bucketed data as per user's correction
                k_bucket = coverage * n_baseline 
                low, high = clopper_pearson_interval(k_bucket, n_baseline, alpha=0.05) 
                y_values.append(coverage)
                lower_bounds.append(low)
                upper_bounds.append(high)
            else:
                y_values.append(None) # Handle missing repeat data if necessary
                lower_bounds.append(None)
                upper_bounds.append(None)
        
        line, = ax.plot(repeats_int, y_values, linestyle='-', label=f'BucketedVaR')
        ax.fill_between(repeats_int, lower_bounds, upper_bounds, color=line.get_color(), alpha=0.2, interpolate=True)
    else:
        print(f"Warning: Bucket size '{bucket_size_to_plot}' not found in data_bucketed.")

    # Plot for data_baseline
    if data_baseline:
        y_baseline_values = []
        lower_baseline_bounds = []
        upper_baseline_bounds = []

        for r_str in repeats_str:
            if r_str in data_baseline:
                coverage = data_baseline[r_str]["coverage"]
                # k_baseline is observed_successes, can be float initially
                k_baseline = coverage * n_baseline
                low, high = clopper_pearson_interval(k_baseline, n_baseline, alpha=0.05)
                y_baseline_values.append(coverage)
                lower_baseline_bounds.append(low)
                upper_baseline_bounds.append(high)
            else:
                y_baseline_values.append(None)
                lower_baseline_bounds.append(None)
                upper_baseline_bounds.append(None)

        line, = ax.plot(repeats_int, y_baseline_values, linestyle='--', label='Baseline')
        ax.fill_between(repeats_int, lower_baseline_bounds, upper_baseline_bounds, color=line.get_color(), alpha=0.2, interpolate=True)

    # Add desired coverage line
    desired_coverage = 0.9
    ax.axhline(y=desired_coverage, color='k', linestyle=':', label=f'Desired Coverage')

    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Empirical quantile coverage") # Changed from "bucketed-coverage" to "Coverage" for generality
    ax.set_xticks(repeats_int) # Ensure x-ticks are 1, 2, 3, 4, 5
    
    # Set y-axis ticks and limits
    y_tick_min = 0.88
    y_tick_max = 0.93
    y_tick_step = 0.01
    ax.set_yticks(np.arange(y_tick_min, y_tick_max + y_tick_step, y_tick_step))
    ax.set_ylim([0.885, 0.922])

    ax.legend()
    ax.grid(True)
    
    plt.savefig(plot_output_path, dpi=300)
    print(f"Plot saved to {plot_output_path}")

    # Reset rcParams to default to avoid affecting other plots if this script is imported
    plt.rcParams.update(plt.rcParamsDefault)

if __name__ == '__main__':
    create_plot()
