import json
import matplotlib.pyplot as plt
import os
import math  # Added for sqrt
from scipy.stats import beta  # Added for Clopper-Pearson interval

# Define file paths
# Assumes 'conformal_experiments_data' is a subdirectory in the CWD or an accessible path
# and the output image will be saved in the CWD.
BASE_DATA_DIR = "conformal_experiments_data"
FILE1_NAME = "fetch-sample-size.json"
FILE2_NAME = "16rooms-spec13-dirl-sample-size.json"
FILE3_NAME = "boxrelay-sample-size.json"  # New file
OUTPUT_IMAGE_NAME = f"{BASE_DATA_DIR}/plots/sample_size_vs_coverage.png"

# Constants for CI calculation
N_SAMPLES_FOR_CI = 10000.0  # Number of samples used for empirical coverage
ALPHA_CI = 0.05             # Alpha for 95% confidence interval (1 - 0.95)

# Construct full paths
file1_path = os.path.join(BASE_DATA_DIR, FILE1_NAME)
file2_path = os.path.join(BASE_DATA_DIR, FILE2_NAME)
file3_path = os.path.join(BASE_DATA_DIR, FILE3_NAME)  # New file path

# Function to load data from JSON file
def load_and_process_data(json_filepath):
    """Loads data from a JSON file, extracts sample sizes, coverages,
       and calculates 95% Clopper-Pearson CIs for coverages. Sorts by sample size."""
    sample_sizes_out = []
    coverages_out = []
    lower_bounds_out = []
    upper_bounds_out = []
    
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        
        temp_data_for_sorting = []
        # Assuming data is a dictionary where keys are sample sizes (as strings)
        # and values are dicts with a 'coverage' key.
        for item_key in data:
            try:
                sample_size = int(item_key)
                coverage_val = data[item_key]['coverage']
            except (ValueError, KeyError, TypeError) as e:
                print(f"Warning: Skipping malformed data item '{item_key}': {data.get(item_key)} in {json_filepath}. Error: {e}")
                continue

            # Validate and clamp coverage to [0,1] for CI calculation
            if not (0 <= coverage_val <= 1):
                print(f"Warning: Coverage value {coverage_val} for sample size {sample_size} in {json_filepath} is outside [0,1]. Clamping to [0,1] for CI.")
                coverage_val = min(max(coverage_val, 0.0), 1.0)
            
            # Calculate Clopper-Pearson CI for coverage
            k = round(coverage_val * N_SAMPLES_FOR_CI)  # Number of "successes"
            n_ci = N_SAMPLES_FOR_CI  # Number of trials

            if k < 0: k = 0  # Ensure k is not negative
            if k > n_ci: k = n_ci  # Ensure k does not exceed n

            lower_bound = 0.0
            if k > 0:  # Lower bound is 0 if k = 0
                lower_bound = beta.ppf(ALPHA_CI / 2, k, n_ci - k + 1)
            
            upper_bound = 1.0
            if k < n_ci:  # Upper bound is 1 if k = n
                upper_bound = beta.ppf(1 - ALPHA_CI / 2, k + 1, n_ci - k)
            
            # Ensure CI bounds are within [0,1] (beta.ppf should ensure this, but good practice)
            lower_bound_clamped = max(0.0, lower_bound)
            upper_bound_clamped = min(1.0, upper_bound)
            
            temp_data_for_sorting.append((sample_size, coverage_val, lower_bound_clamped, upper_bound_clamped))
            
        if temp_data_for_sorting:
            temp_data_for_sorting.sort(key=lambda x: x[0])  # Sort by sample_size
            sample_sizes_out = [item[0] for item in temp_data_for_sorting]
            coverages_out = [item[1] for item in temp_data_for_sorting]
            lower_bounds_out = [item[2] for item in temp_data_for_sorting]
            upper_bounds_out = [item[3] for item in temp_data_for_sorting]
        elif data:  # File was opened and JSON parsed, but no valid items found
            print(f"Warning: No valid data items processed from {json_filepath}.")
            
    except FileNotFoundError:
        print(f"Error: File not found at {json_filepath}")
        return None, None, None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}")
        return None, None, None, None
    except AttributeError: 
        print(f"Error: Data in {json_filepath} is not in the expected dictionary format.")
        return None, None, None, None
    # Catch other unexpected errors during processing
    except Exception as e:
        print(f"An unexpected error occurred while processing {json_filepath}: {e}")
        return None, None, None, None
        
    return sample_sizes_out, coverages_out, lower_bounds_out, upper_bounds_out

def main():
    # Load data for all files
    sample_sizes1, coverages1, lowers1, uppers1 = load_and_process_data(file1_path)
    label1 = "Fetch"

    sample_sizes2, coverages2, lowers2, uppers2 = load_and_process_data(file2_path)
    label2 = "16-Rooms"

    sample_sizes3, coverages3, lowers3, uppers3 = load_and_process_data(file3_path)
    label3 = "BoxRelay"

    # Proceed with plotting only if all data (including CIs) was loaded successfully
    # Check if essential lists are non-empty (they are empty if None was returned or no data points)
    datasets_valid = all([
        sample_sizes1, coverages1, lowers1, uppers1,
        sample_sizes2, coverages2, lowers2, uppers2,
        sample_sizes3, coverages3, lowers3, uppers3
    ])

    if not datasets_valid:
        print("Could not generate plot due to data loading errors or empty datasets. Please check file paths and content.")
        return

    # Set global font size
    plt.rcParams.update({
        'font.size': 16,          # Default font size (e.g., for ticks)
        'axes.labelsize': 20,     # X and Y labels
        # 'axes.titlesize': 14,     # Plot title
        'legend.fontsize': 20,    # Legend
        # 'xtick.labelsize': 10,    # X-axis tick labels
        # 'ytick.labelsize': 10     # Y-axis tick labels
    })

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot data with confidence intervals
    # Dataset 1
    line1, = ax.plot(sample_sizes1, coverages1, linestyle='-', label=label1)
    ax.fill_between(sample_sizes1, lowers1, uppers1, alpha=0.2, color=line1.get_color())

    # Dataset 2
    line2, = ax.plot(sample_sizes2, coverages2, linestyle='-', label=label2)
    ax.fill_between(sample_sizes2, lowers2, uppers2, alpha=0.2, color=line2.get_color())

    # Dataset 3
    line3, = ax.plot(sample_sizes3, coverages3, linestyle='-', label=label3)
    ax.fill_between(sample_sizes3, lowers3, uppers3, alpha=0.2, color=line3.get_color())

    # Add a horizontal line for desired coverage
    ax.axhline(y=0.9, color="k", linestyle=':', label="Desired coverage")

    # Set labels and title
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Empirical quantile coverage")
    # ax.set_title("Sample Size vs. Coverage")
    ax.legend()
    ax.grid(True)

    # Adjust x-axis ticks to prevent overlap
    # Combine all sample sizes and sort them to find the range
    all_sample_sizes = sorted(list(set(sample_sizes1 + sample_sizes2 + sample_sizes3)))  # Include third dataset
    if len(all_sample_sizes) > 5:  # Only adjust if there are many ticks
        # Select a subset of ticks, e.g., 5 ticks spread across the range
        num_ticks = 5
        tick_indices = [int(i * (len(all_sample_sizes) - 1) / (num_ticks - 1)) for i in range(num_ticks)]
        # Ensure the last tick is included if not already
        if tick_indices[-1] != len(all_sample_sizes) - 1:
            tick_indices[-1] = len(all_sample_sizes) - 1
        
        custom_ticks = [all_sample_sizes[i] for i in sorted(list(set(tick_indices)))]  # Use set to remove duplicates if any
        ax.set_xticks(custom_ticks)
    elif all_sample_sizes:  # If few ticks, use them all
        ax.set_xticks(all_sample_sizes)

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(OUTPUT_IMAGE_NAME, dpi=300)
        print(f"Plot saved to {os.path.abspath(OUTPUT_IMAGE_NAME)}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # Optional: Show plot (uncomment if you want to display it)
    # plt.show()

if __name__ == '__main__':
    main()
