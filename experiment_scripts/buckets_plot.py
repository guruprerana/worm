import json
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import beta

# Define file paths
BASE_DATA_DIR = "conformal_experiments_data"
FILE1_NAME = "fetch-buckets.json"  # Changed
FILE2_NAME = "16rooms-spec13-dirl-buckets.json"  # Changed
FILE3_NAME = "boxrelay-buckets.json"  # Changed
OUTPUT_IMAGE_NAME = f"{BASE_DATA_DIR}/plots/buckets_vs_coverage.png"  # Changed

# Constants for CI calculation
N_SAMPLES_FOR_CI = 10000.0
ALPHA_CI = 0.05

# Construct full paths
file1_path = os.path.join(BASE_DATA_DIR, FILE1_NAME)
file2_path = os.path.join(BASE_DATA_DIR, FILE2_NAME)
file3_path = os.path.join(BASE_DATA_DIR, FILE3_NAME)

# Function to load data from JSON file
def load_and_process_data(json_filepath):
    """Loads data from a JSON file, extracts number of buckets, coverages,
       and calculates 95% Clopper-Pearson CIs for coverages. Sorts by number of buckets."""
    num_buckets_out = []  # Changed
    coverages_out = []
    lower_bounds_out = []
    upper_bounds_out = []
    
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        
        temp_data_for_sorting = []
        # Assuming data is a dictionary where keys are number of buckets (as strings)
        # and values are dicts with a 'coverage' key.
        for item_key in data:
            try:
                num_buckets_val = int(item_key)  # Changed
                coverage_val = data[item_key]['coverage']
            except (ValueError, KeyError, TypeError) as e:
                print(f"Warning: Skipping malformed data item '{item_key}': {data.get(item_key)} in {json_filepath}. Error: {e}")
                continue

            if not (0 <= coverage_val <= 1):
                print(f"Warning: Coverage value {coverage_val} for num_buckets {num_buckets_val} in {json_filepath} is outside [0,1]. Clamping to [0,1] for CI.") # Changed
                coverage_val = min(max(coverage_val, 0.0), 1.0)
            
            k = round(coverage_val * N_SAMPLES_FOR_CI)
            n_ci = N_SAMPLES_FOR_CI

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
            
            temp_data_for_sorting.append((num_buckets_val, coverage_val, lower_bound_clamped, upper_bound_clamped)) # Changed
            
        if temp_data_for_sorting:
            temp_data_for_sorting.sort(key=lambda x: x[0])  # Sort by num_buckets_val
            num_buckets_out = [item[0] for item in temp_data_for_sorting]  # Changed
            coverages_out = [item[1] for item in temp_data_for_sorting]
            lower_bounds_out = [item[2] for item in temp_data_for_sorting]
            upper_bounds_out = [item[3] for item in temp_data_for_sorting]
        elif data:
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
    except Exception as e:
        print(f"An unexpected error occurred while processing {json_filepath}: {e}")
        return None, None, None, None
        
    return num_buckets_out, coverages_out, lower_bounds_out, upper_bounds_out # Changed

def main():
    # Load data for all files
    num_buckets1, coverages1, lowers1, uppers1 = load_and_process_data(file1_path) # Changed
    label1 = "Fetch"

    num_buckets2, coverages2, lowers2, uppers2 = load_and_process_data(file2_path) # Changed
    label2 = "16-Rooms"

    num_buckets3, coverages3, lowers3, uppers3 = load_and_process_data(file3_path) # Changed
    label3 = "BoxRelay"

    datasets_valid = all([
        num_buckets1, coverages1, lowers1, uppers1, # Changed
        num_buckets2, coverages2, lowers2, uppers2, # Changed
        num_buckets3, coverages3, lowers3, uppers3  # Changed
    ])

    if not datasets_valid:
        print("Could not generate plot due to data loading errors or empty datasets. Please check file paths and content.")
        return

    plt.rcParams.update({
        'font.size': 16,          # Default font size (e.g., for ticks)
        'axes.labelsize': 20,     # X and Y labels
        # 'axes.titlesize': 14,     # Plot title
        'legend.fontsize': 20,    # Legend
        # 'xtick.labelsize': 10,    # X-axis tick labels
        # 'ytick.labelsize': 10     # Y-axis tick labels
    })
    fig, ax = plt.subplots(figsize=(8, 6)) # Set figure size

    line1, = ax.plot(num_buckets1, coverages1, linestyle='-', label=label1) # Changed
    ax.fill_between(num_buckets1, lowers1, uppers1, alpha=0.2, color=line1.get_color()) # Changed

    line2, = ax.plot(num_buckets2, coverages2, linestyle='-', label=label2) # Changed
    ax.fill_between(num_buckets2, lowers2, uppers2, alpha=0.2, color=line2.get_color()) # Changed

    line3, = ax.plot(num_buckets3, coverages3, linestyle='-', label=label3) # Changed
    ax.fill_between(num_buckets3, lowers3, uppers3, alpha=0.2, color=line3.get_color()) # Changed

    ax.axhline(y=0.9, color="k", linestyle=':', label="Desired coverage")

    ax.set_xlabel("Number of buckets") # Changed
    ax.set_ylabel("Empirical quantile coverage")
    ax.legend()
    ax.grid(True)

    # Adjust x-axis ticks to prevent overlap
    all_num_buckets = sorted(list(set(num_buckets1 + num_buckets2 + num_buckets3))) # Changed
    if len(all_num_buckets) > 5: # Changed
        num_ticks = 5
        tick_indices = [int(i * (len(all_num_buckets) - 1) / (num_ticks - 1)) for i in range(num_ticks)] # Changed
        if tick_indices[-1] != len(all_num_buckets) - 1: # Changed
            tick_indices[-1] = len(all_num_buckets) - 1 # Changed
        
        custom_ticks = [all_num_buckets[i] for i in sorted(list(set(tick_indices)))] # Changed
        ax.set_xticks(custom_ticks)
    elif all_num_buckets: # Changed
        ax.set_xticks(all_num_buckets) # Changed

    plt.tight_layout()

    try:
        plt.savefig(OUTPUT_IMAGE_NAME, dpi=300) # Added dpi argument
        print(f"Plot saved to {os.path.abspath(OUTPUT_IMAGE_NAME)}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == '__main__':
    main()
