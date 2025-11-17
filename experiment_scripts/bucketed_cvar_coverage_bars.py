#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_benchmark_data():
    """Load coverage data from JSON files for all benchmarks."""
    data_dir = 'experiments_data'

    # Benchmark configurations
    benchmarks = {
        '16-Rooms': {
            'file': '16rooms-spec13-dirl-cum-reward.json',
            'label': '16-Rooms'
        },
        'BoxRelay': {
            'file': 'boxrelay-time-taken.json',
            'label': 'BoxRelay'
        },
        'Fetch': {
            'file': 'fetch-spec6-cum-reward.json',
            'label': 'Fetch'
        },
        'MouseNav': {
            'file': 'mousenav-cum-reward.json',
            'label': 'MouseNav'
        }
    }

    # Risk levels to extract
    risk_levels = ['0.2', '0.1', '0.05']

    data = {}

    for benchmark_name, benchmark_info in benchmarks.items():
        file_path = os.path.join(data_dir, benchmark_info['file'])

        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            benchmark_data = {}

            # Get the bucket key from metadata
            bucket_key = None
            if 'metadata' in json_data and 'total_buckets' in json_data['metadata']:
                total_buckets = json_data['metadata']['total_buckets']
                if total_buckets and len(total_buckets) > 0:
                    bucket_key = str(total_buckets[0])

            if not bucket_key:
                print(f"Warning: Could not determine bucket key for {benchmark_name}")
                data[benchmark_name] = {
                    'differences': {rl: None for rl in risk_levels},
                    'label': benchmark_info['label']
                }
                continue

            for risk_level in risk_levels:
                if risk_level in json_data:
                    risk_data = json_data[risk_level]

                    if bucket_key in risk_data:
                        bucket_data = risk_data[bucket_key]

                        # Extract coverages
                        bucketed_cvar_cov = bucket_data.get('bucketed-cvar-coverage')
                        baseline_cvar_cov = bucket_data.get('cvar-coverage')

                        if bucketed_cvar_cov is not None and baseline_cvar_cov is not None:
                            # Calculate absolute difference in percentage
                            diff = abs(bucketed_cvar_cov - baseline_cvar_cov) * 100
                            benchmark_data[risk_level] = diff
                        else:
                            print(f"Warning: Missing coverage data for {benchmark_name} at risk level {risk_level}")
                            print(f"  bucketed-cvar-coverage: {bucketed_cvar_cov}, cvar-coverage: {baseline_cvar_cov}")
                            benchmark_data[risk_level] = None
                    else:
                        print(f"Warning: Bucket key '{bucket_key}' not found in {benchmark_name} at risk level {risk_level}")
                        benchmark_data[risk_level] = None
                else:
                    print(f"Warning: Risk level {risk_level} not found in {benchmark_name}")
                    benchmark_data[risk_level] = None

            data[benchmark_name] = {
                'differences': benchmark_data,
                'label': benchmark_info['label']
            }

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            data[benchmark_name] = {
                'differences': {rl: None for rl in risk_levels},
                'label': benchmark_info['label']
            }
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
            data[benchmark_name] = {
                'differences': {rl: None for rl in risk_levels},
                'label': benchmark_info['label']
            }
        except Exception as e:
            print(f"Error processing {benchmark_name}: {e}")
            data[benchmark_name] = {
                'differences': {rl: None for rl in risk_levels},
                'label': benchmark_info['label']
            }

    return data

def create_bar_chart(data):
    """Create grouped bar chart showing CVaR coverage differences."""
    risk_levels = ['0.2', '0.1', '0.05']
    benchmarks = ['16-Rooms', 'BoxRelay', 'Fetch', 'MouseNav']

    # Prepare data for plotting
    x = np.arange(len(risk_levels))
    width = 0.2

    # Set global font size to match sample_size_plot.py
    plt.rcParams.update({
        'font.size': 16,          # Default font size (e.g., for ticks)
        'axes.labelsize': 20,     # X and Y labels
        'legend.fontsize': 20,    # Legend
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create bars for each benchmark using matplotlib default colors
    bars = []
    for i, benchmark in enumerate(benchmarks):
        if benchmark not in data:
            print(f"Warning: {benchmark} not in data, skipping")
            continue

        values = []
        for risk_level in risk_levels:
            diff = data[benchmark]['differences'].get(risk_level)
            # Use 0 for None values (will be visible as no bar)
            values.append(diff if diff is not None else 0.0)

        bar = ax.bar(x + i * width, values, width,
                    label=data[benchmark]['label'], alpha=0.85,
                    edgecolor='black', linewidth=0.5)
        bars.append(bar)

    # Customize the plot
    ax.set_xlabel('Risk Levels')
    ax.set_ylabel('CVaR coverage difference (%)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(risk_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:  # Only show label if there's a value
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    return fig

def main():
    """Main function to generate the bar chart."""
    # Load data
    data = load_benchmark_data()

    # Create and show the chart
    fig = create_bar_chart(data)

    # Save the plot
    output_dir = 'experiments_data/plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'bucketed_cvar_coverage_differences.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved as {output_path}")

    plt.close()

if __name__ == "__main__":
    main()
