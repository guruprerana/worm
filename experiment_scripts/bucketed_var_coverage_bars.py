#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data():
    """Load coverage data from CSV files."""
    data_dir = '../experiments_data/bucketed_var_coverage_data'

    # Risk levels and their desired coverage levels
    risk_levels = {
        '0.2': {'file': 'risk_level_0_2.csv', 'desired': 80.0},
        '0.1': {'file': 'risk_level_0_1.csv', 'desired': 90.0},
        '0.05': {'file': 'risk_level_0_05.csv', 'desired': 95.0}
    }

    data = {}
    for risk, info in risk_levels.items():
        file_path = os.path.join(data_dir, info['file'])
        df = pd.read_csv(file_path)
        data[risk] = {
            'coverage': df.set_index('benchmark')['coverage'].to_dict(),
            'desired': info['desired']
        }

    return data

def calculate_differences(data):
    """Calculate absolute differences between actual and desired coverage."""
    benchmarks = list(next(iter(data.values()))['coverage'].keys())
    differences = {benchmark: {} for benchmark in benchmarks}

    for risk_level, risk_data in data.items():
        for benchmark in benchmarks:
            actual = risk_data['coverage'][benchmark]
            desired = risk_data['desired']
            differences[benchmark][risk_level] = abs(actual - desired)

    return differences

def create_bar_chart(differences):
    """Create grouped bar chart showing coverage differences."""
    benchmarks = list(differences.keys())
    risk_levels = ['0.2', '0.1', '0.05']

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
        values = [differences[benchmark][risk_level] for risk_level in risk_levels]
        bar = ax.bar(x + i * width, values, width,
                    label=benchmark, alpha=0.85,
                    edgecolor='black', linewidth=0.5)
        bars.append(bar)

    # Customize the plot
    ax.set_xlabel('Risk Levels')
    ax.set_ylabel('Quantile difference (%)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(risk_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
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
    data = load_data()

    # Calculate differences
    differences = calculate_differences(data)

    # Create and show the chart
    fig = create_bar_chart(differences)

    # Save the plot
    output_path = '../experiments_data/plots/bucketed_var_coverage_differences.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved as {output_path}")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()