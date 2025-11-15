# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains code for the paper "Composing Agents to Minimize Worst-case Risk". The project implements a reinforcement learning framework for composing agents along graph paths to minimize worst-case performance using conformal prediction and bucketed variance estimation techniques.

## Core Architecture

### Agent Graph Framework
The codebase is built around a directed acyclic graph (DAG) representation where:
- **Vertices** represent states or milestones in tasks
- **Edges** represent subtasks that agents must learn to accomplish
- **Paths** through the graph represent complete task sequences

Three main agent graph implementations:
1. **`agents/agent_graph.py`**: Base class defining the graph structure and sampling interface
2. **`agents/rl_agent_graph.py`**: RL agents (using Stable Baselines3 PPO) trained for custom Gymnasium environments
3. **`agents/dirl_agent_graphs.py`**: Integration with DIRL (Jothimurugan et al.) policies for navigation benchmarks

### Key Algorithms
- **`agents/bucketed_var.py`**: Dynamic programming algorithm that divides error budget across buckets to find optimal paths minimizing worst-case risk
- **`agents/baseline_var_estim.py`**: Naive baseline that samples all paths and selects the minimum
- **`agents/calculate_coverage.py`**: Evaluates empirical coverage of selected paths

### Policy Training Workflow
1. Policies are trained **per-edge** or **per-path** using PPO (agents/rl_agent_graph.py:91-268)
2. During path-mode training, initial state distributions are collected from successful rollouts (agents/rl_agent_graph.py:193-227)
3. Policies are saved to `./logs/{project_name}/{edge_task_name}/final_model.zip`
4. Sampling caches are persisted to disk to avoid redundant rollouts

## Environment Setup

### Python Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### MuJoCo Setup (for DIRL benchmarks)
1. Download and extract mujoco200 from https://roboti.us/download/mujoco200_linux.zip
2. Place in `$HOME/.mujoco/mujoco200` (remove `_linux` suffix)
3. Download mjkey.txt from https://www.roboti.us/file/mjkey.txt to `$HOME/.mujoco/`
4. Set environment variable:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
   ```

### DIRL Submodule
Add to PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/worm/dirl
```

### Docker (for BoxRelay benchmark)
1. Adjust CUDA version in Dockerfile line 2 based on local GPU
2. Build: `docker build -t worm:latest .`
3. Run: `bash docker-run-interactive.sh`

## Running Experiments

### DIRL Benchmarks (MouseNav, 16-Rooms, Fetch)

**MouseNav**:
```bash
python -m experiment_scripts.mouse_nav_train_policies
python -m experiment_scripts.mouse_nav_experiments
```

**16-Rooms**:
```bash
python -m experiment_scripts.16rooms_train_policies
python -m experiment_scripts.16rooms_experiments
python -m experiment_scripts.16_rooms_sample_size_buckets_experiments
```

**Fetch**:
```bash
python -m experiment_scripts.fetch_dirl_train_policies
python -m experiment_scripts.fetch_dirl_experiments
python -m experiment_scripts.fetch_sample_size_buckets_experiments
```

**16-Rooms Extended (increasing agents along path)**:
```bash
python -m experiment_scripts.16rooms_repeated_experiments
```

### BoxRelay Benchmark (inside Docker)

Prefix all commands with `xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset"` for headless rendering:

```bash
# Train policies
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m experiment_scripts.boxrelay_benchmark train

# Baseline comparison
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m experiment_scripts.boxrelay_benchmark risk_min

# Sample size and buckets variation
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m experiment_scripts.boxrelay_benchmark sample_size_buckets_experiment
```

### Generating Plots
```bash
python -m experiment_scripts.sample_size_plot
python -m experiment_scripts.buckets_plot
python -m experiment_scripts.16_rooms_repeated_plot
```

## Important Implementation Details

### Caching System
- Sample caches are stored in `{cache_save_file}` to avoid redundant environment rollouts
- Cache keys are constructed from path + target vertex + n_samples (agents/agent_graph.py:50)
- Caches are automatically saved after each sampling operation
- To clear caches, delete the `.pkl` files in the logs directory

### GPU Memory Management
- Models are NOT kept in memory during training to conserve GPU memory (agents/rl_agent_graph.py:184-189)
- GPU cache is explicitly cleared between training runs (agents/rl_agent_graph.py:263-265)
- Models are saved to disk and loaded on-demand for evaluation

### Training vs Evaluation
- Training uses `train_all_paths()` or `train_all_edges()` modes
- Path mode: policies conditioned on reaching vertex via specific path
- Edge mode: policies trained independently per edge
- Initial state distributions propagate through paths in path mode (agents/rl_agent_graph.py:121)

### Wandb Integration
- All training runs log to wandb (project name specified per benchmark)
- Tensorboard logs also saved to `./logs/{project}/{task}/tensorboard`
- Video recordings saved at: `./logs/{project}/{task}/policy_recordings`

### Loss Evaluation
- Environments return `info["loss_eval"]` indicating task success (np.inf = failure)
- Only successful rollouts (loss_eval != np.inf) are collected during initial state sampling (agents/rl_agent_graph.py:203)

## File Organization
- `agents/`: Core agent graph implementations and algorithms
- `experiment_scripts/`: Benchmark-specific training and evaluation scripts
- `experiments_data/`: Collected data and results (JSON format)
- `logs/`: Training logs, models, and video recordings
- `dirl/`: Submodule containing DIRL baseline implementations
- `timeParamCPScores/`: Time-parameterized conformal prediction code
