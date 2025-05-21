# worm
Code for the paper "Composing Agents to Minimize Worst-case Risk"

## Instructions to run data collection
All the collected data is available in the `experiments_data` folder. The following describes how to rerun this data collection.

### Running DIRL benchmarks from Jothimurugan et al. (MouseNav, 16-Rooms, and Fetch)

#### Setup
1. Create a `python3.10` virtual environment and install the python package requirements from `requirements.txt` inside the environment
2. Install `mujoco200` using the following steps
   1. Install mujoco200 from https://roboti.us/download/mujoco200_linux.zip and unzip inside `$HOME/.mujoco/mujoco200` (ensure to remove _linux from name)
   2. Set env variable `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin`
   3. Place the file `mjkey.txt` from https://www.roboti.us/file/mjkey.txt in `$HOME/.mujoco folder`
3. Add `dirl` folder submodule to your `PYTHONPATH` env variable with `PYTHONPATH=$PYTHONPATH:{path_to_dirl_folder}`.

#### Running benchmarks data collection
1. To run the MouseNav benchmark data collection, run the following from within the virtual env:
   ```python -m experiment_scripts.mouse_nav_train_policies && python -m experiment_scripts.mouse_nav_experiments.py```
2. For 16-Rooms:
   ```python -m experiment_scripts.16rooms_train_policies && python -m experiment_scripts.16rooms_experiments && python -m experiment_scripts.16rooms_sample_size_buckets_experiments```
3. For Fetch:
   ```python -m experiment_scripts.fetch_dirl_train_policies && python -m experiment_scripts.fetch_dirl_experiments && python -m experiment_scripts.fetch_sample_size_buckets_experiments.py```
4. For the 16-Rooms extended agent graph experiment (increasing agents along path), run the following after having run the train policies module for 16-Rooms previously shown
   ```python -m experiment_scripts.16rooms_repeated_experiments.py```

### Running BoxRelay benchmark

#### Setup
1. Build docker image called `worm` using `docker build -t worm:latest .` by adjusting the base cuda image in the dockerfile based on your local available cuda version.
2. Run `bash docker-run-interactive.sh` to run the `worm` docker image in interactive mode (open a shell with docker image)

#### Running BoxRelay data collection
From inside the docker image interactive shell, run
1. Train policies: `xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m experiment_scripts.boxrelay_benchmark train`
2. Baseline comparison data: `xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m experiment_scripts.boxrelay_benchmark risk_min`
3. Sample size and buckets variation data: `xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m experiment_scripts.boxrelay_benchmark sample_size_buckets_experiment`

## Instructions to generate plots
1. Sample-size variation plot: `python -m experiment_scripts.sample_size_plot`
2. Buckets variation plot: `python -m experiment_scripts.buckets_plot`
3. Increasing agents plot: `python -m experiment_scripts.16_rooms_repeated_plot`
