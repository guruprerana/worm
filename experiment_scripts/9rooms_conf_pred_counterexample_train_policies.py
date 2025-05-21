num_iters = 8000
env_num = 5
spec_num = 0
use_gpu = True

from conformal.all_paths_conformal_pred import all_paths_conformal_pred
from conformal.bucketed_conformal_pred import bucketed_conformal_pred
from conformal.dirl_score_graphs import DIRLCumRewardScoreGraph, DIRLTimeTakenScoreGraph
from spectrl.hierarchy.construction import adj_list_from_task_graph, automaton_graph_from_spec
from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv
from spectrl.main.spec_compiler import ev, seq, choose, alw
from spectrl.rl.ddpg.ddpg import DDPGParams
from spectrl.util.io import parse_command_line_options, save_log_info, save_object
from spectrl.util.rl import print_performance, get_rollout
from spectrl.rl.ars import HyperParams

from spectrl.examples.rooms_envs import (
    GRID_PARAMS_LIST,
    MAX_TIMESTEPS,
    START_ROOM,
    FINAL_ROOM,
)
from spectrl.envs.rooms import RoomsEnvCartesian

import os

render = False
folder = ''
itno = -1

log_info = []

grid_params = GRID_PARAMS_LIST[env_num]

# hyperparams = HyperParams(30, num_iters, 30, 15, 0.05, 0.3, 0.15)

print(
    "\n**** Learning Policy for Spec #{} in Env #{} ****".format(
        spec_num, env_num
    )
)

# Step 1: initialize system environment
system = RoomsEnvCartesian(grid_params, START_ROOM[env_num], FINAL_ROOM[env_num])

state_dim = system.observation_space.shape[0]
action_dim = system.action_space.shape[0]
action_bound = system.action_space.high
hyperparams = DDPGParams(state_dim, action_dim, action_bound,
                            minibatch_size=256, num_episodes=num_iters,
                            discount=0.95, actor_hidden_dim=256,
                            critic_hidden_dim=256, epsilon_decay=3e-6,
                            decay_function='linear', steps_per_update=100,
                            gradients_per_update=100, buffer_size=200000,
                            sigma=0.15, epsilon_min=0.3, target_noise=0.0003,
                            target_clip=0.003, warmup=1000)

# Step 4: List of specs.
if env_num == 2 or env_num == 5:
    bottomright = (0, 2)
    topleft = (2, 0)
if env_num == 3 or env_num == 4:
    bottomright = (0, 3)
    topleft = (3, 0)

spec0 = seq(
    choose(
        alw(grid_params.avoid_center_without_scaling((1, 0), 5.0), ev(grid_params.in_room_without_scaling(topleft))),
        alw(grid_params.avoid_center_without_scaling((0, 1), 7.5), ev(grid_params.in_room_without_scaling(bottomright)))
    ),
    ev(grid_params.in_room_without_scaling(FINAL_ROOM[env_num]))
)
spec1 = seq(
    alw(grid_params.avoid_center_without_scaling((0, 1), 7.5), ev(grid_params.in_room_without_scaling(bottomright))),
    ev(grid_params.in_room_without_scaling(FINAL_ROOM[env_num]))
)
spec2 = alw(grid_params.avoid_center_without_scaling((1, 0), 7.5), ev(grid_params.in_room_without_scaling(topleft)))
spec3 = seq(
    ev(grid_params.in_room_without_scaling(bottomright)),
    ev(grid_params.in_room_without_scaling(FINAL_ROOM[env_num]))
)

specs = [spec0, spec1, spec2, spec3]

# Step 3: construct abstract reachability graph
_, abstract_reach = automaton_graph_from_spec(specs[spec_num])
print("\n**** Abstract Graph ****")
abstract_reach.pretty_print()

# Step 5: Learn policy
path_policies = abstract_reach.learn_all_paths(
    system,
    hyperparams,
    res_model=None,
    max_steps=40,
    render=render,
    neg_inf=-100,
    safety_penalty=-1,
    num_samples=500,
    use_gpu=use_gpu,
    algo="ddpg",
    alpha=0,
)

adj_list = adj_list_from_task_graph(abstract_reach.abstract_graph)
terminal_vertices = [i for i in range(len(adj_list)) if i in adj_list[i]]

import dill as pickle

with open("conformal_experiments_data/9rooms-counterexample-policies/path_policies.pkl", "wb") as f:
    pickle.dump(path_policies, f)

with open("conformal_experiments_data/9rooms-counterexample-policies/adj_list.pkl", "wb") as f:
    pickle.dump(adj_list, f)

with open("conformal_experiments_data/9rooms-counterexample-policies/terminal_vertices.pkl", "wb") as f:
    pickle.dump(terminal_vertices, f)
