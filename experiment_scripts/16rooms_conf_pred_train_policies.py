num_iters = 1200
env_num = 3
spec_num = 13

from conformal.all_paths_conformal_pred import all_paths_conformal_pred
from conformal.bucketed_conformal_pred import bucketed_conformal_pred
from conformal.dirl_score_graphs import DIRLCumRewardScoreGraph, DIRLTimeTakenScoreGraph
from spectrl.hierarchy.construction import adj_list_from_task_graph, automaton_graph_from_spec
from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv
from spectrl.main.spec_compiler import ev, seq, choose, alw
from spectrl.util.io import parse_command_line_options, save_log_info, save_object
from spectrl.util.rl import print_performance, get_rollout
from spectrl.rl.ars import HyperParams

from spectrl.examples.rooms_envs import (
    GRID_PARAMS_LIST,
    MAX_TIMESTEPS,
    START_ROOM,
    FINAL_ROOM,
)
from spectrl.envs.rooms import RoomsEnv

import os


render = False
folder = ''
itno = -1

log_info = []

grid_params = GRID_PARAMS_LIST[env_num]

hyperparams = HyperParams(30, num_iters, 30, 15, 0.05, 0.3, 0.15)

print(
    "\n**** Learning Policy for Spec #{} in Env #{} ****".format(
        spec_num, env_num
    )
)

# Step 1: initialize system environment
system = RoomsEnv(grid_params, START_ROOM[env_num], FINAL_ROOM[env_num])

# Step 4: List of specs.
bottomright = (0, 3)
topleft = (3, 0)

# test specs
spec0 = ev(grid_params.in_room((1, 0)))  # 100 iterations works
spec1 = ev(grid_params.in_room((2, 0)))  # 200 iterations works
spec2 = ev(grid_params.in_room(topleft))  # 1000 iterations not enough
spec3 = alw(grid_params.avoid_center((1, 0)), ev(grid_params.in_room((2, 0))))  # 100 works
spec4 = seq(ev(grid_params.in_room((1, 0))), ev(grid_params.in_room(topleft)))  # 200 works
spec5 = seq(ev(grid_params.in_room((2, 0))), ev(grid_params.in_room((2, 2))))  # 200 works
spec6 = alw(grid_params.avoid_center((1, 0)), ev(
    grid_params.in_room(topleft)))  # 400 doesn't work
spec7 = alw(grid_params.avoid_center((2, 0)), ev(grid_params.in_room(topleft)))
spec8 = choose(ev(grid_params.in_room(topleft)),
                ev(grid_params.in_room(bottomright)))

spec9 = choose(alw(grid_params.avoid_center((1, 0)), ev(grid_params.in_room((2, 0)))),
                ev(grid_params.in_room((0, 2))))
spec10 = seq(spec9, ev(grid_params.in_room((2, 2))))
spec10part1 = choose(ev(grid_params.in_room((2, 1))),
                        ev(grid_params.in_room((3, 2))))
spec10part2 = seq(spec10part1,
                    ev(grid_params.in_room((3, 1))))
spec11 = seq(spec10, spec10part2)

spec11part1 = choose(ev(grid_params.in_room((1, 1))),
                        ev(grid_params.in_room((3, 3))))
spec11part2 = seq(spec11part1,
                    alw(grid_params.avoid_center((2, 3)), ev(grid_params.in_room((1, 3)))))
# spec11part2easy = ev(grid_params.in_room((1,3)))
spec12 = seq(spec11, spec11part2)

spec12part1 = choose(ev(grid_params.in_room((1, 1))),
                        ev(grid_params.in_room((0, 3))))
spec12part2 = seq(spec12part1, ev(grid_params.in_room((0, 1))))
spec13 = seq(spec12, spec12part2)

specs = [spec0, spec1, spec2, spec3, spec4, spec5, spec6,
            spec7, spec8, spec9, spec10, spec11, spec12, spec13]

# Step 3: construct abstract reachability graph
_, abstract_reach = automaton_graph_from_spec(specs[spec_num])
print("\n**** Abstract Graph ****")
abstract_reach.pretty_print()

# Step 5: Learn policy
path_policies = abstract_reach.learn_all_paths(
    system,
    hyperparams,
    res_model=None,
    max_steps=20,
    render=render,
    neg_inf=-100,
    safety_penalty=-1,
    num_samples=500,
)

adj_list = adj_list_from_task_graph(abstract_reach.abstract_graph)
terminal_vertices = [i for i in range(len(adj_list)) if i in adj_list[i]]

import dill as pickle

with open("conformal_experiments_data/16rooms-dirl-policies/path_policies.pkl", "wb") as f:
    pickle.dump(path_policies, f)

with open("conformal_experiments_data/16rooms-dirl-policies/adj_list.pkl", "wb") as f:
    pickle.dump(adj_list, f)

with open("conformal_experiments_data/16rooms-dirl-policies/terminal_vertices.pkl", "wb") as f:
    pickle.dump(terminal_vertices, f)
