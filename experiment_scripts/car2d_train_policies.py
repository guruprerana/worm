num_iters = 300
spec_num = 8

render = False
folder = ''
itno = -1

from spectrl.hierarchy.construction import adj_list_from_task_graph, automaton_graph_from_spec
from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv
from spectrl.main.monitor import Resource_Model
from spectrl.main.spec_compiler import ev, alw, seq, choose
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.util.rl import print_performance, get_rollout
from spectrl.rl.ars import HyperParams
from spectrl.envs.car2d import VC_Env
from numpy import linalg as LA

import numpy as np
import os


# Define the resource model
# Fuel consumption proportional to distance from x-axis and the velocity
# sys_state: np.array(2)
# res_state: np.array(1)
# sys_action: np.array(2)
def resource_delta(sys_state, res_state, sys_action):
    return np.array([res_state[0] - 0.1 * abs(sys_state[0]) * LA.norm(sys_action)])


# Define the specification
# 1. Relevant atomic predicates:
# a. Reach predicate
#    goal: np.array(2), err: float
def reach(goal, err):
    def predicate(sys_state, res_state):
        return (min([sys_state[0] - goal[0],
                     goal[0] - sys_state[0],
                     sys_state[1] - goal[1],
                     goal[1] - sys_state[1]]) + err)
    return predicate


# b. Avoid predicate
#    obstacle: np.array(4): [x_min, y_min, x_max, y_max]
def avoid(obstacle):
    def predicate(sys_state, res_state):
        return 10 * max([obstacle[0] - sys_state[0],
                         obstacle[1] - sys_state[1],
                         sys_state[0] - obstacle[2],
                         sys_state[1] - obstacle[3]])
    return predicate


def have_fuel(sys_state, res_state):
    return res_state[0]


# Goals and obstacles
gtop = np.array([5.0, 10.0])
gbot = np.array([5.0, 0.0])
gright = np.array([10.0, 0.0])
gcorner = np.array([10.0, 10.0])
gcorner2 = np.array([0.0, 10.0])
origin = np.array([0.0, 0.0])
obs = np.array([4.0, 4.0, 6.0, 6.0])

# Specifications
spec1 = alw(avoid(obs), ev(reach(gtop, 1.0)))
spec2 = alw(avoid(obs), alw(have_fuel, ev(reach(gtop, 1.0))))
spec3 = seq(alw(avoid(obs), ev(reach(gtop, 1.0))),
            alw(avoid(obs), ev(reach(gbot, 1.0))))
spec4 = seq(choose(alw(avoid(obs), ev(reach(gtop, 1.0))), alw(avoid(obs), ev(reach(gright, 1.0)))),
            alw(avoid(obs), ev(reach(gcorner, 1.0))))
spec5 = seq(spec3, alw(avoid(obs), ev(reach(gright, 1.0))))
spec6 = seq(spec5, alw(avoid(obs), ev(reach(gcorner, 1.0))))
spec7 = seq(spec6, alw(avoid(obs), ev(reach(origin, 1.0))))


# Examples: Choice but greedy doesn't work
gt1 = np.array([3.0, 4.0])
gt2 = np.array([6.0, 0.0])
gfinal = np.array([3.0, 7.0])
gfinal2 = np.array([7.0, 4.0])

spec8 = seq(choose(alw(avoid(obs), ev(reach(gt2, 0.5))), alw(avoid(obs), ev(reach(gt1, 0.5)))),
            alw(avoid(obs), ev(reach(gfinal, 0.5))))

spec9 = seq(choose(alw(avoid(obs), ev(reach(gt2, 0.5))), alw(avoid(obs), ev(reach(gt1, 0.5)))),
            alw(avoid(obs), ev(reach(gfinal2, 0.5))))


spec10 = choose(alw(avoid(obs), ev(reach(gt1, 0.5))),
                alw(avoid(obs), ev(reach(gt2, 0.5))))

specs = [spec1, spec2, spec3, spec4, spec5, spec6, spec7, spec8, spec9, spec10]

lb = [10., 20., 10., 10., 10., 9., 9., 9., 9., 9.]

hyperparams = HyperParams(30, num_iters, 20, 8, 0.05, 1, 0.2)

print('\n**** Learning Policy for Spec {} for {} Iterations ****'.format(spec_num, num_iters))

# Step 1: initialize system environment
system = VC_Env(500, std=0.05)

# Step 2 (optional): construct resource model
resource = Resource_Model(2, 2, 1, np.array([7.0]), resource_delta)

# Step 3: construct abstract reachability graph
_, abstract_reach = automaton_graph_from_spec(specs[spec_num])
print('\n**** Abstract Graph ****')
abstract_reach.pretty_print()

path_policies = abstract_reach.learn_all_paths(
    system, 
    hyperparams, 
    res_model=resource, 
    max_steps=20,
    neg_inf=-lb[spec_num], 
    safety_penalty=-1, 
    num_samples=500, 
    render=render,
    use_gpu=True,
)

adj_list = adj_list_from_task_graph(abstract_reach.abstract_graph)
terminal_vertices = [i for i in range(len(adj_list)) if i in adj_list[i]]

import dill as pickle

with open("conformal_experiments_data/car2d-policies/path_policies.pkl", "wb") as f:
    pickle.dump(path_policies, f)

with open("conformal_experiments_data/car2d-policies/adj_list.pkl", "wb") as f:
    pickle.dump(adj_list, f)

with open("conformal_experiments_data/car2d-policies/terminal_vertices.pkl", "wb") as f:
    pickle.dump(terminal_vertices, f)

