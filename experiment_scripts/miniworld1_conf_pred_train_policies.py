import conformal.miniworld
from conformal.miniworld import RiskyMiniworldEnv1
from conformal.rl_task_graph import RLTaskGraph

# spec_graph = [
#     {
#         1: RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_BOTTOM_ENTRY,
#         2: RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_TOP_ENTRY,
#     },
#     {
#         3: RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_BOTTOM_EXIT,
#     },
#     {
#         4: RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_TOP_EXIT,
#     },
#     {
#         5: RiskyMiniworldEnv1.Tasks.GOTO_RIGHT_HALL,
#     },
#     {
#         5: RiskyMiniworldEnv1.Tasks.GOTO_RIGHT_HALL,
#     },
#     {}
# ]
spec_graph = [
    {
        1: RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_BOTTOM_ENTRY,
    },
    {
        2: RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_BOTTOM_EXIT,
    },
    {
        3: RiskyMiniworldEnv1.Tasks.GOTO_RIGHT_HALL,
    },
    {}
]
wandb_project_name = "riskyminiworldenv1-topview"

env_kwargs = {"view": "top"}

task_graph = RLTaskGraph(spec_graph, "RiskyMiniworldEnv1-v0", env_kwargs=env_kwargs, eval_env_kwargs=env_kwargs)
task_graph.train_all_paths(wandb_project_name, 200, 300_000)
