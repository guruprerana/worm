import conformal.miniworld
from conformal.miniworld import FairCollection
from conformal.rl_task_graph import RLTaskGraph

spec_graph = [
    {
        1: FairCollection.Tasks.COLLECT_BOXES,
    },
    {},
]
wandb_project_name = "faircollectionenv-agentview"

env_kwargs = {"view": "agent"}

task_graph = RLTaskGraph(spec_graph, "FairCollection-v0", env_kwargs=env_kwargs, eval_env_kwargs=env_kwargs)
task_graph.train_all_paths(wandb_project_name, 200, 2_000_000, policy_class="MultiInputPolicy")
