from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from tqdm import tqdm
import wandb
from wandb.integration.sb3 import WandbCallback
import pickle

from conformal.nonconformity_score_graph import NonConformityScoreGraph
from conformal.video_recorder_callback import VideoRecorderCallback


class RLTaskGraph(NonConformityScoreGraph):
    def __init__(
            self, 
            spec_graph: List[Dict[int, str]], 
            env_name: str,
            env_kwargs: Optional[dict]=None,
            eval_env_kwargs: Optional[dict]=None,
            cache_save_file: str=None,
        ):
        self.spec_graph = spec_graph
        adj_lists = [[v for v in edges.keys()] for edges in spec_graph]
        super().__init__(adj_lists, cache_save_file=cache_save_file)

        self.env_name = env_name

        # these save policies for all paths mode
        self.path_policies: Dict[Tuple[int], Any] = dict()
        self.init_states: Dict[Tuple[int], List[Any]] = dict()
        self.init_states[(0,)] = None

        # these save policies for per-edge mode
        self.edge_policies: Dict[Tuple[int, int], Any] = dict()

        self.env_kwargs = env_kwargs if env_kwargs else dict()
        self.eval_env_kwargs = eval_env_kwargs if eval_env_kwargs else dict()
    
    def train_all_paths(
            self, 
            wandb_project_name: str="project", 
            n_samples: int=300, 
            training_iters: int=200000,
            final_policy_recordings: int=3,
            policy_class: str="CnnPolicy",
            n_envs: int=4,
        ):
        stack = [(0,)]

        while stack:
            path = stack.pop()
            for target_v in self.adj_lists[path[-1]]:
                target_path = path + (target_v,)
                self._train_edge(
                    path=target_path, 
                    wandb_project_name=wandb_project_name, 
                    n_samples=n_samples,
                    training_iters=training_iters,
                    final_policy_recordings=final_policy_recordings,
                    policy_class=policy_class,
                    n_envs=n_envs,
                )
                stack.append(target_path)

    def train_all_edges(
            self,
            wandb_project_name: str="project",
            training_iters: int=200_000,
            final_policy_recordings: int=3,
            policy_class: str="CnnPolicy",
            n_envs: int=4,
        ):
        for u in range(len(self.adj_lists)):
            for v in self.adj_lists[u]:
                self._train_edge(
                    edge=(u,v),
                    wandb_project_name=wandb_project_name,
                    training_iters=training_iters,
                    final_policy_recordings=final_policy_recordings,
                    policy_class=policy_class,
                    n_envs=n_envs,
                    train_independent_edge=True,
                )

    def _train_edge(
            self, 
            path: List[int]=None, 
            edge: Tuple[int, int]=None,
            wandb_project_name: str="project", 
            n_samples: int=300,
            training_iters: int=200000,
            final_policy_recordings: int=3,
            policy_class: str="CnnPolicy",
            n_envs: int=4,
            train_independent_edge: bool=False,
        ):
        edge = edge if edge else (path[-2], path[-1])
        task_str = self.spec_graph[edge[0]][edge[1]]

        if path is not None:
            # training in path mode
            path_file_str = "-".join(str(i) for i in path)
            edge_task_name = f"path-{path_file_str}-{task_str}"
        else:
            # training in edge mode
            edge_task_name = f"edge-{edge}-{task_str}"

        wandb.init(
            project=wandb_project_name,
            monitor_gym=True,
            name=edge_task_name,
            sync_tensorboard=True,
        )

        edge_init_states = self.init_states[tuple(path[:-1])] if not train_independent_edge else None
        def make_env(rank):
            def _init():
                env = gym.make(
                    self.env_name, 
                    render_mode="rgb_array", 
                    task_str=task_str,
                    init_states=edge_init_states,
                    **self.env_kwargs,
                )
                env = Monitor(env)
                return env
            return _init
        
        def make_eval_env():
            env = gym.make(
                self.env_name, 
                render_mode="rgb_array", 
                task_str=task_str,
                init_states=edge_init_states,
                **self.eval_env_kwargs,
            )
            env = Monitor(env)
            return env
        
        env = DummyVecEnv([make_env(0)])

        eval_callback = EvalCallback(
            env, 
            best_model_save_path=f"./logs/{wandb_project_name}/{edge_task_name}/best_models",
            log_path=f"./logs/{wandb_project_name}/{edge_task_name}/logs", 
            eval_freq=10000,                 
            deterministic=True, 
            render=False,
        )
        wandb_callback = WandbCallback(verbose=2)
        video_callback = VideoRecorderCallback(
            env_fn=make_eval_env,
            video_folder=f"./logs/{wandb_project_name}/{edge_task_name}/policy_recordings",
            video_freq=10000,
            verbose=1,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            print(f"Trying to load model for {edge_task_name}")
            model = PPO.load(f"./logs/{wandb_project_name}/{edge_task_name}/final_model.zip")
        except:
            model = PPO(
                policy_class, 
                env, 
                verbose=1, 
                tensorboard_log=f"./logs/{wandb_project_name}/{edge_task_name}/tensorboard",
            )
            print(f"Training policy for {edge_task_name}")
            model.learn(
                total_timesteps=training_iters,
                callback=[eval_callback, wandb_callback, video_callback],
            )
            model.save(f"./logs/{wandb_project_name}/{edge_task_name}/final_model")

        env.close()
        if path:
            self.path_policies[tuple(path)] = model
        else:
            self.edge_policies[edge] = model

        env = make_eval_env()

        if not train_independent_edge:
            # do n_samples rollouts to obtain starting state distribution for next vertex
            try:
                with open(f"./logs/{wandb_project_name}/{edge_task_name}/path_init_states.pkl", "rb") as f:
                    self.init_states[tuple(path)] = pickle.load(f)
            except:
                next_init_states = []
                print(f"Collecting path samples for {edge_task_name}")
                for _ in tqdm(range(n_samples)):
                    loss_eval = np.inf
                    while loss_eval == np.inf:
                        # only collect successful samples
                        obs, info = env.reset()
                        env_state = info["env_state"]
                        done = False

                        rew = 0
                        while not done:
                            action, _ = model.predict(
                                obs, 
                                # deterministic=True,
                            )
                            obs, r, terminated, truncated, info = env.step(action)
                            rew += r
                            env_state = info["env_state"]
                            loss_eval = info["loss_eval"]
                            done = terminated or truncated

                    next_init_states.append(env_state)
                    wandb.log({"eval/path_samples_cumrew": rew})

                self.init_states[tuple(path)] = next_init_states

                with open(f"./logs/{wandb_project_name}/{edge_task_name}/path_init_states.pkl", "wb") as f:
                    pickle.dump(next_init_states, f)


        #### record final_policy_recordings
        videos_folder = f"./logs/{wandb_project_name}/{edge_task_name}/final_policy_recordings"
        env = RecordVideo(env, video_folder=videos_folder, episode_trigger=lambda _: True)

        print(f"Final policy recordings for {edge_task_name}")
        for _ in tqdm(range(final_policy_recordings)):
            loss_eval = np.inf
            while loss_eval == np.inf:
                obs, info = env.reset()
                done = False
                total_reward = 0

                while not done:
                    action, info = model.predict(
                        obs, 
                        # deterministic=True,
                    )
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    loss_eval = info["loss_eval"]
                    done = terminated or truncated

            wandb.log({"eval/final_policy_recordings_cumrew": total_reward})

        try:
            if hasattr(env, 'env'):
                env.env.close()
            else:
                env.close()
        except Exception as e:
            print(f"Warning: Error closing environment: {e}")

        wandb.finish()

    def load_edge_policy(
            self,
            path: Tuple[int]=None,
            edge: Tuple[int, int]=None,
            log_folder: str="./logs", 
            subfolder: str="riskyminiworldenv1",
        ):
        edge = (path[-2], path[-1]) if path is not None else edge
        task_str = self.spec_graph[edge[0]][edge[1]]

        if path is not None:
            # training in path mode
            path_file_str = "-".join(str(i) for i in path)
            edge_task_name = f"path-{path_file_str}-{task_str}"
        else:
            # training in edge mode
            edge_task_name = f"edge-{edge}-{task_str}"

        model_file = f"{log_folder}/{subfolder}/{edge_task_name}/final_model.zip"
        
        if path is not None:
            self.path_policies[path] = PPO.load(model_file)
        else:
            self.edge_policies[edge] = PPO.load(model_file)

    def load_path_policies(
            self, 
            log_folder: str="./logs", 
            subfolder: str="riskyminiworldenv1"
        ):
        stack = [(0,)]

        while stack:
            path = stack.pop()
            for target_v in self.adj_lists[path[-1]]:
                target_path = path + (target_v,)
                self.load_edge_policy(path=target_path, log_folder=log_folder, subfolder=subfolder)
                stack.append(target_path)

    def load_edge_policies(
            self,
            log_folder: str="./logs", 
            subfolder: str="riskyminiworldenv1",
        ):
        for u in range(len(self.adj_lists)):
            for v in self.adj_lists[u]:
                self.load_edge_policy(edge=(u, v), log_folder=log_folder, subfolder=subfolder)
    
    def sample(self, target_vertex, n_samples, path, path_samples):
        assert len(path_samples) == n_samples

        task_str = self.spec_graph[path[-1]][target_vertex]
        def make_eval_env():
            env = gym.make(
                self.env_name, 
                render_mode="rgb_array", 
                task_str=task_str,
                **self.eval_env_kwargs,
            )
            return env
        
        env = make_eval_env()
        
        if self.path_policies:
            model = self.path_policies[tuple(path) + (target_vertex,)]
        else:
            model = self.edge_policies[(path[-1], target_vertex)]

        next_path_samples = []
        losses = []
        
        print(f"Sampling edge {(path[-1], target_vertex)}")
        for i in tqdm(range(len(path_samples))):
            sample = path_samples[i]

            loss_eval = np.inf
            i = 0
            while loss_eval == np.inf:
                obs, info = env.reset(options={"state": sample})
                loss_eval = info["loss_eval"]
                env_state = info["env_state"]
                done = False

                while not done:
                    action, _ = model.predict(
                        obs, 
                        # deterministic=True
                    )
                    obs, _, terminated, truncated, info = env.step(action)
                    loss_eval = info["loss_eval"]
                    env_state = info["env_state"]
                    done = terminated or truncated

                i += 1
                if i == 20:
                    print(sample)
                    sample = None
                if i == 40:
                    print(f"Unable to complete sample for {(path[-1], target_vertex)}")
                    break

            next_path_samples.append(env_state)
            losses.append(loss_eval)

        return next_path_samples, losses

