import os
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo


class VideoRecorderCallback(BaseCallback):
    """
    Records videos of the agent's performance periodically during training
    using RecordVideo wrapper
    """
    def __init__(self, env_fn, video_folder='policy_recordings', 
                 video_freq=10000, verbose=1):
        super().__init__(verbose)
        self.env_fn = env_fn  # function that returns an environment
        self.video_folder = video_folder
        self.video_freq = video_freq
        os.makedirs(video_folder, exist_ok=True)
        
    def _on_step(self):
        if self.n_calls % self.video_freq == 0:
            # Create a new environment for recording
            env = self.env_fn()
            
            # Get a unique video name based on the current training step
            video_subfolder = os.path.join(self.video_folder, f"step_{self.n_calls}")
            
            try:
                # Wrap the environment with RecordVideo
                env = RecordVideo(
                    env, 
                    video_folder=video_subfolder,
                    episode_trigger=lambda _: True  # Record every episode
                )
                
                # Run one episode
                obs, _ = env.reset()
                done = False
                total_reward = 0
                
                while not done:
                    # Step the environment (recording happens automatically)
                    action, _ = self.model.predict(
                        obs, 
                        # deterministic=True,
                    )
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                
                # Log reward
                self.logger.record('eval/video_reward', total_reward)

                if isinstance(info, dict):
                    for key in info.keys():
                        val = info[key]
                        try:
                            self.logger.record(f"eval/{key}", float(val))
                        except:
                            pass
                if self.verbose > 0:
                    print(f"Video recorded at step {self.n_calls}, reward: {total_reward}")
            
            finally:
                # Close the environment safely
                try:
                    # Try to access the underlying environment
                    if hasattr(env, 'env'):
                        env.env.close()
                    else:
                        env.close()
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Error closing environment: {e}")
            
        return True
    
