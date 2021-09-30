# This file was auto-generated by `agentos init` on Sep 14, 2021 15:53:51.
import agentos
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


# A basic agent.
class SB3PPOAgent(agentos.Agent):
    """
    the `entry_points` class attribute is how a component tells the AgentOS
    runtime the different ways that the component can be run (if any).
    `entry_points` is a list of function names that the AgentOS should expose
    to AgentOS users (e.g., via the CLI or runtime client API).
    """
    entry_points = ["run", "learn"]

    def init(self):
        self.sb3_ppo = PPO('MlpPolicy', self.environment)

    def evaluate(self,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
            callback=None,
            reward_threshold=None,
            return_episode_rewards=False,
            warn=True
    ):
        mean_reward, std_reward = evaluate_policy(
            self.sb3_ppo,
            self.sb3_ppo.get_env(),
            int(n_eval_episodes),
            deterministic,
            render,
            callback,
            reward_threshold,
            return_episode_rewards,
            warn
        )
        print(f"Mean reward: {mean_reward}\nStd reward: {std_reward}")

    def learn(self, num_iterations=1):
        self.sb3_ppo.learn(int(num_iterations))