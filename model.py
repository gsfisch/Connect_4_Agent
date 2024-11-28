import gymnasium as gym
from ConnectFourEnv import ConnectFourEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from BabyPlayer import BabyPlayer
from ChildPlayer import ChildPlayer
from TeenagerPlayer import TeenagerPlayer
from AdultPlayer import AdultPlayer

# Choose oponent (only one)
#opponent = BabyPlayer()
#opponent = ChildPlayer()
#opponent = TeenagerPlayer()
opponent = AdultPlayer()

# Create Connect 4 environment
env = ConnectFourEnv(opponent=opponent, render_mode="human")

# Create deep reinforcement learning model (first time only)
""" model = PPO(
    'MlpPolicy',
    env = env,
    device='cpu',
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose = 1
) """

# Load a saved agent
model = PPO.load('Connect-4_agent', env=env)

# Train model
model.learn(total_timesteps=10_000)

# Save model
model.save("Connect-4_agent")

# Evaluate model
eval_env = Monitor(
    ConnectFourEnv()
    )

# Evaluate the model with 10 evaluation episodes and deterministic=True
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100,
                                          deterministic=True)

# Print the results
print(f'Mean reward: {mean_reward}', end='\n')
print(f'Standard reward: {std_reward}', end='\n')
