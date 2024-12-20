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
from AdultSmarterPlayer import AdultSmarterPlayer


# Choose oponent (only one)
#opponent = BabyPlayer()
#opponent = ChildPlayer()
#opponent = TeenagerPlayer()
#opponent = AdultPlayer()
opponent = AdultSmarterPlayer()


# Create Connect 4 environment
env = ConnectFourEnv(opponent=opponent, render_mode="rgb_array") # rgb_array


# Create deep reinforcement learning model (first time only)
#model = PPO(
#            'MlpPolicy', env = env, device='cpu', 
#            n_steps = 1024, batch_size = 64, n_epochs = 4, 
#            gamma = 0.999, gae_lambda = 0.98, ent_coef = 0.01,
#            verbose = 1) 


# Load a saved agent
model = PPO.load('Connect-4_agent_Curriculum2_only.zip', env=env) # Connect-4_agent_Adult_Smarter_only.zip


# Train model
model.learn(total_timesteps=500_000) #100_000


# Save model
model.save("Connect-4_agent_Curriculum2_only") # Connect-4_agent_Adult_Smarter_only


# Evaluate model
eval_env = Monitor(
    ConnectFourEnv()
    )


# Evaluate the model with 10 evaluation episodes and deterministic=True
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100,
                                          deterministic=True)


# Print the results
print(f'Games: {ConnectFourEnv.games_played}', end='\n')
print(f'Games Won: {ConnectFourEnv.win_amount}', end='\n')
print(f'Games lost: {ConnectFourEnv.lost_amount}', end='\n')
print(f'Games drawn: {ConnectFourEnv.draw_count}', end='\n')
print(f'Games truncated: {ConnectFourEnv.truncated_count}', end='\n')
print(f'Mean reward: {mean_reward}', end='\n')
print(f'Standard reward: {std_reward}', end='\n')
