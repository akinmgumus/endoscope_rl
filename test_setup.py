import torch
import gymnasium as gym
from stable_baselines3 import PPO

# 1. GPU kontrolü
print("CUDA:", torch.cuda.is_available())

# 2. MuJoCo ortamı açılıyor mu?
env = gym.make("Reacher-v5")
obs, info = env.reset()
print("Observation shape:", obs.shape)   # (10,)
print("Action shape:", env.action_space.shape)  # (2,)

# 3. PPO kısa eğitim
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)
print("Kurulum tamam.")
