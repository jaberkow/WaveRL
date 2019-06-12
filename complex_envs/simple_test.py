import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from simple_environment import SimpleCorridor

config = {"corridor_length":10}
env = DummyVecEnv([lambda: SimpleCorridor(config)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100)
model.save("ppo2_cartpole")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_cartpole")

obs = env.reset()
for i in range(20):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    #env.render()
env.render(save=True)