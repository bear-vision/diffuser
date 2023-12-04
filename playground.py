import diffuser.utils as utils
import gym
import d4rl # Import required to register environments, you may need to also import the submodule


# import gym
import d4rl_pybullet

# dataset will be automatically downloaded into ~/.d4rl/datasets
env = gym.make('hopper-bullet-mixed-v0')

# interaction with its environment
env.reset()
env.step(env.action_space.sample())

# access to the dataset
dataset = env.get_dataset()
# print(dataset[0])
print(len(dataset['observations'])) # observation data in N x dim_observation
# dataset['actions'] # action data in N x dim_action
# dataset['rewards'] # reward data in N x 1
# dataset['terminals'] # terminal flags in N x 1