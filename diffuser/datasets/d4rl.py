import os
import collections
import numpy as np
import gym
import pdb

# HOLD ADDED
from gym import spaces
import numpy as np



from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#


### HOLD ENTER ##

class CustomOfflineEnv(gym.Env):
    def __init__(self, dataset):
        super(CustomOfflineEnv, self).__init__()

        # Assuming the dataset is a list of tuples (state, action, reward, next_state, done)
        self.dataset = dataset
        self.current_index = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions and observations:
        self.action_space = spaces.Discrete(2)  # Adjust based on your dataset
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)  # Adjust the shape

    def step(self, action):
        # In an offline environment, actions are ignored and data is returned from the dataset
        state, _, reward, next_state, done = self.dataset[self.current_index]
        self.current_index += 1

        return next_state, reward, done, {}

    def reset(self):
        # Reset the environment to an initial state
        self.current_index = 0
        initial_state, _, _, _, _ = self.dataset[0]
        return initial_state

    def render(self, mode='human', close=False):
        # Implement rendering if necessary
        pass

    def get_dataset(self):
        # Return the dataset in the expected format
        return self.dataset


### HOLD ENTER END ##


# MANUALLY LOAD INTO THIS A CUSTOM ENV. 
def load_environment(name):
    print("\n IN LOAD ENV FUNC\n")
    # if type(name) != str:
    #     ## name is already an environment
    #     return name
    # with suppress_output():
    #     wrapped_env = gym.make(name)
    # env = wrapped_env.unwrapped
    # env.max_episode_steps = wrapped_env._max_episode_steps
    # env.name = name

    # print("\n GETTING RAW DATA \n")
    # dataset = env.get_dataset()
    # for key in dataset:
    #     print(key)

    # print(dataset['actions'][0])
    # print(dataset['observations'][0])
    # print(dataset['rewards'][0])
    # print(dataset['terminals'][0])
    # print(dataset['timeouts'][0])


    # return env

    example_dataset = {
        "actions": np.array([
            [0.37950972, 0.31123734, 0.513988, -0.39653406, -0.9828435, -0.20149498, 0.07471082, 0.00157693],
            [-0.241601, 0.147251, 0.423663, -0.312456, 0.786238, -0.564823, 0.142536, -0.215643],
            [0.563462, -0.751246, 0.324576, 0.245356, -0.423567, 0.658735, -0.357845, 0.264573],
            [-0.456378, 0.342578, -0.785642, 0.523874, 0.357843, -0.246357, 0.753824, -0.546372],
            [0.745673, -0.564832, 0.258369, -0.357924, 0.468753, 0.357924, -0.478562, 0.246375]
        ]),
        "observations": np.array([
            [0.78418016, 0.99882466, 0.04459167, -0.01844125, 0.00456757, 0.00768615, 0.04628626, -0.09184544],
            [-0.156372, 0.246357, -0.357842, 0.468752, -0.578643, 0.684753, -0.785634, 0.834756],
            [0.935674, -0.846375, 0.756476, -0.675487, 0.584698, -0.493509, 0.412421, -0.321332],
            [-0.230443, 0.341556, -0.452668, 0.563781, -0.674893, 0.786006, -0.897118, 0.908231],
            [0.019245, -0.128367, 0.237489, -0.346611, 0.455733, -0.564855, 0.673977, -0.783099]
        ]),
        "rewards": np.array([0.22460805, -0.356472, 0.478563, -0.592645, 0.716728]),
        "terminals": np.array([False, False, True, False, True]),
        "timeouts": np.array([False, True, False, True, False])
    }

    env = CustomOfflineEnv(example_dataset)
    env.max_episode_steps = 1000
    env.name = "CUSTOM"
    return env




def get_dataset(env):
    print("\n IN GETDATESET\n")
    dataset = env.get_dataset()

    if 'antmaze' in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    return dataset

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    print("\n IN sequence dataset FUNC\n")
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
