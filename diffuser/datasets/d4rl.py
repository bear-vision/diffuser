import os
import collections
import numpy as np
import gym
import pdb
import os

# HOLD ADDED
from gym import spaces
import numpy as np
import h5py




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




def load_environment(data_name):

    # ORIGINAL FUNCTION
    # name = data_name
    # if type(name) != str:
    #     ## name is already an environment
    #     return name
    # with suppress_output():
    #     wrapped_env = gym.make(name)
    # env = wrapped_env.unwrapped
    # env.max_episode_steps = wrapped_env._max_episode_steps
    # env.name = name

    converted_data = get_h5_data(data_name)
    env = CustomOfflineEnv(converted_data)
    max_episode_steps = calculate_max_episode_steps(converted_data)
    env.max_episode_steps = max_episode_steps
    env.name = "{}".format(data_name)

    return env


def truncate_arrays(arrays):
    return [arr[:10] for arr in arrays]


def get_h5_data(filename):
    """
    Load the contents of an HDF5 file into a dictionary.

    :param filename: Path to the HDF5 file
    :return: Dictionary with dataset names as keys and dataset contents as values
    """

    filename = os.path.expanduser('~/diffuser/data_hold/{}.h5'.format(filename))

    data_dict = {}
    with h5py.File(filename, 'r') as file:
        for name in file:
            data_dict[name] = file[name][()]
    return data_dict


def calculate_max_episode_steps(env_data):
    terminals = env_data["terminals"]
    timeouts = env_data["timeouts"]

    episode_lengths = []
    episode_length = 0

    for terminal, timeout in zip(terminals, timeouts):
        episode_length += 1
        if terminal or timeout:
            episode_lengths.append(episode_length)
            episode_length = 0

    # Add last episode if it doesn't end with a terminal or timeout
    if episode_length > 0:
        episode_lengths.append(episode_length)

    if episode_lengths:
        return np.max(episode_lengths)
    else:
        return 0


def get_dataset(env):
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
