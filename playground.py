import diffuser.utils as utils
import gym
import d4rl 
import h5py
import numpy as np



# # import gym
# import d4rl_pybullet

# # dataset will be automatically downloaded into ~/.d4rl/datasets
# env = gym.make('hopper-bullet-mixed-v0')

# # interaction with its environment
# env.reset()
# env.step(env.action_space.sample())

# # access to the dataset
# dataset = env.get_dataset()
# # print(dataset[0])
# print(len(dataset['observations'])) # observation data in N x dim_observation
# # dataset['actions'] # action data in N x dim_action
# # dataset['rewards'] # reward data in N x 1
# # dataset['terminals'] # terminal flags in N x 1


# for name in file:
#     print(name)
#     group = file[name]
#     print(group)


# def load_hdf5_to_dict(filename):
#     """
#     Load the contents of an HDF5 file into a dictionary.

#     :param filename: Path to the HDF5 file
#     :return: Dictionary with dataset names as keys and dataset contents as values
#     """
#     data_dict = {}
#     with h5py.File(filename, 'r') as file:
#         for name in file:
#             data_dict[name] = file[name][()]
#     return data_dict

# # Usage example
# filename = '/home/bearhaon/diffuser/data_hold/lukas.h5'
# data_dict = load_hdf5_to_dict(filename)

# actions = data_dict['actions']
# observations = data_dict['observations']
# rewards = data_dict['rewards']
# terminals = data_dict['terminals']
# timeouts = data_dict['timeouts']


# print(actions[-1])
# print(observations[-1])
# print(rewards[-1])
# print(terminals[-1])
# print(timeouts[-1])

# def max_length_and_index(arrays):
#     max_len = 0
#     max_index = -1

#     for i, arr in enumerate(arrays):
#         if len(arr) >= max_len:
#             max_len = len(arr)
#             max_index = i

#     return max_len, max_index


# longest_length = max_length_and_index(observations[1:])

# print("The longest array has length:", longest_length)

# print(observations[-1][0:10])

# print(rewards)


# def percentage_zeros(array_of_arrays):
#     # Convert to a NumPy array for efficient computations
#     np_array = np.array(array_of_arrays)
    
#     # Count the number of zero elements
#     num_zeros = np.count_nonzero(np_array == 0)
    
#     # Calculate the total number of elements in the array
#     total_elements = np_array.size
    
#     # Calculate the percentage of zero elements
#     percentage = (num_zeros / total_elements) * 100
    
#     return percentage


# zero_percentage = percentage_zeros(observations)
# print(f"Percentage of zeros: {zero_percentage:.2f}%")



def print_structure(hdf5_file, indent=0):
    """
    Recursively prints the structure of the HDF5 file.
    """
    for key in hdf5_file.keys():
        item = hdf5_file[key]
        print('    ' * indent + key)
        if isinstance(item, h5py.Dataset):
            print('    ' * (indent + 1) + f"Dataset (shape: {item.shape}, dtype: {item.dtype})")
        elif isinstance(item, h5py.Group):
            print_structure(item, indent + 1)

file_path = '/home/bearhaon/diffuser/data_hold/lukas.h5'

with h5py.File(file_path, 'r') as file:
    print(f"Structure of HDF5 file '{file_path}':")
    print_structure(file)
