#######################
###  Replay Buffer  ###
#######################

## Replay Buffer class for experience replay in DRL algorithms (e.g., DDPG)
## Permit offline learning

import numpy as np
import tensorflow as tf

class Buffer(object):
    """Replay Buffer class for experiential replay in the DDPG algorithm
    """
    def __init__(self, state_dims, n_actions, buffer_capacity=100000, batch_size=64):
        """Initializes the replay buffer"""

        self.state_dims = state_dims
        self.n_actions = n_actions
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Number of times record() has been called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, *self.state_dims), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, self.n_actions), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, *self.state_dims), dtype=np.float32)
        self.done_buffer = np.zeros((self.buffer_capacity, 1), dtype=np)


    def record(self, state, action, reward, state_, done):
        """Records a new experience into the buffer"""

        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = np.squeeze(state_)

        self.buffer_counter += 1


    def _get_batch_indices(self):
        """Returns indices of random sample of experiences in the buffer"""

        record_range = self.get_memory_count() 
        batch_indices = np.random.choice(record_range, self.batch_size)
        return batch_indices
    

    def get_memory_count(self):
        """Counts the number of experiences in the buffer"""

        return min(self.buffer_counter, self.buffer_capacity)
    

    def is_full(self):
        """Checks if replay buffer is full"""

        return self.buffer_capacity == self.get_memory_count()


    def get_batch_tensors(self):
        """Returns experiences batch tensors"""
        
        batch_indices = self._get_batch_indices()
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        return state_batch, action_batch, reward_batch, next_state_batch
