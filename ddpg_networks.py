import keras
from keras.layers import InputLayer, Dense, Flatten
from keras.activations import relu, softmax, tanh, linear
from keras.initializers import glorot_uniform
import tensorflow as tf
import os
tf.get_logger().setLevel('ERROR')

## Actor network
class Actor(keras.Model):
    """Actor Network for the DDPG algorithm
    """
    def __init__(self, state_dims, num_actions, layers, name='actor', chkpt_dir='tmp/dir'):
        """Initialize actor network"""

        super(Actor, self).__init__()

        self.state_dims = state_dims   
        self.num_actions = num_actions

        self.input_layer = InputLayer(input_shape=self.state_dims)
        self.custom_layers = layers
        self.output_layer = Dense(num_actions, activation=linear, kernel_initializer=glorot_uniform())

        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, self.model_name + '_ddpg.h5')

        self.built=True

    def call(self, inputs):
        """Make inference on actor network and return action given state"""

        x = self.input_layer(inputs)
        for layer in self.custom_layers:
            x = layer(x)
        mu = self.output_layer(x)   
        return mu
    

## Critic Network
class Critic(keras.Model):
    """Critic Network for the DDPG algorithm
    """
    def __init__(self, layers, name='critic', chkpt_dir='tmp/dir'):
        """Initialize critic network"""

        super(Critic, self).__init__()

        self.state_flatten = Flatten()
        self.action_flatten = Flatten()
        self.custom_layers = layers
        self.fc3 = Dense(1, kernel_initializer=glorot_uniform(), activation=relu) 

        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, self.model_name + '_ddpg.h5')

        self.built=True

    def call(self, state, action):
        """Make inference on critic network and return q-value of state action pair"""
        
        state = self.state_flatten(state)
        action = self.action_flatten(action)
        x = tf.concat([state, action], axis=-1)
        for layer in self.custom_layers:
            x = layer(x)
        q = self.fc3(x)
        return q
