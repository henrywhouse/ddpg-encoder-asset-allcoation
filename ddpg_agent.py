####################
###  DDPG Agent  ###
####################

## Implements the DDPG agent
## Links replay buffer, noise, actor & critic networks, and the environment
 
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from replay_buffer import Buffer
from stochastic_noise import OrnsteinUhlenbeck
from ddpg_networks import Actor, Critic
from keras.optimizers import Adam
from ddpg_learner import Learner
from tensorflow.nn import softmax
import os

class DDPGAgent(object):
    """DDPG Agent Class
       # Handles the replay buffer, noise, actor & critic networks, learner, and the environment
       # Interface between DDPG algorithm and environment
    """
    def __init__(self, env, actor_layers, critic_layers, lr_actor=0.001, lr_critic=0.001, actor_name='actor', 
                 critic_name='critic', target_actor_name='target_actor', target_critic_name='target_critic', 
                 chkpt_dir='tmp', tau=0.005, gamma=0.99, sigma=0.1, buffer_capacity=10000, batch_size=64,
                 strategy=None): 
        """Instantiate the DDPG agent"""

        self.env = env
        self.state_dims = env.observation_space.shape
        self.n_actions = env.action_space.shape[0]

        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.actor_name = actor_name
        self.critic_name = critic_name
        self.target_actor_name = target_actor_name
        self.target_critic_name = target_critic_name

        self.strategy = strategy

        self.chkpt_dir = chkpt_dir
        os.makedirs(self.chkpt_dir, exist_ok=True)

        self.tau = tau
        self.gamma = gamma
        self.sigma = sigma

        self.actor_optimizer = Adam(learning_rate=self.lr_actor)
        self.critic_optimizer = Adam(learning_rate=self.lr_critic)
        
        ## Initialize the networks
        self.actor = self.build_actor(name=self.actor_name, layers=actor_layers)
        self.critic = self.build_critic(name=self.critic_name, layers=critic_layers)
        self.target_actor = self.build_actor(name=self.target_actor_name, layers=actor_layers)
        self.target_critic = self.build_critic(name=self.target_critic_name, layers=critic_layers)

        ## Implement GPU distribution (if applicable)
        if self.strategy is not None:
            with self.strategy.scope():
                self.actor = self.actor
                self.critic = self.critic
                self.target_actor = self.target_actor
                self.target_critic = self.target_critic
                self.actor_optimizer = self.actor_optimizer
                self.critic_optimizer = self.critic_optimizer

        # Make the target networks have the same weights as the initial networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        ## Replay buffer for experience replay
        self.buffer = Buffer(self.state_dims, self.n_actions, 
                             self.buffer_capacity, self.batch_size)

        ## Noise for exploration
        self.noise = OrnsteinUhlenbeck(mu=np.zeros(self.n_actions,), sigma=self.sigma)

        ## Learner for network updates
        self.learner = Learner(lr_actor=self.lr_actor, lr_critic=self.lr_critic, state_dims=self.state_dims, 
                               n_actions=self.n_actions, actor=self.actor, critic=self.critic, 
                               target_actor=self.target_actor, target_critic=self.target_critic, 
                               actor_optimizer=self.actor_optimizer, critic_optimizer=self.critic_optimizer, 
                               buffer=self.buffer, tau=self.tau, gamma=self.gamma)


    def build_actor(self, name, layers):
        """Instantiate actor network"""

        actor =  Actor(self.state_dims, self.n_actions, layers, name=name, chkpt_dir=self.chkpt_dir)
        actor.call(np.zeros((1,) + self.state_dims))  # Dummy call to allow for loading weights
        return actor
    

    def build_critic(self, name, layers):
        """Instantiate critic network"""

        critic = Critic(layers, name=name, chkpt_dir=self.chkpt_dir)
        critic.call(np.zeros((1,) + self.state_dims), np.zeros((1,) + (self.n_actions,)))  # Dummy call to allow for loading weights
        return critic


    def get_action(self, state, use_noise=True):
        """Returns action for given state (inference from actor network)"""

        state = np.expand_dims(state, axis=0)
        action = self.actor.call(state)
        if use_noise:
            noise = self.noise()
            action += noise
            action = softmax(action)
        else:
            action = softmax(action)

        return action.numpy()


    def update_target_networks(self):
        """Updates parameters of target networks according to soft update rule given by tau"""
        
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()

        # Update target networks using Polyak averaging
        for i, actor_weight in enumerate(actor_weights):
            target_actor_weights[i] = self.tau * actor_weight + (1 - self.tau) * target_actor_weights[i]

        for i, critic_weight in enumerate(critic_weights):
            target_critic_weights[i] = self.tau * critic_weight + (1 - self.tau) * target_critic_weights[i]

        self.target_actor.set_weights(target_actor_weights)
        self.target_critic.set_weights(target_critic_weights)

    def learn(self):
        """Wrapper method for updating network parameters using experiences from replay buffer"""

        if self.buffer.get_memory_count() < self.buffer.batch_size:
            return
        self.learner.learn(self.buffer)
        self.update_target_networks()


    def save_models(self):
        """Save network parameters to disk"""

        print('..... Saving models .....')
        self.actor.save_weights(self.actor.chkpt_file)
        self.critic.save_weights(self.critic.chkpt_file)
        self.target_actor.save_weights(self.target_actor.chkpt_file)
        self.target_critic.save_weights(self.target_critic.chkpt_file)
        print('..... Models saved successfully! .....')


    def load_models(self):
        """Load network parameters from disk"""
        
        print('..... Loading Models .....')
        self.actor.load_weights(self.actor.chkpt_file)
        self.critic.load_weights(self.critic.chkpt_file)
        self.target_actor.load_weights(self.target_actor.chkpt_file)
        self.target_critic.load_weights(self.target_critic.chkpt_file)
        print('..... Models loaded successfully! .....')