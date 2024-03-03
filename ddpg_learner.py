#####################
###  DDPG Learner ###
#####################

## Implements learning functionality of DDPG actor & critic network

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

class Learner(object):
    """DDPG Learner class that handles backpropagation of the actor and critic networks and updating the target networks
    """
    def __init__(self, lr_actor, lr_critic, state_dims, n_actions, actor, critic, target_actor, target_critic, 
                 actor_optimizer, critic_optimizer, buffer, tau, gamma=0.99, strategy=None):
        """Initialize the learner"""

        self.state_dims = state_dims
        self.n_actions = n_actions

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer.learning_rate = lr_actor
        self.critic_optimizer.learning_rate = lr_critic

        self.gamma = gamma
        self.tau = tau

        self.buffer = buffer
        self.batch_size = buffer.batch_size

        self.strategy = strategy


    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        """Update parameters of the actor and critic networks during training"""

        ### CRITIC ###
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            ## Compute the target Q value
            y = reward_batch + self.gamma * self.target_critic.call(
                next_state_batch, target_actions)
            
            critic_value = self.critic.call(state_batch, action_batch)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        ## Compute the gradients of the critic network
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
       
       ## Update the critic network parameters
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables))

        ### ACTOR ###
        with tf.GradientTape() as tape:
            actions = self.actor.call(state_batch)
            critic_value = self.critic.call(state_batch, actions)
            actor_loss = -tf.math.reduce_mean(critic_value)  # Used `-value` as we want to maximize 
                                                             # the value given by the critic for our actions

        ## Compute the gradients of the actor network        
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)

        ## Update the actor network parameters
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables))


    def learn(self, buffer):
        """Complete batch updates of the actor and critic networks"""

        state_batch, action_batch, reward_batch, next_state_batch = buffer.get_batch_tensors()
        # Reshape the state_batch and action_batch if needed
        state_batch = tf.reshape(state_batch, (self.batch_size, *self.state_dims))
        action_batch = tf.reshape(action_batch, (self.batch_size, self.n_actions))
        if self.strategy is not None:
            self.strategy.run(self.update, args=(state_batch, action_batch, reward_batch, next_state_batch))
        else:
            self.update(state_batch, action_batch, reward_batch, next_state_batch)