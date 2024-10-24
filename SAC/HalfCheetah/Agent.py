import ReplayBuffer
import Actor
import Critic

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input



class SACagent():
    def __init__(self, state_dim, action_dim, action_bound):
        ## hyperparameters
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 40000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001
        self.ALPHA = 0.5

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        ##########
        self.actor = Actor.Actor(self.action_dim, self.action_bound)
        self.actor.build(input_shape=(None, self.state_dim))

        self.critic = Critic.Critic()
        self.target_critic = Critic.Critic()
        state_in = Input((self.state_dim,))
        action_in = Input((self.action_dim,))
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])

        self.actor.summary()
        self.critic.summary()

        ##########
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        self.buffer = ReplayBuffer.ReplayBuffer(self.BUFFER_SIZE)
        self.save_epi_reward = []

    def get_action(self, state):
        mu, std = self.actor(state)
        action, _ = self.actor.sample_normal(mu, std)
        return action.numpy()[0]

    def update_target_network(self, TAU):
        phi = self.critic.get_weights()
        target_phi = self.target_critic.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_critic.set_weights(target_phi)

    def critic_learn(self, states, actions, q_targets):
        with tf.GradientTape() as tape:
            q = self.critic([states, actions], training=True)
            loss = tf.reduce_mean(tf.square(q-q_targets))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            mu, std = self.actor(states, training=True)
            actions, log_pdfs = self.actor.sample_normal(mu, std)
            log_pdfs = tf.squeeze(log_pdfs, 1)
            soft_q = self.critic([states, actions])
            loss = tf.reduce_mean(self.ALPHA * log_pdfs - soft_q)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

    def q_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k