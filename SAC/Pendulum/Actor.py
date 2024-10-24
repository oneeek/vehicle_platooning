import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda



class Actor(Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-2, 1.0]
        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.mu = Dense(action_dim, activation='tanh')
        self.std = Dense(action_dim, activation='softplus')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        mu = self.mu(x)
        std = self.std(x)

        mu = Lambda(lambda x: x*self.action_bound)(mu)
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        return mu, std

    def sample_normal(self, mu, std):
        normal_prob = tfp.distributions.Normal(mu, std)
        action = normal_prob.sample()
        action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
        log_pdf = normal_prob.log_prob(action)
        log_pdf = tf.reduce_sum(log_pdf, 1, keepdims=True)
        return action, log_pdf