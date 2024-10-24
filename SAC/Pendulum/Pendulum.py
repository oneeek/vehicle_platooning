import Agent

import gym
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



class Pendulum:
    def __init__(self):
        self.env = gym.make('Pendulum-v1')
        self.env_rend = gym.make('Pendulum-v1', render_mode='human')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.agent = Agent.SACagent(self.state_dim, self.action_dim, self.action_bound)

    def save_weights(self):
        cur_dir = os.getcwd()
        ckpt_dir = 'checkpoint'
        dr = os.path.join(cur_dir, ckpt_dir)
        os.makedirs(dr, exist_ok=True)

        file_name_a = 'actor_w'
        file_name_c = 'critic_w'
        file_path_a = os.path.join(dr, file_name_a)
        file_path_c = os.path.join(dr, file_name_c)

        self.agent.actor.save_weights(file_path_a)
        self.agent.critic.save_weights(file_path_c)
        return file_path_a, file_path_c

    def load_weights(self, file_path_a, file_path_c):
        self.agent.actor.load_weights(file_path_a)
        self.agent.critic.load_weights(file_path_c)

    def train(self, max_episode):
        Reward = []
        self.agent.update_target_network(1.0)

        for ep in range(int(max_episode)):
            time, episode_reward, done = 0, 0, False
            state, _ = self.env.reset()

            while not done:
                action = self.agent.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
                action = np.clip(action, -self.agent.action_bound, self.agent.action_bound)
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                self.agent.buffer.add_buffer(state, action, reward, next_state, done)

                if self.agent.buffer.buffer_count() > 1000:
                    states, actions, rewards, next_states, dones = self.agent.buffer.sample_batch(self.agent.BATCH_SIZE)
                    next_mu, next_std = self.agent.actor(tf.convert_to_tensor(next_states, dtype=tf.float32))
                    next_actions, next_log_pdf = self.agent.actor.sample_normal(next_mu, next_std)

                    target_qs = self.agent.target_critic([next_states, next_actions])
                    target_qi = target_qs - self.agent.ALPHA * next_log_pdf
                    y_i = self.agent.q_target(rewards, target_qi.numpy(), dones)

                    self.agent.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                    tf.convert_to_tensor(actions, dtype=tf.float32),
                                    tf.convert_to_tensor(y_i, dtype=tf.float32))
                    self.agent.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32))
                    self.agent.update_target_network(self.agent.TAU)

                state = next_state
                episode_reward += reward
                time += 1

            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)
            Reward.append(episode_reward)

        plt.plot(Reward)
        plt.show()
        return Reward

    def test(self):
        env = self.env_rend
        episode_reward, time, done = 0, 0, False
        state, _ = env.reset()

        while not done:
            action = self.agent.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            time += 1
            episode_reward += reward

            print('Time: ', time, 'Reward: ', reward, 'EReward: ', episode_reward)
        env.close()
