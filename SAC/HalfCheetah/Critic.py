from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, concatenate



class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.x1 = Dense(32, activation='relu')
        self.a1 = Dense(32, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.q = Dense(1, activation='linear')

    def call(self, state_action):
        state = state_action[0]
        action = state_action[1]

        x = self.x1(state)
        a = self.a1(action)
        h = concatenate([x, a], axis=-1)
        x = self.h2(h)
        x = self.h3(x)
        q = self.q(x)
        return q