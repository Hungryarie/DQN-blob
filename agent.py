
import keras.backend.tensorflow_backend as backend

import tensorflow as tf

from collections import deque
from model import ModifiedTensorBoard
import time
import numpy as np
import random

from constants import *


# Agent class
class DQNAgent:

    def __init__(self, Model):
        # Main model: gets trained every step => .fit()
        self.model = Model.model  # self.create_model(input_shape=(10,10,3),output_num=9)

        # Target network: this is what will get predict against every step => .predict()
        self.target_model = Model.target_model  # self.create_model(input_shape=(10, 10, 3), output_num=9)
        self.target_model.set_weights(self.model.get_weights())

        print(f"model:{self.model}, target_model:{self.target_model}")

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0


    
    def update_replay_memory(self, transition):
        # Adds step's data to a memory replay array
        # (observation space, action, reward, new observation space, done)
        self.replay_memory.append(transition)

    
    def train(self, terminal_state, step):
        # Trains main network every step during episode

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        # 2do: normilize function instead of /255
        current_states = np.array([transition[0] for transition in minibatch])/255  # transition:(observation space, action, reward, new observation space, done)
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        # 2do: normilize function instead of /255
        new_current_states = np.array([transition[3] for transition in minibatch])/255  # transition:(observation space, action, reward, new observation space, done)
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward  # no further max_future_q possible, because done=True

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q  # replace the current_q for the self.model.predict() action by the new_q from the self.target_model.predict() action q-value

            # And append to our training data
            X.append(current_state) # features = state
            y.append(current_qs)    # labels = actions

        # Fit on all samples as one batch, log only on terminal state
        # 2do: normilize function instead of /255
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0  # reset

    def get_qs(self, state):
            # Queries main network for Q values given current observation space (environment state)
        # So this is just doing a .predict(). We do the reshape because TensorFlow wants that exact explicit way to shape. The -1 just means a variable amount of this data will/could be fed through.
        # divided by 255 is to normalize is.
        # 2do: write own normalize method
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]