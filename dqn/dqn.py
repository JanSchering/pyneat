# %%
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, InputLayer
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque
from tsboardmod import ModifiedTensorBoard
import time
import numpy as np
import random
import os
from tqdm import tqdm
import gym
import tensorflow as tf

DISCOUNT = 0.99
MIN_MEMORY_SIZE = 1_000
MAX_MEMORY_SIZE = int(1e5)
MODEL_NAME = "DQN"
MINIBATCH_SIZE = 64
UPDATE_EVERY = 4
TAU = 1e-3  # Interpolation Param

# Environment settings
EPISODES = 5_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

MEAN_REWARD = 200


class DQNAgent:
    def create_model(self, obs_dim, act_dim):
        model = Sequential()
        model.add(InputLayer(input_shape=(obs_dim,)))

        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(act_dim, activation='tanh'))
        model.compile(loss="mse", optimizer=Adam(
            lr=5e-4), metrics=['accuracy'])
        return model

    def __init__(self, obs_dim, act_dim):
        # Main model
        self.model = self.create_model(obs_dim, act_dim)
        # Target network
        self.target_model = self.create_model(obs_dim, act_dim)
        self.target_model.set_weights(self.model.get_weights())
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=MAX_MEMORY_SIZE)
        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        # Used to count when to update target network with main network's weights
        self.update_counter = 0

    def update_replay_memory(self, transition):
        """
        Adds step's data to a memory replay array
        Transition is a Tuple(state, action, reward, next_state, done)
        """
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state, step):
        """
        Executes the training process for the network
        """
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0]
                                  for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array(
            [transition[3] for transition in minibatch])
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
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        self.update_counter = (self.update_counter + 1) % UPDATE_EVERY
        if self.update_counter == 0:
            # Fit on all samples as one batch, log only on terminal state
            self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0,
                           shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

            # θ_target = τ*θ_local + (1 - τ)*θ_target
            for local_var, target_var in zip(self.model.trainable_variables, self.target_model.trainable_variables):
                target_var.assign(TAU*local_var + (1-TAU)*target_var)
                self.target_update_counter = 0


# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

env = gym.make("LunarLander-v2")
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
# We will average the rewards over a window of the last 100
ep_rewards = deque(maxlen=100)

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        # count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory(
            (current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        mean_reward = np.mean(ep_rewards)
        min_reward = np.min(ep_rewards)
        max_reward = np.max(ep_rewards)
        agent.tensorboard.update_stats(
            reward_mean=mean_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if mean_reward >= MEAN_REWARD:
            agent.model.save(
                f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{mean_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

# %%
