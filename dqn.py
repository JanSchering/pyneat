# %%
from typing import Tuple
from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from collections import deque
from tsboardmod import ModifiedTensorBoard
import time
import numpy as np
import random
import os
from tqdm import tqdm
import gym
import tensorflow as tf
import params
import argparse


class DQNAgent:
    def create_q_model(self, obs_dim: int, act_dim: int) -> Sequential:
        """Creates a TF Sequential NN that approximates a mapping
        from state to expected value of the possible actions.

        Args:
            obs_dim (int): Dimensionality of the state space
            act_dim (int): Dimensionality of the action space

        Returns:
            Sequential: the model
        """
        model = Sequential()
        model.add(InputLayer(input_shape=(obs_dim,)))

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(act_dim))
        return model

    def create_estimator(self, dim: int) -> Sequential:
        """Creates a TF Sequential NN that tries to denoise the state space.

        Args:
            dim (int): DImensionality of the state space

        Returns:
            Sequential: the model.
        """
        model = Sequential()
        model.add(InputLayer(input_shape=(dim,)))

        model.add(Dense(32))
        model.add(Activation('relu'))

        model.add(Dense(dim))
        return model

    def __init__(self, obs_dim, act_dim):
        # Main model
        self.model = self.create_q_model(obs_dim, act_dim)
        # Target network
        self.target_model = self.create_q_model(obs_dim, act_dim)
        self.target_model.set_weights(self.model.get_weights())
        # create state estimator networkk
        self.estimator = self.create_estimator(obs_dim)
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=params.MAX_MEMORY_SIZE)
        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}-{}".format(params.MODEL_NAME, int(time.time())))
        # Used to count when to update target network with main network's weights
        self.update_counter = 0

        self.optimizer = Adam(learning_rate=5e-4)
        self.loss_function = MeanSquaredError()

    def update_replay_memory(self, transition: Tuple[np.ndarray, int, np.ndarray, bool]) -> None:
        """Adds step's data to a memory replay array

        Args:
            transition: (state, action, reward, next_state, done)
        """
        self.replay_memory.append(transition)

    def get_qs(self, state: np.ndarray) -> np.ndarray:
        """Queries the main Q network for Q values of the given state. 

        Args:
            state (np.ndarray): The state to query the Q values for

        Returns:
            np.ndarray: The Q values
        """
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state: bool, step: int) -> None:
        """Executes the training process for the network

        Args:
            terminal_state (bool): Whether the current episode has reached a terminal state
            step (int): The current step of the episode
        """
        if len(self.replay_memory) < params.MINIBATCH_SIZE:
            return
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(
            self.replay_memory, params.MINIBATCH_SIZE)
        # First collect all current states from batch
        current_states = np.array([transition[0]
                                   for transition in minibatch])
        # Collect all next_states from batch
        new_current_states = np.array(
            [transition[3] for transition in minibatch])
        # Apply a step of gradient descent on the weights of the state estimator
        with tf.GradientTape() as tape:
            # predict next states using the estimator
            preds = self.estimator(np.array(current_states))
            # Loss value for this batch.
            loss_value = self.loss_function(
                np.array(new_current_states), preds)
        # Get gradients of loss wrt the weights.
        gradients = tape.gradient(loss_value, self.estimator.trainable_weights)
        # Update the weights of the model.
        self.optimizer.apply_gradients(
            zip(gradients, self.estimator.trainable_weights))

        # If counter hits target, update the policy networks
        self.update_counter = (self.update_counter + 1) % params.UPDATE_EVERY
        if self.update_counter == 0 and len(self.replay_memory) > params.MIN_MEMORY_SIZE:
            # Get an estimate of the current state
            state_estimate = self.estimator.predict(current_states)
            # Query main model for Q values of the current states
            current_qs_list = self.model.predict(state_estimate)
            # Get future states from minibatch, then query target policy model for Q values
            future_qs_list = self.target_model.predict(state_estimate)

            X = []
            y = []

            # Now we need to enumerate our batches
            for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

                # If not a terminal state, get new q from future states, otherwise set it to 0
                # almost like with Q Learning, but we use just part of equation here
                if not done:
                    max_future_q = np.max(future_qs_list[index])
                    new_q = reward + params.DISCOUNT * max_future_q
                else:
                    new_q = reward

                # Update Q value for given state
                current_qs = current_qs_list[index]
                current_qs[action] = new_q

                # And append to our training data
                X.append(current_state)
                y.append(current_qs)

            # Apply a step of gradient descent on the created dataset
            with tf.GradientTape() as tape:
                # Forward pass.
                logits = self.model(np.array(X))
                # Loss value for this batch.
                loss_value = self.loss_function(np.array(y), logits)
            # Get gradients of loss wrt the weights.
            gradients = tape.gradient(loss_value, self.model.trainable_weights)
            # Update the weights of the model.
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_weights))

            # θ_target = τ*θ_local + (1 - τ)*θ_target
            for local_var, target_var in zip(self.model.trainable_variables, self.target_model.trainable_variables):
                target_var.assign(params.TAU*local_var +
                                  (1-params.TAU)*target_var)
                self.target_update_counter = 0


def run_dqn(agent: DQNAgent, env: gym.Env, noisy=False, no_pos=False, no_vel=False) -> None:
    """Implementation of the Deep Q Learning algorithm

    Args:
        agent (DQNAgent): Agent to train
        env (gym.Env): the environment to train the agent on.
        noisy (bool, optional): Defaults to False.
        no_pos (bool, optional): Defaults to False.
        no_vel (bool, optional): Defaults to False.
    """
    epsilon = 1  # not a constant, going to be decayed
    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))

    # We will average the rewards over a window of the last 100
    ep_rewards = deque(maxlen=100)

    # Iterate over episodes
    for episode in tqdm(range(1, params.EPISODES + 1), ascii=True, unit='episodes'):

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()
        # If partially observable, we remove the position observations from the state vector
        if no_pos:
            current_state = current_state[2:]
        elif no_vel:
            del current_state[2:4]

        if noisy:
            current_state += np.random.normal(0, 0.1, len(current_state))

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
            # If partially observable we also have to remove the position info from the new state
            if no_pos:
                new_state = new_state[2:]
            elif no_vel:
                del new_state[2:4]

            if noisy:
                new_state += np.random.normal(0, 0.1, len(new_state))

            # count reward
            episode_reward += reward

            if params.SHOW_PREVIEW and not episode % params.AGGREGATE_STATS_EVERY:
                env.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory(
                (current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % params.AGGREGATE_STATS_EVERY or episode == 1:
            mean_reward = np.mean(ep_rewards)
            min_reward = np.min(ep_rewards)
            max_reward = np.max(ep_rewards)
            agent.tensorboard.update_stats(
                reward_mean=mean_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if mean_reward >= params.MEAN_REWARD:
                agent.model.save(
                    f'models/{params.MODEL_NAME}__{max_reward:_>7.2f}max_{mean_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > params.MIN_EPSILON:
            epsilon *= params.EPSILON_DECAY
            epsilon = max(params.MIN_EPSILON, epsilon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial-no-pos', dest='no_pos',
                        default=False, action='store_true')
    parser.add_argument('--partial-no-vel', dest='no_vel',
                        default=False, action='store_true')
    parser.add_argument('--noisy', dest='noisy',
                        default=False, action='store_true')
    args = parser.parse_args()
    env = gym.make("LunarLander-v2")
    # Set random seeds for comparable results between different trainng runs
    env.seed(1)
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    if args.noisy:
        print("using noisy state measurements")

    # Adjust network architecture if partially observable
    if args.no_pos or args.no_vel:
        agent = DQNAgent(
            env.observation_space.shape[0]-2, env.action_space.n)
    else:
        agent = DQNAgent(
            env.observation_space.shape[0], env.action_space.n)

    run_dqn(agent, env, args.noisy, args.no_pos, args.no_vel)

# %%
