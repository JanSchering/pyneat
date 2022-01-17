from __future__ import annotations
import gym
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import time
from tsboardmod import ModifiedTensorBoard
import params
from collections import deque
from ..innovationcounter import init
from ..tfgenome import TFGenome
from ..tfpopulation import TFPopulation
from typing import Any, Callable, List, Tuple
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class NEATQAgent:
    """
    Implements a combination of NEAT and Deep Q Learning.
    """

    def __init__(self, obs_dim: int, act_dim: int):
        self.population = TFPopulation(
            obs_dim, act_dim, self.eval_genome, 50)

        self.replay_memory = deque(params.MAX_MEMORY_SIZE)

        self.model = None
        self.prev_model = None

        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}-{}".format(params.NQN_MODEL_NAME, int(time.time())))

        self.optimizer = Adam(learning_rate=5e-4)
        self.loss_function = MeanSquaredError()

        self.innovationcounter = init()

        self.best_fitness = float("-inf")
        self.mean_score = float("-inf")

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
        Executes the training process for the network. The network is trained on 1000 batches of random samples.
        """
        for i in range(1000):
            # Get a minibatch of random samples from memory replay table
            minibatch = random.sample(
                self.replay_memory, params.MINIBATCH_SIZE)

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

    def evolve(self):
        new_structure = False
        self.population.natural_selection()
        # Check if a new predominant structure evolved
        if self.population.best_score > self.best_fitness:
            new_structure = True
            self.prev_model = self.model
            self.model = self.population.best_genome.create_model()

        self.train()
        # evaluate new network performance on 100 games
        scores = []
        for i in range(100):
            global env
            # env.render(mode=None)
            observation = env.reset()
            score = 0.0
            done = False
            while not done:
                outputs = self.get_qs(observation).numpy()
                # Move the game forward
                observation, reward, done, info = env.step(np.argmax(outputs))
                score += reward
            scores.append(score)
        mean_score = np.mean(scores)
        if mean_score > self.mean_score:
            self.mean_score = mean_score
        # New structure has not empirically proven better than old, discard
        elif new_structure:
            self.model = self.prev_model
        return self.mean_score

    def eval_genome(self, genome: TFGenome) -> float:
        """
        The fitness criterion that will be used to evaluate the Genomes 
        in the natural selection process
        """
        fitnesses = []
        for runs in range(params.RUNS_PER_NET):
            global env
            # env.render(mode=None)
            observation = env.reset()
            fitness = 0.0
            done = False
            while not done:
                outputs = genome.forward(observation)
                # Move the game forward
                observation, reward, done, info = env.step(np.argmax(outputs))
                fitness += reward
            fitnesses.append(fitness)
        return (np.mean(fitnesses), np.max(fitnesses), np.min(fitnesses))


env = gym.make("LunarLander-v2")
env.seed(0)
agent = NEATQAgent(env.observation_space.shape[0], env.action_space.n)

for gen in range(params.MAX_GENERATIONS):
    # The first 200 Gens, we purely evolve on an evolutionary timescale
    # Allows to build some structure and to collect memory data
    if gen < 200:
        agent.population.natural_selection()
    else:
        mean_score, max_score, min_score = agent.evolve()
        print(
            f"GEN: {gen+1}\nMEAN SCORE: {mean_score}\nMAX SCORE: {max_score}\nMIN SCORE: {min_score}")
