

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
AGGREGATE_STATS_EVERY = 100  # episodes
SHOW_PREVIEW = False

MEAN_REWARD = 200

# NEATQN SPECIFIC
MAX_GENERATIONS = 500

RUNS_PER_NET = 5
