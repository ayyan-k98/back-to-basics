"""
Configuration parameters for QMIX multi-agent coverage system.
"""


# Grid and Agent Configuration
GRID_SIZE = 20
NUM_AGENTS = 4
SENSOR_RANGE = 5
COMM_RANGE = 8.0
COVERAGE_THRESHOLD = 0.80
COMPLETION_THRESHOLD = 95.0
FOV_DEGREES = 120.0

# Training parameters
MAX_TRAINING_EPISODES = 150
EPISODES_PER_MAP_TYPE = 50
MAX_STEPS_PER_EPISODE_TRAIN = 100
SAVE_INTERVAL = 50

# Evaluation parameters
NUM_EVAL_EPISODES = 10
MAX_STEPS_PER_EPISODE_EVAL = 150

# QMIX / Training Hyperparameters
QMIX_PARAMS = {
    'memory_capacity': 50000,
    'batch_size': 128,
    'gamma': 0.99,
    'lr': 0.0005,  # Learning rate for shared optimizer
    'mixer_embed_dim': 64,
    'target_update_freq': 200,  # Steps between target updates
    'use_soft_update': True,
    'soft_update_tau': 0.005,
}

# Agent Hyperparameters (Epsilon handled here)
AGENT_HYPERPARAMS = {
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.999,
    'use_dueling': True,  # Use Dueling DQN for agent networks
}

# Reward parameters
REWARD_PARAMS = {
    'gamma_coverage': 20.0,
    'step_penalty': -0.01,
    'orientation_cost_factor': 0.02,
    'invalid_move_penalty': -0.5,
}

# Device Configuration
DEVICE = "cuda"  # or "cpu"

# Directory Configuration
MODEL_DIR = "./models_coverage_qmix_final"
RUN_DIR = "./runs/coverage_qmix_experiment_final"
ANIMATION_DIR = "./eval_animations_qmix_final"

# Visualization Flags
VISUALIZE_TRAINING = True
VISUALIZE_AGENT_LOCAL_TRAINING = True
VISUALIZE_EVAL_FINAL = True
VISUALIZE_EVAL_STEPS = True
SAVE_EVAL_ANIMATIONS = True
