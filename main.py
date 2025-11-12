"""
Main entry point for QMIX Multi-Agent Coverage System.
"""

import os
import random
import numpy as np
import torch

from config import (
    GRID_SIZE, NUM_AGENTS, SENSOR_RANGE, COMM_RANGE, COVERAGE_THRESHOLD,
    COMPLETION_THRESHOLD, FOV_DEGREES, MAX_TRAINING_EPISODES, EPISODES_PER_MAP_TYPE,
    MAX_STEPS_PER_EPISODE_TRAIN, SAVE_INTERVAL, NUM_EVAL_EPISODES,
    MAX_STEPS_PER_EPISODE_EVAL, QMIX_PARAMS, AGENT_HYPERPARAMS, REWARD_PARAMS,
    DEVICE, MODEL_DIR, RUN_DIR, ANIMATION_DIR, VISUALIZE_TRAINING,
    VISUALIZE_AGENT_LOCAL_TRAINING, VISUALIZE_EVAL_FINAL, VISUALIZE_EVAL_STEPS,
    SAVE_EVAL_ANIMATIONS
)
from environment import MARL_QMIX_Environment
from train import train, evaluate
from utils import ensure_dir


def set_seeds(seed=0):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    """Main function to run training and evaluation."""
    # Set seeds for reproducibility
    set_seeds(0)

    # Setup directories
    ensure_dir(MODEL_DIR)
    ensure_dir(RUN_DIR)
    if SAVE_EVAL_ANIMATIONS or VISUALIZE_AGENT_LOCAL_TRAINING:
        ensure_dir(ANIMATION_DIR)

    # Print configuration
    print(f"Using device: {DEVICE}")
    print(f"QMIX Config: {QMIX_PARAMS}")
    print(f"Agent Config: {AGENT_HYPERPARAMS}")

    # Combine QMIX params and agent params for env init
    env_params = {**QMIX_PARAMS, 'agent_config': AGENT_HYPERPARAMS}

    # Create environment
    env = MARL_QMIX_Environment(
        grid_size=GRID_SIZE,
        num_agents=NUM_AGENTS,
        sensor_range=SENSOR_RANGE,
        comm_range=COMM_RANGE,
        coverage_threshold=COVERAGE_THRESHOLD,
        completion_threshold_perc=COMPLETION_THRESHOLD,
        max_episodes=MAX_TRAINING_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE_TRAIN,
        fov_degrees=FOV_DEGREES,
        device=DEVICE,
        tensorboard_dir=RUN_DIR,
        **env_params,
        **REWARD_PARAMS
    )

    # Train
    print("Starting QMIX training...")
    training_metrics, last_ep_local_maps, last_ep_positions, last_ep_orientations, tracked_agent_id = train(
        env,
        episodes_per_type=EPISODES_PER_MAP_TYPE,
        verbose=True,
        save_interval=SAVE_INTERVAL,
        model_dir=MODEL_DIR
    )
    print("\nQMIX Training complete.")

    # Visualize Local Agent Training Animation
    if VISUALIZE_AGENT_LOCAL_TRAINING and last_ep_local_maps:
        print(f"Generating local map animation for Agent {tracked_agent_id} from final training episode...")
        local_anim_save_path = os.path.join(ANIMATION_DIR, f"training_qmix_agent{tracked_agent_id}_local_steps.mp4")
        env.visualize_agent_local_timesteps(
            agent_id=tracked_agent_id,
            local_map_history=last_ep_local_maps,
            position_history=last_ep_positions,
            orientation_history=last_ep_orientations,
            episode_label="Final Training Episode",
            save_path=local_anim_save_path
        )

    # Evaluate
    print("Starting QMIX evaluation...")
    eval_results = evaluate(
        env,
        num_eval_episodes=NUM_EVAL_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE_EVAL,
        model_dir=MODEL_DIR,
        visualize_each_episode=VISUALIZE_EVAL_FINAL,
        visualize_timesteps=VISUALIZE_EVAL_STEPS,
        save_animations=SAVE_EVAL_ANIMATIONS,
        animation_dir=ANIMATION_DIR
    )

    if eval_results:
        print("\n--- Final QMIX Evaluation Results ---")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.2f}")
        print("-----------------------------")
    else:
        print("Evaluation failed or produced no results.")

    print("\nScript finished.")


if __name__ == "__main__":
    main()
