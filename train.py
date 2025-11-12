"""
Training and evaluation functions for QMIX multi-agent coverage system.
"""

import os
import time
import numpy as np
from utils import ensure_dir


def train(env, episodes_per_type=25, verbose=True, save_interval=50, model_dir="./qmix_models"):
    """Train QMIX agents, cycling through map types (DQN Style)."""
    print("\n--- Starting QMIX Training ---")
    ensure_dir(model_dir)
    if env.writer:
        ensure_dir(env.writer.log_dir)
    if episodes_per_type <= 0:
        episodes_per_type = 1
    num_map_types = len(env.training_map_order)
    if num_map_types == 0:
        raise ValueError("Training map order is empty.")

    initial_map_idx = 0
    env.current_training_map_index = initial_map_idx
    states = env.reset(full_reset=True, mode='train')
    if not env.agents:
        raise RuntimeError("Agents not created after initial reset.")
    print(f"Training start. Initial map type: {env.training_map_order[initial_map_idx]}")

    total_steps_across_episodes = 0
    last_ep_local_map_history, last_ep_position_history, last_ep_orientation_history = [], [], []
    agent_id_to_track = 0

    for episode in range(env.max_episodes):
        map_type_index = (episode // episodes_per_type) % num_map_types
        current_map_type = env.training_map_order[map_type_index]
        if env.current_training_map_index != map_type_index or episode == 0:
            if verbose and episode > 0:
                print(f"\n--- Switching to Map Type: {current_map_type} (Episode {episode}) ---")
            env.current_training_map_index = map_type_index
            states = env.reset(full_reset=False, mode='train')
        else:
            states = env.reset(full_reset=False, mode='train')

        episode_trajectories = {i: [env.agents[i].state.position] for i in range(env.num_agents)}
        episode_rewards_sum = {i: 0 for i in range(env.num_agents)}
        episode_steps = 0
        episode_done = False
        ep_start_time = time.time()

        is_last_episode = (episode == env.max_episodes - 1)
        if is_last_episode:
            print(f"Tracking agent {agent_id_to_track}'s local map for final training episode {episode}")
            last_ep_local_map_history, last_ep_position_history, last_ep_orientation_history = [], [], []
            if agent_id_to_track < len(env.agents):
                tracked_agent = env.agents[agent_id_to_track]
                last_ep_local_map_history.append(tracked_agent.local_map.copy())
                last_ep_position_history.append(tracked_agent.state.position)
                last_ep_orientation_history.append(tracked_agent.state.orientation)

        if verbose and episode % 10 == 0:
            print(f"--- Ep {episode}/{env.max_episodes} (Map: {current_map_type}, Eps: {env.agents[0].epsilon:.3f}) ---")

        for step in range(env.max_steps_per_episode):
            actions = {}
            for agent_id, agent in enumerate(env.agents):
                if agent_id in states:
                    valid_actions = env.get_valid_actions(agent_id)
                    action = agent.select_action(states[agent_id], valid_actions)
                    actions[agent_id] = action
                else:
                    actions[agent_id] = (0, 0)

            results = env.step(actions)

            for agent_id, result_tuple in results.items():
                if agent_id < len(env.agents):
                    next_state_grid, next_state_feat, reward, done, _ = result_tuple
                    states[agent_id] = (next_state_grid, next_state_feat)
                    episode_rewards_sum[agent_id] += reward
                    episode_trajectories[agent_id].append(env.agents[agent_id].state.position)
                    if done:
                        episode_done = True

            if is_last_episode:
                if agent_id_to_track < len(env.agents):
                    tracked_agent = env.agents[agent_id_to_track]
                    last_ep_local_map_history.append(tracked_agent.local_map.copy())
                    last_ep_position_history.append(tracked_agent.state.position)
                    last_ep_orientation_history.append(tracked_agent.state.orientation)

            current_total_step = total_steps_across_episodes + episode_steps
            if env.writer:
                env.writer.add_scalar('Perf/Coverage_Area', env.current_global_coverage, current_total_step)
                env.writer.add_scalar('Perf/Coverage_Increase', env.global_coverage_increase, current_total_step)
                if env.current_rewards:
                    env.writer.add_scalar('Perf/Avg_Step_Reward', np.mean(list(env.current_rewards.values())), current_total_step)
                if env.metrics.communication_events:
                    env.writer.add_scalar('Perf/Comm_Events', env.metrics.communication_events[-1], current_total_step)

            episode_steps += 1
            total_steps_across_episodes += 1
            if episode_done or episode_steps >= env.max_steps_per_episode:
                break

        # End of Episode Updates
        final_coverage_perc = env.calculate_coverage_percentage()
        avg_ep_reward = np.mean(list(episode_rewards_sum.values())) if episode_rewards_sum else 0.0
        ep_duration = time.time() - ep_start_time
        for agent_id, agent in enumerate(env.agents):
            agent.update_epsilon()
            if env.writer:
                env.writer.add_scalar(f'Reward/Agent_{agent_id}_EpSum', episode_rewards_sum[agent_id], episode)
                env.writer.add_scalar(f'Params/Agent_{agent_id}_Epsilon', agent.epsilon, episode)
            env.metrics.epsilon_values[agent_id].append(agent.epsilon)

        if env.writer:
            env.writer.add_scalar('Episode/Steps', episode_steps, episode)
            env.writer.add_scalar('Episode/Coverage_Percent', final_coverage_perc, episode)
            env.writer.add_scalar('Episode/Avg_Reward', avg_ep_reward, episode)
            env.writer.add_scalar('Episode/Duration_Sec', ep_duration, episode)
            opt_steps_this_episode = episode_steps
            if opt_steps_this_episode > 0 and env.metrics.qmix_loss:
                avg_qmix_loss = np.mean(env.metrics.qmix_loss[-opt_steps_this_episode:])
                env.writer.add_scalar('Episode/Avg_QMIX_Loss', avg_qmix_loss, episode)

        # Save models periodically
        if episode > 0 and (episode % save_interval == 0 or episode == env.max_episodes - 1):
            print(f"--- Saving QMIX models at episode {episode} ---")
            env.save_models(episode, model_dir)
            print(f"--- Models saved to {model_dir} ---")

        # Visualize periodically or at the end
        if verbose and (episode % 50 == 0 or episode == env.max_episodes - 1):
            print(f"Visualizing episode {episode}...")
            try:
                env.visualize_coverage(episode_trajectories, episode=f"{episode} ({current_map_type})", title_suffix=" (Train Final)")
                if episode == env.max_episodes - 1:
                    env.visualize_learning_metrics()
            except Exception as e:
                print(f"   Visualization error: {e}")

        # Print episode summary
        if verbose and episode % 10 == 0:
            print(f"   Ep {episode} Summary: Steps={episode_steps}, Cov={final_coverage_perc:.2f}%, AvgRew={avg_ep_reward:.3f}, Time={ep_duration:.2f}s")

    print(f"\n--- QMIX Training finished after {env.max_episodes} episodes ---")
    if verbose:
        try:
            print("Generating final training visualizations...")
            env.visualize_coverage(episode_trajectories, episode=f"Final ({current_map_type})", title_suffix=" (Train Final)")
            env.visualize_learning_metrics()
        except Exception as e:
            print(f"   Final visualization error: {e}")
    if env.writer:
        env.writer.close()
        print("TensorBoard writer closed.")
    return env.metrics, last_ep_local_map_history, last_ep_position_history, last_ep_orientation_history, agent_id_to_track


def evaluate(env, num_eval_episodes=20, max_steps_per_episode=150, model_dir="./qmix_models",
            model_episode_tag=None, visualize_each_episode=False, visualize_timesteps=False,
            save_animations=False, animation_dir="./eval_animations_qmix"):
    """Evaluate trained QMIX agents on random maps (DQN Style)."""
    print("\n--- Starting QMIX Evaluation Phase ---")
    if not env.agents:
        env.reset(full_reset=True, mode='evaluate')
    if not env.agents:
        print("Error: Failed to init agents for eval.")
        return None
    if save_animations:
        ensure_dir(animation_dir)

    if model_episode_tag is None:
        latest_ep = -1
        try:
            if os.path.exists(model_dir):
                files = [f for f in os.listdir(model_dir) if f.startswith("qmix_checkpoint_ep") and f.endswith(".pt")]
                epochs = [int(f.split('_ep')[1].split('.pt')[0]) for f in files if f.split('_ep')[1].split('.pt')[0].isdigit()]
                if epochs:
                    latest_ep = max(epochs)
        except Exception:
            pass
        if latest_ep != -1:
            model_episode_tag = latest_ep
            print(f"Using latest model tag found: {model_episode_tag}")
        else:
            model_episode_tag = env.max_episodes - 1
            print(f"Using default model tag: {model_episode_tag}")

    models_loaded = env.load_models(model_episode_tag, model_dir)
    if not models_loaded:
        print("Warning: Evaluation proceeding with potentially untrained models.")
    for agent in env.agents:
        agent.epsilon = 0.0
        agent.policy_net.eval()
    if env.mixer:
        env.mixer.eval()

    all_episode_coverages = []
    all_episode_steps = []
    for episode in range(num_eval_episodes):
        is_last_episode = (episode == num_eval_episodes - 1)
        episode_label = f"Eval Ep {episode + 1}/{num_eval_episodes}"
        states = env.reset(full_reset=False, mode='evaluate')
        eval_trajectories = {i: [env.agents[i].state.position] for i in range(env.num_agents)}
        episode_done = False
        steps_taken = 0
        episode_coverage_history = []
        if visualize_timesteps and is_last_episode:
            episode_coverage_history.append(env.grid_to_matrix())

        for step in range(max_steps_per_episode):
            actions = {}
            for agent_id, agent in enumerate(env.agents):
                if agent_id in states:
                    valid_actions = env.get_valid_actions(agent_id)
                    action = agent.select_action(states[agent_id], valid_actions)
                    actions[agent_id] = action
                else:
                    actions[agent_id] = (0, 0)

            results = env.step(actions)

            for agent_id, result_tuple in results.items():
                if agent_id < len(env.agents):
                    next_state_grid, next_state_feat, _, done, _ = result_tuple
                    states[agent_id] = (next_state_grid, next_state_feat)
                    eval_trajectories[agent_id].append(env.agents[agent_id].state.position)
                    if done:
                        episode_done = True

            if visualize_timesteps and is_last_episode:
                episode_coverage_history.append(env.grid_to_matrix())
            steps_taken += 1
            if episode_done or steps_taken >= max_steps_per_episode:
                break

        final_coverage_perc = env.calculate_coverage_percentage()
        all_episode_coverages.append(final_coverage_perc)
        all_episode_steps.append(steps_taken)
        print(f"  {episode_label}: Steps={steps_taken}, Final Coverage={final_coverage_perc:.2f}%")
        if visualize_each_episode:
            env.visualize_coverage(eval_trajectories, episode=episode_label, title_suffix=" (Eval Final)")
            if env.num_agents > 0:
                env.visualize_agent_local_map(agent_id=0, episode=episode_label)
        if visualize_timesteps and is_last_episode:
            print(f"Generating animation for the final evaluation episode ({episode_label})...")
            anim_save_path = os.path.join(animation_dir, f"eval_qmix_ep_{episode + 1}_steps.mp4") if save_animations else None
            env.visualize_evaluation_timesteps(episode_coverage_history, episode_label, save_path=anim_save_path)

    avg_coverage = np.mean(all_episode_coverages) if all_episode_coverages else 0
    std_coverage = np.std(all_episode_coverages) if all_episode_coverages else 0
    avg_steps = np.mean(all_episode_steps) if all_episode_steps else 0
    print("\n--- QMIX Evaluation Summary ---")
    print(f" Model Tag: {model_episode_tag}, Episodes: {num_eval_episodes}")
    print(f" Avg Final Coverage (%): {avg_coverage:.2f} (StdDev: {std_coverage:.2f})")
    print(f" Avg Steps per Episode: {avg_steps:.1f}")
    print("--------------------------\n")
    return {"avg_coverage": avg_coverage, "std_coverage": std_coverage, "avg_steps": avg_steps}
