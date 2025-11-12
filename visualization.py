"""
Visualization functions for QMIX multi-agent coverage system.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import ensure_dir, moving_average


def visualize_coverage_plot(env, trajectories, episode=None, coverage_matrix=None, title_suffix=""):
    """Plot coverage map and agent trajectories (DQN Style)."""
    plt.figure(figsize=(11, 10))
    matrix_to_plot = coverage_matrix if coverage_matrix is not None else env.grid_to_matrix()
    cmap = plt.cm.YlGn.copy()
    cmap.set_under('black')
    plt.imshow(matrix_to_plot.T, cmap=cmap, origin='lower', interpolation='nearest', vmin=0.0, vmax=1.0)
    colors = plt.cm.get_cmap('tab10', env.num_agents)
    for idx, path in trajectories.items():
        if idx < len(env.agents):
            agent_color = colors(idx)
            if path and len(path) > 1:
                plt.plot([p[0] for p in path], [p[1] for p in path], '-', linewidth=1, color=agent_color, alpha=0.6)
            current_pos = env.agents[idx].state.position
            plt.scatter(current_pos[0], current_pos[1], color=agent_color, s=120, marker='o', edgecolors='black', label=f'Agent {idx}')
            orientation = env.agents[idx].state.orientation
            arrow_len = 0.8
            plt.arrow(current_pos[0], current_pos[1], arrow_len * math.cos(orientation), arrow_len * math.sin(orientation),
                     head_width=0.3, head_length=0.4, fc=agent_color, ec='black', length_includes_head=True)
    coverage_percentage = env.calculate_coverage_percentage()
    title = "QMIX Multi-agent Coverage Map"
    if episode is not None:
        title += f" (Episode {episode}, Coverage: {coverage_percentage:.1f}%)"
    if title_suffix:
        title += f" {title_suffix}"
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.colorbar(label='Coverage Probability (0-1)', shrink=0.8, extend='min')
    plt.xlim([-0.5, env.grid_size - 0.5])
    plt.ylim([-0.5, env.grid_size - 0.5])
    plt.xticks(np.arange(0, env.grid_size, step=max(1, env.grid_size // 10)))
    plt.yticks(np.arange(0, env.grid_size, step=max(1, env.grid_size // 10)))
    plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='white', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()


def visualize_agent_local_map_plot(env, agent_id, episode=None):
    """Visualizes the final local map perspective of a specific agent."""
    if not (0 <= agent_id < len(env.agents)):
        return
    agent = env.agents[agent_id]
    if not agent.local_map:
        return
    local_matrix = env.grid_to_matrix(graph=agent.local_map)
    plt.figure(figsize=(9, 8))
    cmap = plt.cm.YlGn.copy()
    cmap.set_under('black')
    plt.imshow(local_matrix.T, cmap=cmap, origin='lower', interpolation='nearest', vmin=0.0, vmax=1.0)
    agent_pos = agent.state.position
    agent_color = plt.cm.get_cmap('tab10', env.num_agents)(agent_id)
    plt.scatter(agent_pos[0], agent_pos[1], color=agent_color, s=120, marker='o', label=f'Agent {agent_id} Pos', edgecolors='black')
    orientation = agent.state.orientation
    arrow_len = 0.8
    plt.arrow(agent_pos[0], agent_pos[1], arrow_len * math.cos(orientation), arrow_len * math.sin(orientation),
             head_width=0.3, head_length=0.4, fc=agent_color, ec='black', length_includes_head=True)
    title = f"Agent {agent_id}'s Final Local Map Perspective"
    if episode is not None:
        title += f" (End of Episode {episode})"
    plt.title(title)
    plt.colorbar(label='Coverage Probability (0-1)', shrink=0.8, extend='min')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.xlim([-0.5, env.grid_size - 0.5])
    plt.ylim([-0.5, env.grid_size - 0.5])
    plt.xticks(np.arange(0, env.grid_size, step=max(1, env.grid_size // 10)))
    plt.yticks(np.arange(0, env.grid_size, step=max(1, env.grid_size // 10)))
    plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='white', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()


def visualize_agent_local_timesteps_animation(env, agent_id, local_map_history, position_history,
                                              orientation_history, episode_label, save_path=None):
    """Animates an agent's local map evolution with position/orientation."""
    if not local_map_history or not position_history or not orientation_history:
        print("No history data for local agent timestep visualization")
        return
    min_len = min(len(local_map_history), len(position_history), len(orientation_history))
    if min_len == 0:
        return
    local_map_history, position_history, orientation_history = local_map_history[:min_len], position_history[:min_len], orientation_history[:min_len]
    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.cm.YlGn.copy()
    cmap.set_under('black')
    initial_matrix = env.grid_to_matrix(graph=local_map_history[0])
    im = ax.imshow(initial_matrix.T, cmap=cmap, origin='lower', vmin=0.0, vmax=1.0, interpolation='nearest')
    colorbar = fig.colorbar(im, ax=ax, label='Coverage Prob (0-1)', shrink=0.8, extend='min')
    agent_color = plt.cm.get_cmap('tab10', env.num_agents)(agent_id)
    scatter = ax.scatter([], [], s=120, marker='o', color=agent_color, edgecolors='k', zorder=2)
    arrow_obj = ax.arrow(0, 0, 0, 0, color=agent_color, head_width=0.3, head_length=0.4, zorder=3, length_includes_head=True)
    trajectory_line, = ax.plot([], [], lw=1, color=agent_color, alpha=0.6)
    ax.set_xlim([-0.5, env.grid_size - 0.5])
    ax.set_ylim([-0.5, env.grid_size - 0.5])
    ax.set_xticks(np.arange(0, env.grid_size, step=max(1, env.grid_size // 10)))
    ax.set_yticks(np.arange(0, env.grid_size, step=max(1, env.grid_size // 10)))
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='white', alpha=0.5)

    def update(frame):
        nonlocal arrow_obj
        current_matrix = env.grid_to_matrix(graph=local_map_history[frame])
        im.set_data(current_matrix.T)
        x, y = position_history[frame]
        scatter.set_offsets([x, y])
        arrow_angle = orientation_history[frame]
        dx = 0.8 * math.cos(arrow_angle)
        dy = 0.8 * math.sin(arrow_angle)
        if arrow_obj:
            arrow_obj.remove()
        arrow_obj = ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.4, fc=agent_color, ec='k', zorder=3, length_includes_head=True)
        x_vals = [p[0] for p in position_history[:frame+1]]
        y_vals = [p[1] for p in position_history[:frame+1]]
        trajectory_line.set_data(x_vals, y_vals)
        ax.set_title(f"Agent {agent_id} Local Map\n{episode_label} - Step {frame}/{len(local_map_history)-1}")
        return [im, scatter, arrow_obj, trajectory_line]

    ani = animation.FuncAnimation(fig, update, frames=len(local_map_history), interval=200, blit=False, repeat=False)
    if save_path:
        try:
            ensure_dir(os.path.dirname(save_path))
            ani.save(save_path, writer='ffmpeg', fps=5, dpi=100)
            print(f"Agent local map animation saved: {save_path}")
        except Exception as e:
            print(f"Error saving agent local map animation: {e}\nDisplaying instead.")
            plt.show()
    else:
        plt.show()
    plt.close(fig)


def visualize_evaluation_timesteps_animation(env, coverage_history, episode_label, save_path=None):
    """Generates an animation showing global coverage evolution (DQN Style)."""
    if not coverage_history:
        print("No coverage history data for evaluation timestep visualization.")
        return
    fig, ax = plt.subplots(figsize=(9, 8))
    num_steps = len(coverage_history)
    initial_matrix = coverage_history[0]
    cmap = plt.cm.YlGn.copy()
    cmap.set_under('black')
    im = ax.imshow(initial_matrix.T, cmap=cmap, origin='lower', interpolation='nearest', vmin=0.0, vmax=1.0)
    colorbar = fig.colorbar(im, ax=ax, label='Coverage Prob (0-1)', shrink=0.8, extend='min')
    title = ax.set_title(f"{episode_label} - Step 0/{num_steps-1}")
    ax.set_xlim([-0.5, env.grid_size - 0.5])
    ax.set_ylim([-0.5, env.grid_size - 0.5])
    ax.set_xticks(np.arange(0, env.grid_size, step=max(1, env.grid_size // 5)))
    ax.set_yticks(np.arange(0, env.grid_size, step=max(1, env.grid_size // 5)))
    ax.grid(True, linestyle=':', linewidth=0.5, color='white', alpha=0.5)
    obstacle_mask = (env.obstacle_grid == 1.0) if env.obstacle_grid is not None else np.zeros_like(initial_matrix, dtype=bool)
    if obstacle_mask.shape != initial_matrix.shape:
        traversable_nodes_count = initial_matrix.size
    else:
        traversable_nodes_count = initial_matrix.size - np.sum(obstacle_mask)

    def update(frame):
        matrix = coverage_history[frame]
        im.set_data(matrix.T)
        covered_count = 0
        if traversable_nodes_count > 0:
            covered_mask = (matrix >= env.coverage_threshold)
            if covered_mask.shape == obstacle_mask.shape:
                covered_count = np.sum(covered_mask & (~obstacle_mask))
            else:
                covered_count = np.sum(covered_mask)
            cov_pct = (covered_count / traversable_nodes_count) * 100.0
        else:
            cov_pct = 100.0 if np.all(matrix >= env.coverage_threshold) else 0.0
        title.set_text(f"{episode_label} - Step {frame}/{num_steps-1} (Coverage: {cov_pct:.1f}%)")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=150, blit=False, repeat=False)
    if save_path:
        try:
            ensure_dir(os.path.dirname(save_path))
            ani.save(save_path, writer='ffmpeg', fps=10, dpi=100)
            print(f"Evaluation animation saved: {save_path}")
        except Exception as e:
            print(f"Error saving evaluation animation: {e}\nDisplaying instead.")
            plt.show()
    else:
        plt.show()
    plt.close(fig)


def visualize_learning_metrics_dashboard(env):
    """Visualize learning metrics dashboard (DQN Style + QMIX)."""
    if not env.metrics.total_coverage and not env.metrics.qmix_loss:
        print("No metrics data available.")
        return
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()
    fig.suptitle("QMIX Learning Metrics Dashboard", fontsize=16)

    smooth_window = max(1, len(env.metrics.total_coverage) // 20) if env.metrics.total_coverage else 1

    # Plot 1: Total Coverage Area
    ax = axes[0]
    steps = np.arange(len(env.metrics.total_coverage))
    if len(steps) > 0:
        ax.plot(steps, env.metrics.total_coverage, 'g-', alpha=0.3, label='Raw')
        smoothed_cov, x_cov = moving_average(env.metrics.total_coverage, smooth_window)
        ax.plot(x_cov, smoothed_cov, 'g-', linewidth=2, label=f'Smoothed ({smooth_window})')
    ax.set_title('Total Coverage Area (Sum pc)')
    ax.set_xlabel('Environment Step')
    ax.set_ylabel('Area')
    ax.legend()
    ax.grid(True, alpha=0.5)

    # Plot 2: Epsilon Decay
    ax = axes[1]
    num_eps = 0
    colors = plt.cm.viridis(np.linspace(0, 0.8, env.num_agents))
    for aid, eps_vals in env.metrics.epsilon_values.items():
        if eps_vals:
            num_eps = len(eps_vals)
            ax.plot(eps_vals, label=f'Agent {aid}', color=colors[aid], alpha=0.8)
    if num_eps > 0:
        ax.set_title('Epsilon Decay')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
    ax.legend()
    ax.grid(True, alpha=0.5)

    # Plot 3: Average Agent Reward (Individual Rewards)
    ax = axes[2]
    avg_rewards = []
    step_counts = [len(r) for r in env.metrics.agent_rewards.values()]
    if step_counts:
        max_steps = max(step_counts) if step_counts else 0
        for step in range(max_steps):
            step_rewards = [env.metrics.agent_rewards[aid][step] for aid in env.metrics.agent_rewards if step < len(env.metrics.agent_rewards[aid])]
            if step_rewards:
                avg_rewards.append(np.mean(step_rewards))
        if avg_rewards:
            ax.plot(np.arange(len(avg_rewards)), avg_rewards, 'b-', alpha=0.3, label='Raw Avg Individual Reward')
            smoothed_rew, x_rew = moving_average(avg_rewards, smooth_window * 2)
            ax.plot(x_rew, smoothed_rew, 'b-', linewidth=2, label=f'Smoothed ({smooth_window*2})')
    ax.set_title('Average Agent Reward per Step')
    ax.set_xlabel('Environment Step')
    ax.set_ylabel('Avg Reward')
    ax.legend()
    ax.grid(True, alpha=0.5)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

    # Plot 4: QMIX Loss
    ax = axes[3]
    if env.metrics.qmix_loss:
        loss_steps = np.arange(len(env.metrics.qmix_loss))
        ax.plot(loss_steps, env.metrics.qmix_loss, 'r-', alpha=0.3, label='Raw QMIX Loss')
        smoothed_loss, x_loss = moving_average(env.metrics.qmix_loss, smooth_window * 2)
        ax.plot(x_loss, smoothed_loss, 'r-', linewidth=2, label=f'Smoothed QMIX Loss ({smooth_window*2})')
    ax.set_title('QMIX Loss per Opt. Step')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.5)

    # Plot 5: Communication Events
    ax = axes[4]
    if env.metrics.communication_events:
        ax.plot(np.arange(len(env.metrics.communication_events)), env.metrics.communication_events, 'm-', alpha=0.6)
    ax.set_title('Communication Events per Step')
    ax.set_xlabel('Environment Step')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.5)

    # Plot 6: Coverage Rate (Team Reward Basis)
    ax = axes[5]
    if env.metrics.coverage_rate:
        ax.plot(np.arange(len(env.metrics.coverage_rate)), env.metrics.coverage_rate, 'c-', alpha=0.3, label='Raw')
        smoothed_rate, x_rate = moving_average(env.metrics.coverage_rate, smooth_window)
        ax.plot(x_rate, smoothed_rate, 'c-', linewidth=2, label=f'Smoothed ({smooth_window})')
    ax.set_title('Coverage Rate (Δ Area / Step)')
    ax.set_xlabel('Environment Step')
    ax.set_ylabel('Δ Area')
    ax.legend()
    ax.grid(True, alpha=0.5)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
