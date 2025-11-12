"""
Utility functions for QMIX multi-agent coverage system.
"""

import os
import errno
import random
import numpy as np
from data_structures import Room, NUM_ROOMS_RANGE, ROOM_SIZE_RANGE


def ensure_dir(directory):
    """Ensures that a directory exists, creating it if necessary."""
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def generate_room_map(grid_size, num_rooms_range=NUM_ROOMS_RANGE, room_size_range=ROOM_SIZE_RANGE):
    """Generate a room-based map."""
    grid = np.ones((grid_size, grid_size), dtype=np.float32)
    rooms = []
    num_rooms_target = random.randint(*num_rooms_range)
    max_tries = num_rooms_target * 5

    for _ in range(max_tries):
        if len(rooms) >= num_rooms_target:
            break
        h = random.randint(*room_size_range)
        w = random.randint(*room_size_range)
        if h >= grid_size - 2 or w >= grid_size - 2:
            continue
        r = random.randint(1, grid_size - h - 2)
        c = random.randint(1, grid_size - w - 2)
        new_room = Room(r, c, h, w)
        intersects = False
        for other_room in rooms:
            buffered_other = Room(other_room.r1 - 1, other_room.c1 - 1,
                                 other_room.height + 2, other_room.width + 2)
            if new_room.intersects(buffered_other):
                intersects = True
                break
        if not intersects:
            grid[new_room.r1:new_room.r2, new_room.c1:new_room.c2] = 0.0
            if rooms:
                prev_room = rooms[-1]
                pr, pc = prev_room.center
                cr, cc = new_room.center
                min_r, max_r = min(pr, cr), max(pr, cr)
                min_c, max_c = min(pc, cc), max(pc, cc)
                safe_pr = np.clip(pr, 0, grid_size - 1)
                safe_cc = np.clip(cc, 0, grid_size - 1)
                safe_pc = np.clip(pc, 0, grid_size - 1)
                safe_cr = np.clip(cr, 0, grid_size - 1)
                if random.random() < 0.5:
                    grid[safe_pr, min_c : max_c + 1] = 0.0
                    grid[min_r : max_r + 1, safe_cc] = 0.0
                else:
                    grid[min_r : max_r + 1, safe_pc] = 0.0
                    grid[safe_cr, min_c : max_c + 1] = 0.0
            rooms.append(new_room)

    if not rooms:
        center_r, center_c = grid_size // 2, grid_size // 2
        size = max(1, grid_size // 10)
        r1, c1 = max(0, center_r - size // 2), max(0, center_c - size // 2)
        r2, c2 = min(grid_size, center_r + (size + 1) // 2), min(grid_size, center_c + (size + 1) // 2)
        grid[r1:r2, c1:c2] = 0.0

    grid[0, :] = 1.0
    grid[-1, :] = 1.0
    grid[:, 0] = 1.0
    grid[:, -1] = 1.0
    return grid


def generate_cave_map(grid_size, fill_prob=0.48, smoothing_iterations=5,
                      birth_limit=4, death_limit=3):
    """Generate a cave-like map using cellular automata."""
    width, height = grid_size, grid_size
    grid = np.ones((height, width), dtype=np.float32)
    inner_fill = (np.random.rand(height - 2, width - 2) < fill_prob).astype(np.float32)
    grid[1:-1, 1:-1] = inner_fill

    def count_walls(x, y, current_grid):
        sub_grid = current_grid[max(0, y-1):min(height, y+2), max(0, x-1):min(width, x+2)]
        return np.sum(sub_grid) - current_grid[y, x]

    for _ in range(smoothing_iterations):
        new_grid = grid.copy()
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                walls = count_walls(x, y, grid)
                if grid[y, x] == 1:
                    if walls < death_limit:
                        new_grid[y, x] = 0
                elif walls > birth_limit:
                    new_grid[y, x] = 1
        grid = new_grid

    grid[0, :] = 1.0
    grid[-1, :] = 1.0
    grid[:, 0] = 1.0
    grid[:, -1] = 1.0
    return grid


def generate_empty_map(grid_size):
    """Generate an empty map with only boundary walls."""
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    grid[0, :] = 1.0
    grid[-1, :] = 1.0
    grid[:, 0] = 1.0
    grid[:, -1] = 1.0
    return grid


def generate_random_map(grid_size, obstacle_density=0.15):
    """Generate a random obstacle map."""
    grid = generate_empty_map(grid_size)
    inner_area_size = (grid_size - 2) * (grid_size - 2)
    if inner_area_size <= 0:
        return grid
    num_obstacles = min(int(obstacle_density * inner_area_size), inner_area_size)
    if num_obstacles > 0:
        inner_indices = np.random.choice(inner_area_size, num_obstacles, replace=False)
        rows, cols = np.unravel_index(inner_indices, (grid_size - 2, grid_size - 2))
        grid[rows + 1, cols + 1] = 1.0
    return grid


def moving_average(data, window_size):
    """Calculate moving average for smoothing data."""
    if not data or len(data) < window_size:
        return data, np.arange(len(data))
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    x_ticks = np.arange(window_size - 1, len(data))
    return smoothed, x_ticks
