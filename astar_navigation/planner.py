#!/usr/bin/env python3
"""
A* path planner
Operates on a float occupancy grid where:
  1.0  = free space
  0.0  = obstacle / unknown

Two planners are provided:
  plan_astar_4  — 4-connected grid (Euclidean heuristic, admissible)
  plan_astar_8  — 8-connected grid (diagonal moves, fewer nodes explored)

"""

import heapq
import numpy as np
from collections import deque


# ──────────────────────────────────────────────────────
#  Node
# ──────────────────────────────────────────────────────

class PlanNode:
    """Path-search tree node."""
    __slots__ = ('parent', 'cell', 'g', 'h', 'f')

    def __init__(self, parent=None, cell=None):
        self.parent = parent
        self.cell   = cell          # np.array([row, col])
        self.g      = 0.0           # cost from start
        self.h      = 0.0           # heuristic to goal
        self.f      = 0.0           # g + h

    def __lt__(self, other):        # needed for heapq tie-breaks
        return self.f < other.f

    def __str__(self):
        return str(self.cell)

    def __repr__(self):
        return str(self)


# ──────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────

def is_free(v, thr=0.9):
    """Return True if cell value v represents free space."""
    return v > thr


def _reconstruct_path(node):
    """Trace parent pointers back from goal to start."""
    path = []
    while node is not None:
        path.append(node.cell)
        node = node.parent
    path.reverse()
    return path


def _get_neighbors_4(cell, shape):
    """4-connected (N/S/E/W) neighbours, step cost 1."""
    r, c = cell
    candidates = [
        (np.array([r - 1, c]), 1.0),
        (np.array([r + 1, c]), 1.0),
        (np.array([r, c - 1]), 1.0),
        (np.array([r, c + 1]), 1.0),
    ]
    return [(nc, cost) for nc, cost in candidates
            if 0 <= nc[0] < shape[0] and 0 <= nc[1] < shape[1]]


def _get_neighbors_8(cell, shape):
    """8-connected (diagonals included), exact Euclidean step cost."""
    r, c = cell
    candidates = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nc = np.array([r + dr, c + dc])
            cost = np.sqrt(float(dr * dr + dc * dc))
            candidates.append((nc, cost))
    return [(nc, cost) for nc, cost in candidates
            if 0 <= nc[0] < shape[0] and 0 <= nc[1] < shape[1]]


def _euclidean(cell, goal):
    return np.sqrt(float((cell[0] - goal[0]) ** 2 + (cell[1] - goal[1]) ** 2))


# ──────────────────────────────────────────────────────
#  Core A*
# ──────────────────────────────────────────────────────

def _astar(n_start, n_goal, M, get_neighbors):
    """
    Internal A* implementation shared by both public functions.

    Parameters
    ----------
    n_start      : array-like [row, col]
    n_goal       : array-like [row, col]
    M            : (H, W) float array  (1.0=free, 0.0=obstacle)
    get_neighbors: callable(cell, shape) → list of (neighbor_cell, cost)

    Returns
    -------
    path    : list of np.array([row, col]), start→goal  ([] if not found)
    visited : (H, W) float array, 1.0 where cells were expanded
    """
    n_start = np.asarray(n_start, dtype=int)
    n_goal  = np.asarray(n_goal,  dtype=int)

    visited = np.zeros(M.shape, dtype=np.float32)

    start_node   = PlanNode(parent=None, cell=n_start)
    start_node.g = 0.0
    start_node.h = _euclidean(n_start, n_goal)
    start_node.f = start_node.h

    counter   = 0
    open_heap = [(start_node.f, counter, start_node)]
    open_dict = {(int(n_start[0]), int(n_start[1])): 0.0}

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        key = (int(current.cell[0]), int(current.cell[1]))
        if visited[key] != 0:
            continue
        visited[key] = 1.0

        if np.array_equal(current.cell, n_goal):
            return _reconstruct_path(current), visited

        for neighbor_cell, step_cost in get_neighbors(current.cell, M.shape):
            r, c = int(neighbor_cell[0]), int(neighbor_cell[1])
            if not is_free(M[r, c]) or visited[r, c] != 0:
                continue

            g    = current.g + step_cost
            h    = _euclidean(neighbor_cell, n_goal)
            f    = g + h
            nkey = (r, c)

            if nkey not in open_dict or open_dict[nkey] > g:
                open_dict[nkey] = g
                node        = PlanNode(parent=current, cell=neighbor_cell)
                node.g      = g
                node.h      = h
                node.f      = f
                counter    += 1
                heapq.heappush(open_heap, (f, counter, node))

    return [], visited


# ──────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────

def plan_astar_4(n_start, n_goal, M):
    """4-connected A* (same as plan_path_astar in the assignment)."""
    return _astar(n_start, n_goal, M, _get_neighbors_4)


def plan_astar_8(n_start, n_goal, M):
    """
    8-connected A* (same as plan_path_fast in the assignment).
    Explores fewer nodes and produces shorter paths than 4-connected.
    """
    return _astar(n_start, n_goal, M, _get_neighbors_8)


def nearest_free_cell(cell, free_map):
    """
    BFS outward from `cell` until a free cell is found.
    Returns the cell itself if it is already free.
    Used to snap start/goal into free space after inflation.
    """
    r0, c0 = int(cell[0]), int(cell[1])
    H, W   = free_map.shape

    if is_free(free_map[r0, c0]):
        return np.array([r0, c0])

    queue   = deque([(r0, c0)])
    visited = set()
    visited.add((r0, c0))

    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    if is_free(free_map[nr, nc]):
                        return np.array([nr, nc])
                    queue.append((nr, nc))

    return np.array([r0, c0])   # fallback — map is completely blocked
