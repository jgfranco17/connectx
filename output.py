"""
OUTPUT FILE FOR CONNECTX
"""
import random
import numpy as np


def count_windows(grid, discs, piece, config) -> int:
    windows = 0

    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if window_check(window, discs, piece, config):
                windows += 1

    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if window_check(window, discs, piece, config):
                windows += 1
    
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if window_check(window, discs, piece, config):
                windows += 1
    
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if window_check(window, discs, piece, config):
                windows += 1
    
    return windows


def drop(grid, col, piece, config):
    next_grid = grid.copy()

    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
        
    next_grid[row][col] = piece

    return next_grid


def is_end_window(window, config) -> bool:
    return window.count(1) == config.inarow or window.count(2) == config.inarow


def is_end_node(grid, config) -> bool:
    # Check for draw 
    if list(grid[0,:]).count(0) == 0:
        return True
        
    # check horizontal 
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if is_end_window(window, config):
                return True

    # check vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if is_end_window(window, config):
                return True

    # check positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if is_end_window(window, config):
                return True

    # check negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if is_end_window(window, config):
                return True
    
    return False


def window_check(window, discs, piece, config) -> bool:
    return (window.count(piece) == discs and window.count(0) == config.inarow-discs)


def heuristic(grid, mark, config) -> float:
    num_threes = count_windows(grid, 3, mark, config)
    num_fours = count_windows(grid, 4, mark, config)
    num_threes_opp = count_windows(grid, 3, mark%2+1, config)
    num_fours_opp = count_windows(grid, 4, mark%2+1, config)
    score = num_threes - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours

    return score


def minimax(node, depth, maximize, mark, config) -> float:
    is_terminal = is_end_node(node, config)
    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
    value = 0
    
    if depth == 0 or is_terminal:
        return heuristic(node, mark, config)

    if maximize:
        value = -np.Inf
        for col in valid_moves:
            child = drop(node, col, mark, config)
            value = max(value, minimax(child, depth-1, False, mark, config))

    else:
        value = np.Inf
        for col in valid_moves:
            child = drop(node, col, mark%2+1, config)
            value = min(value, minimax(child, depth-1, True, mark, config))
        
    return value


def score_move(grid, col, mark, config, nsteps) -> float:
    next_grid = drop(grid, col, mark, config)
    score = minimax(next_grid, nsteps-1, False, mark, config)

    return score


def agent_minimax(obs, config):
    N_STEPS = 3
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    
    return random.choice(max_cols)

