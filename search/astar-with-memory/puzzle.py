
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

# 8-puzzle representation:
# state is a tuple of length 9 with values 0..8, where 0 is the blank.
# Goal is (1,2,3,4,5,6,7,8,0)

GOAL = (1,2,3,4,5,6,7,8,0)
IDX_TO_POS = {i: (i//3, i%3) for i in range(9)}
POS_TO_IDX = {(r,c): r*3 + c for r in range(3) for c in range(3)}

MOVES = {
    'U': (-1, 0),
    'D': ( 1, 0),
    'L': ( 0,-1),
    'R': ( 0, 1),
}

def manhattan(state: Tuple[int, ...]) -> int:
    """Sum of Manhattan distances for all tiles except blank."""
    dist = 0
    for i, tile in enumerate(state):
        if tile == 0: 
            continue
        r, c = IDX_TO_POS[i]
        gr, gc = IDX_TO_POS[tile-1]  # goal index of tile
        dist += abs(r-gr) + abs(c-gc)
    return dist

def is_solvable(state: Tuple[int, ...]) -> bool:
    """Check solvability for 8-puzzle."""
    arr = [x for x in state if x != 0]
    inversions = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                inversions += 1
    # For 3x3, solvable iff inversions is even.
    return inversions % 2 == 0

def neighbors(state: Tuple[int, ...]) -> List[Tuple[str, Tuple[int,...]]]:
    """Return list of (action, next_state) pairs."""
    zi = state.index(0)
    zr, zc = IDX_TO_POS[zi]
    result = []
    for a, (dr, dc) in MOVES.items():
        nr, nc = zr + dr, zc + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            nzi = POS_TO_IDX[(nr, nc)]
            new_state = list(state)
            new_state[zi], new_state[nzi] = new_state[nzi], new_state[zi]
            result.append((a, tuple(new_state)))
    return result

def pretty(state: Tuple[int,...]) -> str:
    """ASCII rendering."""
    s = ""
    for r in range(3):
        row = []
        for c in range(3):
            v = state[r*3+c]
            row.append(" " if v==0 else str(v))
        s += " ".join(x.rjust(2) for x in row) + "\n"
    return s

def describe_state(state: Tuple[int,...]) -> str:
    """Short text description for semantic memory."""
    blank_corner = "corner_blank" if state.index(0) in (0,2,6,8) else "center_or_edge_blank"
    tiles_in_place = sum(1 for i,v in enumerate(state) if v != 0 and v == i+1)
    many_solved = "many_solved" if tiles_in_place >= 5 else "few_solved"
    # find tile with largest individual Manhattan distance
    max_tile, max_dist = None, -1
    for i, tile in enumerate(state):
        if tile == 0: 
            continue
        r, c = IDX_TO_POS[i]
        gr, gc = IDX_TO_POS[tile-1]
        d = abs(r-gr) + abs(c-gc)
        if d > max_dist:
            max_dist = d
            max_tile = tile
    max_token = f"max_far_{max_tile}_{max_dist}"
    return f"{blank_corner} {many_solved} {max_token}"
