
from __future__ import annotations
from typing import Tuple, Dict, Any, List, Optional
from collections import deque

State = Tuple[int,...]

class EpisodicMemory:
    """
    Simple episodic memory:
    - Tabu: penalize moves that lead to states seen in the recent window
    - Backtrack penalty: discourage immediate undo of the last action
    """
    def __init__(self, window: int = 50, tabu_penalty: float = 0.5, backtrack_penalty: float = 0.5):
        self.recent = deque(maxlen=window)
        self.tabu_penalty = tabu_penalty
        self.backtrack_penalty = backtrack_penalty
        self.last_action = None

    def remember(self, state: State, action: Optional[str]):
        self.recent.append(state)
        self.last_action = action

    def cost_hook(self, cur: State, action: str, nxt: State, ctx: Dict[str,Any]) -> float:
        # discourage stepping into recently seen states
        penalty = 0.0
        if nxt in self.recent:
            penalty += self.tabu_penalty

        # discourage immediate backtracking (undo)
        inverse = {'U':'D','D':'U','L':'R','R':'L'}
        if ctx.get("last_action") and inverse.get(ctx["last_action"]) == action:
            penalty += self.backtrack_penalty

        return penalty
