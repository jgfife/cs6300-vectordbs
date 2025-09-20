
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Callable, Any
import heapq
import time

State = Tuple[int,...]
CostHook = Callable[[State, str, State, Dict[str,Any]], float]

@dataclass(order=True)
class Node:
    f: float
    g: float
    h: float
    state: State = field(compare=False)
    action: Optional[str] = field(default=None, compare=False)
    parent: Optional["Node"] = field(default=None, compare=False)

class AStar:
    def __init__(self, h_fn: Callable[[State], int], 
                 neighbor_fn: Callable[[State], List[Tuple[str, State]]],
                 cost_hooks: Optional[List[CostHook]] = None):
        self.h_fn = h_fn
        self.neighbor_fn = neighbor_fn
        self.cost_hooks = cost_hooks or []

    def search(self, start: State, goal: State) -> Dict[str,Any]:
        start_time = time.time()
        open_heap: List[Node] = []
        start_h = self.h_fn(start)
        heapq.heappush(open_heap, Node(f=float(start_h), g=0.0, h=float(start_h), state=start))
        closed: Dict[State, float] = {}
        expansions = 0

        while open_heap:
            node = heapq.heappop(open_heap)
            # Closed check
            if node.state in closed and closed[node.state] <= node.g:
                continue
            closed[node.state] = node.g

            if node.state == goal:
                runtime = time.time() - start_time
                path = self._reconstruct(node)
                return {
                    "solution": path,
                    "nodes_expanded": expansions,
                    "runtime_s": runtime,
                    "depth": len(path)-1,
                }

            expansions += 1
            for action, nxt in self.neighbor_fn(node.state):
                # base move cost 1
                cost = 1.0
                # cost shaping hooks
                extra_context = {
                    "parent": node.parent.state if node.parent else None,
                    "last_action": node.action,
                }
                for hook in self.cost_hooks:
                    cost += float(hook(node.state, action, nxt, extra_context))

                g2 = node.g + cost
                h2 = float(self.h_fn(nxt))
                f2 = g2 + h2
                heapq.heappush(open_heap, Node(f=f2, g=g2, h=h2, 
                                               state=nxt, action=action, parent=node))
        # No solution
        runtime = time.time() - start_time
        return {
            "solution": None,
            "nodes_expanded": expansions,
            "runtime_s": runtime,
            "depth": None,
        }

    @staticmethod
    def _reconstruct(node: Node) -> List[Tuple[State, Optional[str]]]:
        seq = []
        cur = node
        while cur:
            seq.append((cur.state, cur.action))
            cur = cur.parent
        seq.reverse()
        return seq
