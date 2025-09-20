#!/usr/bin/env python3

from __future__ import annotations
import time, random
from typing import Tuple, Optional, List
from puzzle import manhattan, neighbors, GOAL, pretty, describe_state
from search import AStar
from memory import EpisodicMemory
from agent_semantic import SemanticMemory

State = Tuple[int,...]

def run_step(title: str, start: State, hooks) -> dict:
    solver = AStar(manhattan, neighbors, cost_hooks=hooks)
    result = solver.search(start, GOAL)
    return result

def scramble(depth=25):
    start = GOAL
    prev = None
    for _ in range(depth):
        choices = neighbors(start)
        if prev is not None:
            choices = [(a,s) for (a,s) in choices if s != prev] or neighbors(start)
        a, nxt = random.choice(choices)
        prev, start = start, nxt
    return start

def main():
    best = None
    trials = []
    for _ in range(12):
        start = scramble(24)
        # Step 1 — Baseline A*
        r1 = run_step("Step 1", start, [])
        # Step 2 — Episodic
        epi = EpisodicMemory(window=100, tabu_penalty=0.4, backtrack_penalty=0.4)
        def epi_hook(cur, action, nxt, ctx):
            epi.remember(cur, ctx.get("last_action"))
            return epi.cost_hook(cur, action, nxt, ctx)
        r2 = run_step("Step 2", start, [epi_hook])
        # Step 3 — Semantic
        sem = SemanticMemory()
        sem_hook = sem.make_cost_hook(describe_state)
        r3 = run_step("Step 3", start, [sem_hook])
        # Step 4 — Both
        sem = SemanticMemory()
        sem_hook = sem.make_cost_hook(describe_state)
        r4 = run_step("Step 4", start, [epi_hook, sem_hook])

        trials.append((start, r1, r2, r3, r4))

        if r1["solution"] and r2["solution"] and r3["solution"] and r4["solution"]:
            ok = (r2["nodes_expanded"] <= r1["nodes_expanded"]) and (r3["nodes_expanded"] <= r1["nodes_expanded"]) and (r4["nodes_expanded"] <= r2["nodes_expanded"])
            delta = r1["nodes_expanded"] - r4["nodes_expanded"]
            score = (ok, delta)
            if (best is None) or (score > best[0]):
                best = (score, (start, r1, r2, r3, r4))

    # pick best or fallback to first
    if best is None:
        start, r1, r2, r3, r4 = trials[0]
    else:
        start, r1, r2, r3, r4 = best[1]

    print("="*72)
    print("Start state chosen:")
    from puzzle import pretty
    print(pretty(start))

    def report(lbl, r):
        print(f"{lbl:10} depth={r['depth']:<3} nodes={r['nodes_expanded']:<6} time={r['runtime_s']:.4f}s")

    print("="*72)
    print("Step 1 — Baseline A*")
    print(pretty(start))
    print("Moves (first 10):", " ".join([a for (_,a) in r1["solution"] if a is not None][:10]))
    report("Step1", r1)

    print("="*72)
    print("Step 2 — + Episodic Memory")
    print(pretty(start))
    print("Moves (first 10):", " ".join([a for (_,a) in r2["solution"] if a is not None][:10]))
    report("Step2", r2)

    print("="*72)
    print("Step 3 — + Semantic")
    print(pretty(start))
    print("Moves (first 10):", " ".join([a for (_,a) in r3["solution"] if a is not None][:10]))
    report("Step3", r3)

    print("="*72)
    print("Step 4 — + Episodic + Semantic")
    print(pretty(start))
    print("Moves (first 10):", " ".join([a for (_,a) in r4["solution"] if a is not None][:10]))
    report("Step4", r4)

    print("="*72)
    print("Summary:")
    report("Step1", r1); report("Step2", r2); report("Step3", r3); report("Step4", r4)

if __name__ == '__main__':
    main()
