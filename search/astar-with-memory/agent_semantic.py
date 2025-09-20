
from __future__ import annotations
from typing import Tuple, List, Dict, Any, Callable
from math import sqrt

State = Tuple[int,...]

# ---- Tiny "Vector DB" ----

def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().replace("_"," ").split() if t.isalnum()]

def _bow(text: str, vocab: Dict[str,int]) -> List[float]:
    vec = [0.0]*len(vocab)
    for tok in _tokenize(text):
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    return vec

def _cosine(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x,y in zip(a,b))
    da = sqrt(sum(x*x for x in a))
    db = sqrt(sum(x*x for x in b))
    if da == 0 or db == 0:
        return 0.0
    return num/(da*db)

class TinyVectorDB:
    def __init__(self):
        self.docs: List[Dict[str,Any]] = []
        self.vocab: Dict[str,int] = {}

    def add(self, text: str, payload: Dict[str,Any]):
        for tok in _tokenize(text):
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        self.docs.append({"text": text, "payload": payload})

    def search(self, query: str, k: int = 2) -> List[Dict[str,Any]]:
        qv = _bow(query, self.vocab)
        scored = []
        for d in self.docs:
            dv = _bow(d["text"], self.vocab)
            s = _cosine(qv, dv)
            scored.append((s, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s,d in scored[:k] if s > 0.0]

# ---- Hint Policies ----
# Each hint returns a move cost adjustment (positive=penalty, negative=reward).

def hint_protect_solved(cur: State, action: str, nxt: State, ctx: Dict[str,Any]) -> float:
    """Penalty if a move displaces any tile that's in its goal spot."""
    penalty = 0.0
    for i, v in enumerate(cur):
        if v != 0 and v == i+1:
            if nxt[i] != i+1:
                penalty += 0.3
    return penalty

def hint_focus_max_far_tile(cur: State, action: str, nxt: State, ctx: Dict[str,Any]) -> float:
    """Reward moves that bring the blank closer to the tile with the largest Manhattan distance."""
    from puzzle import IDX_TO_POS
    # find farthest tile in cur
    max_tile, max_d = None, -1
    for i, t in enumerate(cur):
        if t == 0: 
            continue
        r, c = IDX_TO_POS[i]
        gr, gc = IDX_TO_POS[t-1]
        d = abs(r-gr)+abs(c-gc)
        if d > max_d:
            max_d, max_tile = d, t
    # compute blank pos distance to that tile
    def blank_to_tile(state):
        bi = state.index(0)
        ti = state.index(max_tile)
        br, bc = IDX_TO_POS[bi]
        tr, tc = IDX_TO_POS[ti]
        return abs(br-tr)+abs(bc-tc)
    before = blank_to_tile(cur)
    after = blank_to_tile(nxt)
    # reward if blank moves closer to the far tile
    if after < before:
        return -0.2
    return 0.0

def hint_corner_blank_edge_slide(cur: State, action: str, nxt: State, ctx: Dict[str,Any]) -> float:
    """If blank is in a corner, lightly reward moves that keep it along edges."""
    corners = {0,2,6,8}
    if cur.index(0) in corners:
        return -0.05
    return 0.0

HINTS_CATALOG = [
    {
        "text": "many_solved corner_blank protect placed tiles",
        "fn": hint_protect_solved,
    },
    {
        "text": "few_solved max_far encourage blank move closer to far tile",
        "fn": hint_focus_max_far_tile,
    },
    {
        "text": "corner_blank prefer moving along edges",
        "fn": hint_corner_blank_edge_slide,
    },
]

class SemanticMemory:
    def __init__(self):
        self.db = TinyVectorDB()
        for h in HINTS_CATALOG:
            self.db.add(h["text"], {"fn": h["fn"]})

    def make_cost_hook(self, describe_fn) -> Callable:
        def hook(cur, action, nxt, ctx):
            query = describe_fn(cur)
            results = self.db.search(query, k=2)
            bonus = 0.0
            for r in results:
                fn = r["payload"]["fn"]
                bonus += float(fn(cur, action, nxt, ctx))
            return bonus
        return hook
