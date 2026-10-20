"""Pluggable feed ranking. Every recommender sees only state from before the
current tick (the environment hands it a frozen snapshot), so all agents can
act in parallel within a tick.

    rank(agent_id, candidates, state) -> list[Post]   (ordered, may be shorter)
"""

from __future__ import annotations

import hashlib
import math
import random
import re


class Recommender:
    name = "base"

    def __init__(self, cfg, seed: int):
        self.cfg = cfg
        self.rng = random.Random(seed)

    def rank(self, agent_id: str, candidates: list, state) -> list:
        raise NotImplementedError


class RandomRecommender(Recommender):
    name = "random"

    def rank(self, agent_id, candidates, state):
        c = list(candidates)
        self.rng.shuffle(c)
        return c


class FollowingRecommender(Recommender):
    """Followed accounts first (newest first), then nothing -- a pure follow-graph feed.
    With `backfill` the remainder is random, so new accounts can be discovered."""
    name = "following"
    backfill = True

    def rank(self, agent_id, candidates, state):
        following = state.agents[agent_id].following
        mine = [p for p in candidates if p.creator_id in following
                or any(r in following for r in p.resharers)]
        mine.sort(key=lambda p: (-p.tick_created, p.post_id))
        if not self.backfill:
            return mine
        rest = [p for p in candidates if p not in mine]
        self.rng.shuffle(rest)
        return mine + rest


class SemanticRecommender(Recommender):
    """Cosine similarity between a post and the agent's taste vector (mean
    embedding of posts it opened). Embeddings come from an `embed(text)`
    callable; the default is a deterministic hashed bag-of-words so nothing
    external is needed. Swap in a real model by passing `embed`."""
    name = "semantic"

    def __init__(self, cfg, seed, embed=None):
        super().__init__(cfg, seed)
        self.embed = embed or self._hash_embed
        self._cache: dict[str, list[float]] = {}

    def _hash_embed(self, text: str) -> list[float]:
        v = [0.0] * self.cfg.semantic_dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.cfg.semantic_dim] += 1.0 if (h >> 8) % 2 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def _vec(self, post):
        if post.post_id not in self._cache:
            self._cache[post.post_id] = self.embed(post.hook + " " + post.body)
        return self._cache[post.post_id]

    def rank(self, agent_id, candidates, state):
        viewed = [state.posts[p] for p in state.agents[agent_id].viewed if p in state.posts]
        if not viewed:
            c = list(candidates)
            self.rng.shuffle(c)
            return c
        taste = [sum(col) / len(viewed) for col in zip(*(self._vec(p) for p in viewed))]
        def score(p):
            v = self._vec(p)
            return sum(a * b for a, b in zip(taste, v)) + self.rng.random() * 1e-6
        return sorted(candidates, key=score, reverse=True)


class EngagementRecommender(Recommender):
    """score = (w_view*views + w_like*likes + w_reshare*reshares)^alpha * decay^age
    using engagement recorded through the previous tick only."""
    name = "engagement"

    def rank(self, agent_id, candidates, state):
        cfg = self.cfg
        def score(p):
            eng = cfg.w_view * p.full_views + cfg.w_like * p.likes + cfg.w_reshare * p.reshares
            age = max(0, state.tick - p.tick_created)
            return (eng ** cfg.engagement_exponent) * (cfg.recency_decay ** age) + self.rng.random() * 1e-6
        return sorted(candidates, key=score, reverse=True)


REGISTRY = {
    "random": RandomRecommender,
    "following": FollowingRecommender,
    "semantic": SemanticRecommender,
    "engagement": EngagementRecommender,
}


def make(name: str, cfg, seed: int) -> Recommender:
    if name not in REGISTRY:
        raise KeyError(f"unknown recommender {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](cfg, seed)
