"""
Common utilities for EWR/NED computation on k-hop neighborhoods.

Dependencies:
  - Python >= 3.10
  - networkx, numpy, matplotlib

Author: (c) 2025, Biró Csaba
License: MIT (or as you prefer; edit if needed)
"""

from collections import deque
from typing import Set, Tuple, Dict, List

import numpy as np
import networkx as nx


def k_hop_nodes(G: nx.Graph, source: int, k: int) -> Set[int]:
    """Return the set of nodes within <= k hops from 'source', excluding 'source'.
    BFS over unweighted graph.
    """
    if k <= 0:
        return set()
    seen = {source}
    q = deque([(source, 0)])
    Nk = set()
    while q:
        u, dist = q.popleft()
        if dist == k:
            continue
        for w in G[u]:
            if w not in seen:
                seen.add(w)
                Nk.add(w)
                q.append((w, dist + 1))
    return Nk


def local_degrees_in_subset(G: nx.Graph, nodes: Set[int]) -> np.ndarray:
    """Return local degrees measured inside the subgraph induced by 'nodes'."""
    if not nodes:
        return np.array([], dtype=float)
    nodeset = set(nodes)
    degs = []
    for u in nodes:
        cnt = 0
        for w in G[u]:
            if w in nodeset:
                cnt += 1
        degs.append(float(cnt))
    return np.array(degs, dtype=float)


def shannon_entropy_from_probs(p: np.ndarray, log_base: float = 2.0) -> float:
    """Shannon entropy for a probability vector p (ignoring zeros)."""
    if p.size == 0:
        return 0.0
    mask = p > 0
    if not np.any(mask):
        return 0.0
    logp = np.log(p[mask]) / np.log(log_base)
    return float(-np.sum(p[mask] * logp))


def ewr_ned_for_node(G: nx.Graph, v, k: int) -> Tuple[float, float, float, float, int]:
    """Compute EWR, NED, H, R, and Nk size for a single node v.

    Definitions follow the manuscript:
      - H: Shannon entropy of local-degree distribution inside the induced subgraph on N_k(v)
      - R: induced edge density on N_k(v)
      - EWR = H * R
      - NED = (H / |N_k(v)|) * R  (0 if |N_k(v)| == 0)
    """
    Nk = k_hop_nodes(G, v, k)
    n_ind = len(Nk)
    if n_ind == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    local_degs = local_degrees_in_subset(G, Nk)
    S = float(local_degs.sum())
    if S <= 0:
        return 0.0, 0.0, 0.0, 0.0, n_ind

    p = local_degs / S
    H = shannon_entropy_from_probs(p, log_base=2.0)
    m_ind = float(local_degs.sum() / 2.0)
    R = 0.0 if n_ind < 2 else float(m_ind / (n_ind * (n_ind - 1) / 2.0))
    EWR = H * R
    NED = (H / n_ind) * R
    return EWR, NED, H, R, n_ind


def pearson_corr(x, y) -> float:
    """Sample Pearson correlation (ddof=1). Returns NaN if degenerate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n != y.size or n < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = x.std(ddof=1) * y.std(ddof=1)
    if denom == 0:
        return float("nan")
    return float(np.dot(xm, ym) / ((n - 1) * denom))
