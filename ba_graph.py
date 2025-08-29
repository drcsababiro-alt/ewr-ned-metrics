"""
BA graph experiment: computes EWR/NED on a Barabási–Albert graph.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from utils.common import ewr_ned_for_node, pearson_corr

# -------------- Parameters --------------
n        = 1000  # nodes
m        = 3     # edges to attach per new node
kmain    = 2
seed     = 42
outdir   = "./ba_core_figs"
btw_k    = 150
# ----------------------------------------

os.makedirs(outdir, exist_ok=True)
csvdir = os.path.join(outdir, "csv")
os.makedirs(csvdir, exist_ok=True)

rng = np.random.default_rng(seed)
G = nx.barabasi_albert_graph(n=n, m=m, seed=int(rng.integers(1, 1_000_000)))

deg = dict(G.degree())
clu = nx.clustering(G)
btw = nx.betweenness_centrality(G, k=btw_k, seed=seed, normalized=True)

EWR = {}; NED = {}
for v in G.nodes():
    e, n, *_ = ewr_ned_for_node(G, v, k=kmain)
    EWR[v] = e; NED[v] = n

df = pd.DataFrame({
    "node": list(G.nodes()),
    "degree": [deg[v] for v in G.nodes()],
    "clustering": [clu[v] for v in G.nodes()],
    "betweenness_approx": [btw[v] for v in G.nodes()],
    f"EWR_k{kmain}": [EWR[v] for v in G.nodes()],
    f"NED_k{kmain}": [NED[v] for v in G.nodes()],
})
df.to_csv(os.path.join(csvdir, f"ba_n{n}_m{m}_core.csv"), index=False)

# Degree vs metrics
r_deg_ewr = pearson_corr(df["degree"], df[f"EWR_k{kmain}"])
r_deg_ned = pearson_corr(df["degree"], df[f"NED_k{kmain}"])

# Betweenness vs metrics
r_btw_ewr = pearson_corr(df["betweenness_approx"], df[f"EWR_k{kmain}"])
r_btw_ned = pearson_corr(df["betweenness_approx"], df[f"NED_k{kmain}"])

plt.figure(figsize=(7,5))
plt.scatter(df["degree"], df[f"EWR_k{kmain}"], s=9, alpha=0.7, label=f"EWR (r={r_deg_ewr:.3f})")
plt.scatter(df["degree"], df[f"NED_k{kmain}"], s=9, alpha=0.7, marker='x', label=f"NED (r={r_deg_ned:.3f})")
plt.xlabel("Degree"); plt.ylabel("Metric value"); plt.title("BA: Degree vs EWR/NED"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig_ba_degree.png"), dpi=200); plt.close()

plt.figure(figsize=(7,5))
plt.scatter(df["betweenness_approx"], df[f"EWR_k{kmain}"], s=9, alpha=0.7, label=f"EWR (r={r_btw_ewr:.3f})")
plt.scatter(df["betweenness_approx"], df[f"NED_k{kmain}"], s=9, alpha=0.7, marker='x', label=f"NED (r={r_btw_ned:.3f})")
plt.xlabel("Betweenness (approx)"); plt.ylabel("Metric value"); plt.title("BA: Betweenness vs EWR/NED"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig_ba_betweenness.png"), dpi=200); plt.close()

print("Done. Saved to:", outdir)
