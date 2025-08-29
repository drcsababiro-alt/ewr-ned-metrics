"""
RGG experiment: computes EWR/NED on a Random Geometric Graph in the unit square.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from utils.common import ewr_ned_for_node, pearson_corr

# ---------- Parameters ----------
n        = 1000
radius   = 0.06   # connection radius in unit square
seed     = 42
kmain    = 2
btw_k    = 150
outdir   = "./rgg_core_figs"
# --------------------------------

os.makedirs(outdir, exist_ok=True)
csvdir = os.path.join(outdir, "csv")
os.makedirs(csvdir, exist_ok=True)

rng = np.random.default_rng(seed)
pos = {i: rng.random(2) for i in range(n)}  # positions in [0,1]^2
G = nx.random_geometric_graph(n, radius, pos=pos)

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
df.to_csv(os.path.join(csvdir, f"rgg_n{n}_r{str(radius).replace('.','p')}_core.csv"), index=False)

r_deg_ewr = pearson_corr(df["degree"], df[f"EWR_k{kmain}"])
r_deg_ned = pearson_corr(df["degree"], df[f"NED_k{kmain}"])
r_btw_ewr = pearson_corr(df["betweenness_approx"], df[f"EWR_k{kmain}"])
r_btw_ned = pearson_corr(df["betweenness_approx"], df[f"NED_k{kmain}"])

plt.figure(figsize=(7,5))
plt.scatter(df["degree"], df[f"EWR_k{kmain}"], s=9, alpha=0.7, label=f"EWR (r={r_deg_ewr:.3f})")
plt.scatter(df["degree"], df[f"NED_k{kmain}"], s=9, alpha=0.7, marker='x', label=f"NED (r={r_deg_ned:.3f})")
plt.xlabel("Degree"); plt.ylabel("Metric value"); plt.title("RGG: Degree vs EWR/NED"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig_rgg_degree.png"), dpi=200); plt.close()

plt.figure(figsize=(7,5))
plt.scatter(df["betweenness_approx"], df[f"EWR_k{kmain}"], s=9, alpha=0.7, label=f"EWR (r={r_btw_ewr:.3f})")
plt.scatter(df["betweenness_approx"], df[f"NED_k{kmain}"], s=9, alpha=0.7, marker='x', label=f"NED (r={r_btw_ned:.3f})")
plt.xlabel("Betweenness (approx)"); plt.ylabel("Metric value"); plt.title("RGG: Betweenness vs EWR/NED"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig_rgg_betweenness.png"), dpi=200); plt.close()

print("Done. Saved to:", outdir)
