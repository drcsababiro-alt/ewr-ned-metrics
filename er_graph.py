"""
ER graph experiment: computes EWR/NED on an Erdős–Rényi G(n,p) graph
and produces CSV + basic figures as in the manuscript.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from utils.common import ewr_ned_for_node, pearson_corr

# ---------------- Parameters ----------------
n        = 1000       # number of nodes
p        = 0.01       # edge probability
kmain    = 2          # k-hop
seed     = 42         # RNG seed
outdir   = "./er_core_figs"
btw_k    = 150        # betweenness approx samples for speed
# --------------------------------------------

os.makedirs(outdir, exist_ok=True)
csvdir = os.path.join(outdir, "csv")
os.makedirs(csvdir, exist_ok=True)

rng = np.random.default_rng(seed)
G = nx.fast_gnp_random_graph(n, p, seed=int(rng.integers(1, 1_000_000)))

deg = dict(G.degree())
clu = nx.clustering(G)
btw = nx.betweenness_centrality(G, k=btw_k, seed=seed, normalized=True)

EWR = {}
NED = {}
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
df.to_csv(os.path.join(csvdir, f"er_n{n}_p{str(p).replace('.','p')}_core.csv"), index=False)

# Histograms
plt.figure(figsize=(7,5))
plt.hist(df[f"EWR_k{kmain}"], bins=40, alpha=0.6, label=f"EWR (k={kmain})")
plt.hist(df[f"NED_k{kmain}"], bins=40, alpha=0.6, label=f"NED (k={kmain})")
plt.title(f"ER(n={n}, p={p}): EWR & NED distributions (k={kmain})")
plt.xlabel("Value"); plt.ylabel("Frequency"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig1_er_hist_ewr_ned.png"), dpi=200); plt.close()

# Degree vs metrics
r_deg_ewr = pearson_corr(df["degree"], df[f"EWR_k{kmain}"])
r_deg_ned = pearson_corr(df["degree"], df[f"NED_k{kmain}"])
plt.figure(figsize=(7,5))
plt.scatter(df["degree"], df[f"EWR_k{kmain}"], s=9, alpha=0.7, label=f"EWR (r={r_deg_ewr:.3f})")
plt.scatter(df["degree"], df[f"NED_k{kmain}"], s=9, alpha=0.7, marker='x', label=f"NED (r={r_deg_ned:.3f})")
plt.xlabel("Degree"); plt.ylabel("Metric value")
plt.title(f"Degree vs EWR/NED (k={kmain})"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig2_er_scatter_degree_both.png"), dpi=200); plt.close()

# Betweenness vs metrics
r_btw_ewr = pearson_corr(df["betweenness_approx"], df[f"EWR_k{kmain}"])
r_btw_ned = pearson_corr(df["betweenness_approx"], df[f"NED_k{kmain}"])
plt.figure(figsize=(7,5))
plt.scatter(df["betweenness_approx"], df[f"EWR_k{kmain}"], s=9, alpha=0.7, label=f"EWR (r={r_btw_ewr:.3f})")
plt.scatter(df["betweenness_approx"], df[f"NED_k{kmain}"], s=9, alpha=0.7, marker='x', label=f"NED (r={r_btw_ned:.3f})")
plt.xlabel("Betweenness (approx)"); plt.ylabel("Metric value")
plt.title(f"Betweenness vs EWR/NED (k={kmain})"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig3_er_scatter_betweenness_both.png"), dpi=200); plt.close()

print("Done. Saved to:", outdir)
