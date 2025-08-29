"""
ER density sweep: vary p and track mean/std of EWR/NED to reproduce density trends.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from utils.common import ewr_ned_for_node

# ---------- Parameters ----------
n        = 1000
k        = 2
p_values = np.linspace(0.002, 0.1, 12)
seed     = 42
outdir   = "./er_density"
# --------------------------------

os.makedirs(outdir, exist_ok=True)
rng = np.random.default_rng(seed)

rows = []
for p in p_values:
    G = nx.fast_gnp_random_graph(n, p, seed=int(rng.integers(1, 1_000_000)))
    EWR = []; NED = []
    for v in G.nodes():
        e, n, *_ = ewr_ned_for_node(G, v, k)
        EWR.append(e); NED.append(n)
    EWR = np.array(EWR); NED = np.array(NED)
    rows.append((float(p), float(EWR.mean()), float(EWR.std()), float(NED.mean()), float(NED.std())))
    print(f"p={p:.4f} | EWR mean={EWR.mean():.4f} std={EWR.std():.4f} | NED mean={NED.mean():.4f} std={NED.std():.4f}")

import pandas as pd
df = pd.DataFrame(rows, columns=["p", "ewr_mean", "ewr_std", "ned_mean", "ned_std"])
df.to_csv(os.path.join(outdir, "er_density_sweep.csv"), index=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(df["p"], df["ewr_mean"], marker="o", label="EWR mean")
plt.plot(df["p"], df["ned_mean"], marker="s", label="NED mean")
plt.xlabel("Edge probability p (≈ density)"); plt.ylabel("Mean metric value")
plt.title(f"ER(n={n}) — k={k}: EWR & NED means vs density")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig_er_density_means.png"), dpi=160); plt.close()

plt.figure(figsize=(8,5))
plt.plot(df["p"], df["ewr_std"], marker="o", label="EWR std")
plt.plot(df["p"], df["ned_std"], marker="s", label="NED std")
plt.xlabel("Edge probability p (≈ density)"); plt.ylabel("Standard deviation")
plt.title(f"ER(n={n}) — k={k}: EWR & NED std vs density")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(outdir, "fig_er_density_stds.png"), dpi=160); plt.close()

print("Done. Saved to:", outdir)
