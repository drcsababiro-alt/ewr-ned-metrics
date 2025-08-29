# EWR/NED Experiments — Reproducibility Package

This repository contains reference Python implementations and scripts to reproduce the
experiments reported in the manuscript (EWR = Entropy-Weighted Redundancy, NED = Normalized Entropy Density).
All scripts compute the metrics on $k$-hop neighbourhoods and export CSVs and figures mirroring the paper.

> Tip: default parameters are aligned with the paper (e.g., $n=1000$, $k=2$). Random seeds are fixed.

## Folder structure
```
utils/
  common.py          # k-hop BFS, entropy, redundancy, EWR/NED, pearson corr
er_graph.py          # Erdős–Rényi (G(n,p))
ba_graph.py          # Barabási–Albert
ws_graph.py          # Watts–Strogatz
rgg_graph.py         # Random Geometric Graph (unit square)
zephyr_graph.py      # D-Wave Zephyr architecture (requires dwave-networkx)
er_density_sweep.py  # density sweep for ER: p vs mean/std of EWR/NED
```

## Requirements
- Python >= 3.10
- `networkx`, `numpy`, `pandas`, `matplotlib`
- `dwave-networkx` (only for `zephyr_graph.py`)

Install:
```bash
pip install networkx numpy pandas matplotlib
pip install dwave-networkx  # only if running Zephyr
```

## How to run
From the repo root:
```bash
python er_graph.py
python ba_graph.py
python ws_graph.py
python rgg_graph.py
python zephyr_graph.py
python er_density_sweep.py
```
Each script creates an output folder (e.g., `*_core_figs/` or `er_density/`), with CSVs and PNG figures.

## Citation
If you use this code, please cite the accompanying article:

> Cs. Biró (2025). Hybrid Entropy-Based Metrics for k-Hop Environment Analysis
in Complex Networks *[Journal/Preprint]*.
> (Replace with the exact final citation once available.)


## License
MIT (or adjust to your preferred license).
