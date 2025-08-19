# Connect Four Heuristics Tournament

A reproducible benchmark comparing four evaluation heuristics for Connect Four under minimax + alpha–beta pruning:

- **Material**
- **Positional**
- **Threat-based**
- **Pattern Recognition**

We run depth-limited search at `d ∈ {3, 5, 7}` in a balanced round-robin (180 total games). All agents share the same search, move ordering, and logging; **only the evaluation function changes**.

## Key Findings (from the paper)
Material is strongest overall and wins at depths 3 and 7; Positional peaks at depth 5 and provides the best efficiency–strength trade-off at moderate depth. Pattern is mid-tier across settings, and Threat underperforms even at shallow depth. Alpha–beta prune ratios rise with depth across all heuristics, but pruning alone does not predict strength—evaluation quality and its interaction with move ordering do.

## Repository Structure
```
.
├─ src/
│  ├─ resarch_tornament.py        # tournament runner (your script)
│  └─ plot_from_data.py           # regenerate the 4 paper figures from data/
├─ data/
│  ├─ comprehensive_tournament_results.json
│  ├─ tournament_analysis.json
│  └─ tournament_results_summary.csv
├─ figs/
│  ├─ fig_winrate_by_depth.png
│  ├─ fig_cost_benefit_time.png
│  ├─ fig_stage_winrates.png
│  ├─ fig_nodes_per_move_by_depth.png
│  └─ fig_prune_ratio_by_depth.png   # optional for appendix
├─ docs/
│  └─ latex/
│     ├─ results.tex
│     └─ conclusions.tex
├─ requirements.txt
├─ environment.yml
├─ LICENSE (MIT)
├─ .gitignore
├─ Makefile
└─ .github/workflows/ci.yml
```

## Installation

### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate c4-heuristics
```

### pip + venv
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Optional: If you have CUDA and want GPU arrays for future extensions, adjust `cupy-*` in `environment.yml`. The baseline experiments here are CPU-only.

## Reproduce the Tournament

The tournament parameters and seeds live inside `src/resarch_tornament.py`.

```bash
python src/resarch_tornament.py
```

This script writes deterministic game logs, metrics, and aggregated JSON/CSV under `connect_four_research_output/` (or as configured in the script). To fully replicate the paper figures from the static data bundled in this repo, use the plotting helper below.



## Reproducibility Notes

- Identical search, move ordering, and alpha–beta across agents; only the evaluation function differs.
- Deterministic seeds are logged so all runs are reproducible.
- Depths tested: 3, 5, 7 plies (can be changed in the script if you want more).




