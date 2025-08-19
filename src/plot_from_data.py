#!/usr/bin/env python
import argparse, json, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from pathlib import Path

def wilson_ci(wins, n, z=1.96):
    if n <= 0: return (np.nan, np.nan)
    p = wins / n
    denom = 1 + z**2 / n
    centre = p + z**2/(2*n)
    half = z*sqrt(p*(1-p)/n + z**2/(4*n**2))
    return (centre - half)/denom, (centre + half)/denom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", type=str, default="data")
    ap.add_argument("--out", dest="outdir", type=str, default="figs")
    args = ap.parse_args()
    indir = Path(args.indir); outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read data
    comp = json.loads(Path(indir/"comprehensive_tournament_results.json").read_text())
    games = pd.DataFrame(comp["games"])
    games["avg_time_per_move"] = games["game_duration"] / games["moves_count"]
    games["nodes_per_move"] = games["nodes_explored"] / games["moves_count"]
    games["prune_ratio"] = games["alpha_beta_cutoffs"] / games["nodes_explored"]

    def stage_from_moves(m):
        if m <= 14: return "Opening"
        if m <= 28: return "Mid-game"
        return "Endgame"
    games["stage"] = games["moves_count"].apply(stage_from_moves)

    def winner_heur(row):
        if row["winner"] == "RED": return row["red_player"]
        if row["winner"] == "YELLOW": return row["yellow_player"]
        return "DRAW"
    games["winner_heuristic"] = games.apply(winner_heur, axis=1)

    heuristics = sorted(set(games["red_player"]).union(set(games["yellow_player"])))

    # Per-depth performance (exclude draws)
    rows = []
    for d, df_d in games.groupby("search_depth"):
        for h in heuristics:
            df_h = df_d[(df_d["red_player"]==h) | (df_d["yellow_player"]==h)]
            wins = (df_h["winner_heuristic"]==h).sum()
            draws = (df_h["winner_heuristic"]=="DRAW").sum()
            losses = len(df_h) - wins - draws
            n = wins + losses
            wr = wins/n if n>0 else np.nan
            lo, hi = wilson_ci(wins, n) if n>0 else (np.nan, np.nan)
            rows.append(dict(depth=d, heuristic=h, wins=wins, losses=losses, draws=draws,
                             win_rate=wr, ci_low=lo, ci_high=hi, n=n))
    depth_df = pd.DataFrame(rows)

    # 1) Win rate by depth
    plt.figure(figsize=(7,5))
    for h in heuristics:
        sub = depth_df[depth_df["heuristic"]==h].sort_values("depth")
        depths = sub["depth"].values
        wr = sub["win_rate"].values
        yerr = np.vstack([np.maximum(0, wr - sub["ci_low"].values),
                          np.maximum(0, sub["ci_high"].values - wr)])
        plt.errorbar(depths, wr, yerr=yerr, marker="o", capsize=3, label=h)
    plt.xlabel("Search Depth (plies)"); plt.ylabel("Win Rate (excl. draws)")
    plt.title("Win Rate by Heuristic Across Depths"); plt.ylim(0, 1.05)
    plt.xticks(sorted(games["search_depth"].unique())); plt.legend(title="Heuristic", loc="best")
    plt.tight_layout(); plt.savefig(outdir/"fig_winrate_by_depth.png", dpi=180)

    # 2) Cost–benefit: time vs win rate
    perf_rows = []
    for d, df_d in games.groupby("search_depth"):
        for h in heuristics:
            df_h = df_d[(df_d["red_player"]==h) | (df_d["yellow_player"]==h)]
            wins = (df_h["winner_heuristic"]==h).sum()
            draws = (df_h["winner_heuristic"]=="DRAW").sum()
            losses = len(df_h) - wins - draws
            n = wins + losses
            wr = wins/n if n>0 else np.nan
            perf_rows.append(dict(depth=d, heuristic=h, win_rate=wr,
                                  avg_time_per_move=df_h["avg_time_per_move"].mean(),
                                  avg_nodes_per_move=df_h["nodes_per_move"].mean(),
                                  prune_ratio=(df_h["alpha_beta_cutoffs"].sum()/df_h["nodes_explored"].sum()
                                               if df_h["nodes_explored"].sum()>0 else np.nan)))
    perf = pd.DataFrame(perf_rows)
    plt.figure(figsize=(7,5))
    for h in heuristics:
        sub = perf[perf["heuristic"]==h]
        plt.scatter(sub["avg_time_per_move"], sub["win_rate"], label=h)
        for _, r in sub.iterrows():
            plt.annotate(f"d{int(r['depth'])}", (r["avg_time_per_move"], r["win_rate"]),
                         textcoords="offset points", xytext=(4,4), fontsize=8)
    plt.xlabel("Average Time per Move (s)"); plt.ylabel("Win Rate (excl. draws)")
    plt.title("Cost–Benefit: Time per Move vs Strength")
    plt.legend(title="Heuristic", loc="best"); plt.tight_layout()
    plt.savefig(outdir/"fig_cost_benefit_time.png", dpi=180)

    # 3) Stage-specific win rates
    stage_results = (games[games["winner_heuristic"]!="DRAW"]
                        .groupby(["stage","winner_heuristic"]).size().unstack(fill_value=0))
    def appearances_by_stage(df, heuristic):
        return ((df["red_player"]==heuristic) | (df["yellow_player"]==heuristic)).groupby(df["stage"]).sum()
    stage_appear = pd.DataFrame({h: appearances_by_stage(games, h) for h in heuristics})
    stage_wr = (stage_results.reindex(columns=heuristics, fill_value=0) / stage_appear).reset_index()
    stage_order = ["Opening","Mid-game","Endgame"]
    stage_wr["stage"] = pd.Categorical(stage_wr["stage"], categories=stage_order, ordered=True)

    plt.figure(figsize=(8,5))
    width = 0.18; x = np.arange(len(stage_order))
    for i, h in enumerate(heuristics):
        sub = stage_wr[stage_wr["stage"].notna()][["stage", h]].sort_values("stage")
        plt.bar(x + (i-1.5)*width, sub[h].values, width=width, label=h)
    plt.xticks(x, stage_order); plt.ylim(0, 1.05); plt.ylabel("Win Rate (excl. draws)")
    plt.title("Stage-Specific Win Rates by Heuristic"); plt.legend(title="Heuristic", ncol=2)
    plt.tight_layout(); plt.savefig(outdir/"fig_stage_winrates.png", dpi=180)

    # 4) Nodes per move by depth
    plt.figure(figsize=(7,5))
    for h in heuristics:
        sub = perf[perf["heuristic"]==h].sort_values("depth")
        plt.plot(sub["depth"], sub["avg_nodes_per_move"], marker="o", label=h)
    plt.xlabel("Search Depth (plies)"); plt.ylabel("Nodes per Move (avg)")
    plt.title("Search Effort by Heuristic and Depth"); plt.xticks(sorted(games["search_depth"].unique()))
    plt.legend(title="Heuristic"); plt.tight_layout()
    plt.savefig(outdir/"fig_nodes_per_move_by_depth.png", dpi=180)

    # optional: prune ratio by depth
    plt.figure(figsize=(7,5))
    for h in heuristics:
        sub = perf[perf["heuristic"]==h].sort_values("depth")
        plt.plot(sub["depth"], sub["prune_ratio"], marker="o", label=h)
    plt.xlabel("Search Depth (plies)"); plt.ylabel("Alpha–Beta Prune Ratio (cutoffs / nodes)")
    plt.title("Alpha–Beta Effectiveness by Heuristic and Depth")
    plt.xticks(sorted(games["search_depth"].unique())); plt.legend(title="Heuristic")
    plt.tight_layout(); plt.savefig(outdir/"fig_prune_ratio_by_depth.png", dpi=180)

if __name__ == "__main__":
    main()
