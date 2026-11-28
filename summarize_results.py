import glob
import math
import os

import numpy as np
import pandas as pd

from glicko2_calc import LILA_RATING_PERIOD_DAYS

COHORTS = [
    ("Game-Weighted (All Games)", "game_ll", "game_brier", "eval_games", "games"),
    ("Player-Weighted: Established (Top)", "est_ll", "est_brier", "est_players", "players"),
    ("Player-Weighted: Rankable (Top)", "rankable_ll", "rankable_brier", "rankable_players", "players"),
    ("Player-Weighted: Established (Locked)", "est_locked_ll", "est_locked_brier", "est_locked_players", "players"),
    ("Player-Weighted: Rankable (Locked)", "rankable_locked_ll", "rankable_locked_brier", "rankable_locked_players", "players"),
]


def delta_elo(delta_ll: float) -> float:
    if delta_ll <= 0.0 or math.isnan(delta_ll):
        return 0.0
    return (400.0 / math.log(10.0)) * math.sqrt(2.0 * delta_ll)


def main():
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    csv_files = sorted(glob.glob("**/*_sweep.csv", recursive=True))
    if not csv_files:
        msg = "No `*_sweep.csv` artifacts were produced by any sweep job.\n"
        print(msg)
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(msg)
        return

    lines: list[str] = ["## Combined Glicko-2 Sweep Summary\n\n"]
    lines.append("| Dataset | Cohort | Units | #1 Period | #1 Log-Loss | #1 Brier | Lila (`4.665d`) Δ Elo |\n")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")

    for path in csv_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        dataset = str(df["dataset"].iloc[0])
        periods = df["period_days"].to_numpy(dtype=float)
        lila_idx = int(np.argmin(np.abs(periods - LILA_RATING_PERIOD_DAYS)))

        for title, ll_col, bs_col, cnt_col, unit_lbl in COHORTS:
            if ll_col not in df.columns:
                continue
            ll = df[ll_col].to_numpy(dtype=float)
            bs = df[bs_col].to_numpy(dtype=float)
            n_units = int(df[cnt_col].iloc[lila_idx])
            if n_units == 0 or np.all(np.isnan(ll)):
                continue
            best_idx = int(np.nanargmin(ll))
            d_elo = delta_elo(float(ll[lila_idx] - ll[best_idx]))
            lines.append(f"| `{dataset}` | {title} | {n_units:,} {unit_lbl} | `{periods[best_idx]:.3f}d` | `{ll[best_idx]:.6f}` | `{bs[best_idx]:.6f}` | `+{d_elo:.2f} Elo` |\n")

    output = "".join(lines)
    print(output)
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(output)


if __name__ == "__main__":
    main()
