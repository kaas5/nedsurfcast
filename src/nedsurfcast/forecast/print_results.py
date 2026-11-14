"""
Pretty-print the score tables of the spectrum correction model
================================================================

Reads the CSV files written by forecast/train_spectrum.py (fcfg.SPECTRUM_MODEL_DIR) and prints them
per validation scheme, with readable column names and units. Skill is shown in percent, and the
best value per column is marked with *.

    spectrum_scores.csv   errors of the corrected spectrum against the buoy (lower is better)
    spectrum_skill.csv    1 - RMSE_model / RMSE_GFS (higher is better; GFS = 0 %)
    system_scores.csv     wave systems >= 0.3 m after partitioning, matched against the buoy systems
    swell_hm0_by_lead.csv RMSE of swell Hm0 (Tp > 6 s) per lead time

The first three tables are over lead times >= 3 h: models with the issue-time measurement get the
target itself as input at lead 0.

Run from anywhere with: python -m nedsurfcast.forecast.print_results
"""

import pandas as pd

from nedsurfcast.forecast import config as fcfg

# column -> (header, format, which value is best: "low", "high" or None)
SPECTRUM_COLUMNS = {
    "rmse_log10_energy": ("log10 E", "{:.3f}", "low"),
    "rmse_hm0": ("Hm0 (m)", "{:.3f}", "low"),
    "rmse_hm0 Tp > 10 s": ("Hm0 >10 s", "{:.3f}", "low"),
    "rmse_hm0 6-10 s": ("Hm0 6-10 s", "{:.3f}", "low"),
    "rmse_hm0 Tp < 6 s": ("Hm0 <6 s", "{:.3f}", "low"),
    "swell_dir_error": ("swell dir (deg)", "{:.1f}", "low"),
}
SKILL_COLUMNS = {k: (v[0], "{:+.1%}", "high") for k, v in SPECTRUM_COLUMNS.items() if k != "swell_dir_error"}
SYSTEM_COLUMNS = {
    "hit_rate": ("hit rate", "{:.1%}", "high"),
    "false_alarm_rate": ("false alarms", "{:.1%}", "low"),
    "n_pairs": ("pairs", "{:,.0f}", None),
    "rmse_hs": ("Hs (m)", "{:.3f}", "low"),
    "bias_hs": ("bias Hs (m)", "{:+.3f}", None),
    "rmse_tp": ("Tp (s)", "{:.2f}", "low"),
    "mae_dir": ("dir (deg)", "{:.1f}", "low"),
    "rmse_hs_swell": ("swell Hs (m)", "{:.3f}", "low"),
    "rmse_tp_swell": ("swell Tp (s)", "{:.2f}", "low"),
    "mae_dir_swell": ("swell dir (deg)", "{:.1f}", "low"),
}

TABLES = [
    ("spectrum_scores.csv", "Spectrum errors against the buoy (RMSE, lower is better)", SPECTRUM_COLUMNS),
    ("spectrum_skill.csv", "Skill vs GFS: 1 - RMSE_model / RMSE_GFS (higher is better)", SKILL_COLUMNS),
    ("system_scores.csv", "Wave systems >= 0.3 m after partitioning (swell: buoy Tp >= 6 s)", SYSTEM_COLUMNS),
    ("swell_hm0_by_lead.csv", "RMSE of swell Hm0 (Tp > 6 s) per lead time (m, lower is better)", None),
]


def format_table(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """Format the known columns and mark the best model per column with *."""
    out = pd.DataFrame(index=df.index)
    for col, (header, fmt, best) in columns.items():
        if col not in df:
            continue
        values = df[col]
        text = values.map(lambda v: "-" if pd.isna(v) else fmt.format(v))
        if best is not None and values.notna().sum() > 1:
            target = values.min() if best == "low" else values.max()
            text = text.where(values != target, text + " *")
        out[header] = text
    return out


def print_table(path, title: str, columns: dict):
    df = pd.read_csv(path, index_col=["scheme", "model"])
    if columns is None:                       # every column is the same kind of score
        columns = {c: (c, "{:.3f}", "low") for c in df.columns}
    unknown = [c for c in df.columns if c not in columns]
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")
    for scheme, part in df.groupby(level="scheme", sort=False):
        print(f"\n  {scheme}")
        table = format_table(part.droplevel("scheme"), columns)
        table.index.name = None
        print("    " + table.to_string(justify="right").replace("\n", "\n    "))
    if unknown:
        print(f"\n  (columns not shown: {unknown})")


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    out_dir = fcfg.SPECTRUM_MODEL_DIR
    print(f"Results in {out_dir}")
    for name, title, columns in TABLES:
        path = out_dir / name
        if path.exists():
            print_table(path, title, columns)
        else:
            print(f"\n{name}: not found, run python -m nedsurfcast.forecast.train_spectrum first")
    print("\n* = best value in the column for that scheme")


if __name__ == "__main__":
    main()
