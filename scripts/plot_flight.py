"""Plot key signals of one flight from the cleaned lake layer, with labelled
anomaly ranges shaded.

Usage:
    uv run python scripts/plot_flight.py --lake data/lake \
        --key 2019-01-15/07_28_00 --out reports/phase3
"""

import argparse
from pathlib import Path

import duckdb
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aeropulse.features import row_derived
from aeropulse.signals import MAX_MOTORS

# Reference palette (categorical slots 1-3) and text inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT, MUTED, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#52514e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scores", type=Path, default=None,
                    help="optional scores parquet (models.apply) to add a detector panel")
    ap.add_argument("--xlim", type=float, nargs=2, default=None)
    args = ap.parse_args()

    con = duckdb.connect(str(args.lake / "aeropulse.duckdb"), read_only=True)
    fid, cls, n_motors = con.execute(
        "select flight_id, flight_class, n_motors from flights where key = ?", [args.key]).fetchone()
    df = con.execute("select * from cleaned where flight_id = ? order by t_us", [fid]).fetchdf()
    ranges = con.execute("select start_us, end_us, signal from anomaly_ranges "
                         "where flight_id = ?", [fid]).fetchall()
    con.close()

    d = row_derived(df)
    t, t0 = df["t_s"], df["t_us"].iloc[0]
    panels = [
        (f"Motor commands across {int(n_motors)} motors (normalised 0-1)",
         [(d["m_max"], "max"), (d["m_mean"], "mean"), (d["m_min"], "min")]),
        ("Thrust command (0-1)", [(df["ctrl_thrust"], "thrust")]),
        ("Tilt (deg)", [(df["tilt"] * 57.2958, "tilt")]),
        ("Altitude, local (m, up)", [(d["alt"], "altitude")]),
    ]
    alarm_t = None
    if args.scores:
        sc = pd.read_parquet(args.scores)
        sc = sc[sc["flight_id"] == fid].sort_values("t_us")
        ts = (sc["t_us"] - t0) / 1e6
        panels.append(("Motor-fault probability (gbt_mechanical); dots = alarm",
                       [(pd.Series(sc["p_mech"].to_numpy(), index=ts), "p")]))
        alarm_t = ts[sc["alarm_gbt"].to_numpy()]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 1.9 * len(panels) + 0.6), sharex=True,
                             facecolor=SURFACE, constrained_layout=True)
    for ax, (title, series) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        for a, b, _ in ranges:
            ax.axvspan((a - t0) / 1e6, (b - t0) / 1e6, color=BAND, alpha=0.12, lw=0)
        for i, (y, label) in enumerate(series):
            x = y.index if label == "p" else t
            ax.plot(x, y.to_numpy(), color=SERIES[i], lw=1.5, label=label)
            if label == "p" and alarm_t is not None and len(alarm_t):
                ax.plot(alarm_t, [1.02] * len(alarm_t), "o", ms=3, color=SERIES[1])
        ax.set_title(title, loc="left", fontsize=9, color=TEXT)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if len(series) > 1:
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=len(series),
                      frameon=False, fontsize=8, borderaxespad=0.2)
    axes[-1].set_xlabel("Time (s)")
    if args.xlim:
        axes[-1].set_xlim(*args.xlim)
    signals = sorted({s for _, _, s in ranges}) or ["none"]
    fig.suptitle(f"{cls} flight {args.key}: shaded = labelled anomaly ({', '.join(signals)})",
                 x=0.01, ha="left", fontsize=10, color=TEXT)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"flight_{args.key.replace('/', '_')}.png"
    fig.savefig(path, dpi=130)
    print(path)


if __name__ == "__main__":
    assert MAX_MOTORS >= 1
    main()
