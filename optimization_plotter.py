# -*- coding: utf-8 -*-
"""
Per-trial plotting for multiple methods listed in file_info.
- For each method, creates one PNG per trial with two subplots:
  (top) Laplacian, (bottom) Loss-like column.
- Output structure:
  OUTPUT_ROOT/
    ForceEquillibrium/trial_001.png, ...
    Hybrid/trial_001.png, ...
    RegularizedHybrid/...
    Conventional/...
    RegularizedConventional/...
"""

import os
import numpy as np
import pandas as pd

# ----------------- INPUT: method -> CSV absolute path -----------------
file_info = {
    'ForceEquillibrium': "/content/inputs/force_equillibrium_result.csv",
    'Hybrid': "/content/inputs/hybrid_result.csv",
    'RegularizedHybrid': "/content/inputs/regularized_hybrid_result.csv",
    'Conventional': "/content/inputs/conventional_result.csv",
    'RegularizedConventional': "/content/inputs/regularized_conventional_result.csv"
}

# ----------------- CONFIG (수정 가능; 절대경로 권장) -----------------
CONFIG = {
    "OUTPUT_ROOT": r"/content/results_per_trial_plots",  # 메서드별 하위 폴더 생성
    "DPI": 200,
    "WIDTH": 12.0,
    "HEIGHT": 6.0,
}

# GUI 미사용 백엔드
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

LOSS_CANDS = ["loss", "loss_value", "objective", "lossValue"]

def _validate_abs_path(path: str, what: str):
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError(f"{what}는 절대경로여야 합니다: {path}")

def _pick_loss_col(df: pd.DataFrame):
    for c in LOSS_CANDS:
        if c in df.columns:
            return c
    return None

def _plot_one_trial_group(g: pd.DataFrame, trial_id: int, out_path: str,
                          dpi=150, figsize=(12.0, 6.0)):
    step = pd.to_numeric(g["step"], errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, constrained_layout=True)
    ax1, ax2 = axes

    # Laplacian
    if "laplacian" in g.columns:
        y1 = pd.to_numeric(g["laplacian"], errors="coerce")
        m1 = step.notna() & y1.notna()
        if m1.any():
            ax1.plot(step[m1].to_numpy(), y1[m1].to_numpy(), linewidth=1.0)
            ax1.set_title(f"Trial {int(trial_id)} — Laplacian")
            ax1.set_ylabel("laplacian")
        else:
            ax1.text(0.5, 0.5, "No valid laplacian data", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_title(f"Trial {int(trial_id)} — Laplacian")
    else:
        ax1.text(0.5, 0.5, "Column 'laplacian' not found", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title(f"Trial {int(trial_id)} — Laplacian")

    # Loss-like
    loss_col = _pick_loss_col(g)
    if loss_col is not None:
        y2 = pd.to_numeric(g[loss_col], errors="coerce")
        m2 = step.notna() & y2.notna()
        if m2.any():
            ax2.plot(step[m2].to_numpy(), y2[m2].to_numpy(), linewidth=1.0)
            ax2.set_title(f"Trial {int(trial_id)} — {loss_col}")
            ax2.set_ylabel(loss_col)
        else:
            ax2.text(0.5, 0.5, f"No valid {loss_col} data", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_title(f"Trial {int(trial_id)} — {loss_col}")
    else:
        ax2.text(0.5, 0.5, "No loss-like column found", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title(f"Trial {int(trial_id)} — loss")

    axes[-1].set_xlabel("Step")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def process_one_method(method_name: str, csv_path: str, out_root: str, dpi=150, figsize=(12.0, 6.0)):
    # 경로 검증
    _validate_abs_path(csv_path, "CSV 경로")
    _validate_abs_path(out_root, "OUTPUT_ROOT")
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV 파일이 존재하지 않습니다: {csv_path}")
        return

    # 출력 폴더
    out_dir = os.path.join(out_root, method_name)
    os.makedirs(out_dir, exist_ok=True)

    # 데이터 로드
    df = pd.read_csv(csv_path)
    for col in ["trial", "step"]:
        if col not in df.columns:
            raise ValueError(f"[{method_name}] CSV에 '{col}' 열이 없습니다: {csv_path}")

    # 정렬 및 trial 집계
    df = df.sort_values(["trial", "step"]).reset_index(drop=True)
    trials = df["trial"].dropna().unique()
    n_trials = len(trials)
    pad = max(3, len(str(int(n_trials))))  # 파일명 자리수

    saved = 0
    for trial, g in df.groupby("trial", sort=True):
        g = g.copy()
        g["step"] = pd.to_numeric(g["step"], errors="coerce")
        g = g.sort_values("step")

        out_path = os.path.join(out_dir, f"trial_{int(trial):0{pad}d}.png")
        _plot_one_trial_group(
            g=g,
            trial_id=int(trial),
            out_path=out_path,
            dpi=dpi,
            figsize=figsize
        )
        saved += 1

    print(f"[OK] {method_name}: {saved} plot(s) saved -> {out_dir}")

def main():
    out_root = CONFIG["OUTPUT_ROOT"]
    dpi = CONFIG["DPI"]
    figsize = (CONFIG["WIDTH"], CONFIG["HEIGHT"])

    os.makedirs(out_root, exist_ok=True)
    _validate_abs_path(out_root, "OUTPUT_ROOT")

    for method_name, csv_path in file_info.items():
        try:
            process_one_method(method_name, csv_path, out_root, dpi=dpi, figsize=figsize)
        except Exception as e:
            print(f"[ERROR] {method_name}: {e}")

if __name__ == "__main__":
    main()
