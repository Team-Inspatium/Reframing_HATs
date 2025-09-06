# -*- coding: utf-8 -*-
# 메서드별 CSV를 읽어, trial×10개 지표 수만큼 per-trial 플롯 생성 + 지표별 평균 진행도 플롯(10개) 생성
# - x축: trial 내부 진행도(0~100%) (각 trial의 step을 0~100%로 정규화)
# - y축: 각 지표 값
# - 출력 구조:
#   OUTPUT_ROOT/
#     Conventional/                 <- per-trial 플롯들 + 평균 플롯 10개
#     RegularizedConventional/
#     ForceEquillibrium/
#     Hybrid/
#     RegularizedHybrid/
#
# 주의: CSV에 지표 열이 누락되면 해당 지표는 스킵됩니다.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 입력 CSV 경로 (질문에서 제공)
# -----------------------------
file_info = {
    'ForceEquillibrium': "/content/inputs/force_equillibrium_result.csv",
    'Hybrid': "/content/inputs/hybrid_result.csv",
    'RegularizedHybrid': "/content/inputs/regularized_hybrid_result.csv",
    'Conventional': "/content/inputs/conventional_result.csv",
    'RegularizedConventional': "/content/inputs/regularized_conventional_result.csv"
}

# -----------------------------
# 출력 루트 폴더
# -----------------------------
OUTPUT_ROOT = "/content/outputs"  # 필요 시 변경
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# -----------------------------
# 플롯 대상 10개 지표 열 이름
# -----------------------------
METRICS = [
    "delta_tr_radius",
    "rho_lin",
    "gnorm",
    "pnorm",
    "alpha_imp",
    "cos_theta",
    "r_ratio",
    "sTy",
    "cos_phi",
    "gamma_scale",
]

# 공통 설정
PROGRESS_GRID = np.linspace(0.0, 100.0, 101)  # 0..100% (101점)
DPI = 150

def require_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

def interpolate_to_progress(step_vals: np.ndarray, y_vals: np.ndarray, progress_grid: np.ndarray):
    """
    step_vals: 1D (오름차순 가정), 길이 M>=1
    y_vals   : 1D, step_vals에 대응, 길이 M
    progress_grid: 0..100 %, 길이 K
    -> 선형보간 결과(길이 K) 반환
    """
    step_vals = np.asarray(step_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    if step_vals.size == 0 or y_vals.size == 0:
        return np.full_like(progress_grid, np.nan, dtype=float)

    s0 = float(step_vals[0])
    s1 = float(step_vals[-1])
    if s1 == s0:  # 스텝이 1개뿐이면 상수선
        return np.full_like(progress_grid, fill_value=float(y_vals[-1]), dtype=float)

    prog = (step_vals - s0) / (s1 - s0) * 100.0

    # 동일 진행도 값 중복 제거
    prog_u, idx = np.unique(prog, return_index=True)
    vals_u = y_vals[idx]

    # 유효값만 보간
    mask = np.isfinite(prog_u) & np.isfinite(vals_u)
    if mask.sum() == 0:
        return np.full_like(progress_grid, np.nan, dtype=float)
    if mask.sum() == 1:
        return np.full_like(progress_grid, float(vals_u[mask][0]), dtype=float)

    # np.interp는 구간 밖에서 양끝값으로 연장
    return np.interp(progress_grid, prog_u[mask], vals_u[mask])

def plot_single_line(progress_grid: np.ndarray, y_vals: np.ndarray,
                     title: str, ylabel: str, save_path: str):
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(progress_grid, y_vals, linewidth=1.2)
    plt.xlabel("Progress (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()

def process_one_method(method_name: str, csv_path: str):
    print(f"\n[INFO] Method: {method_name}")
    if not os.path.exists(csv_path):
        print(f"  [WARN] CSV가 존재하지 않습니다: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # trial, step 필수
    require_columns(df, ["trial", "step"])

    # 플롯 가능한 메트릭 선별
    available_metrics = [m for m in METRICS if m in df.columns]
    missing_metrics = [m for m in METRICS if m not in df.columns]
    if missing_metrics:
        print(f"  [WARN] 누락된 메트릭 열: {missing_metrics} (해당 플롯은 스킵)")

    # trial 목록
    trials = sorted(df["trial"].dropna().unique().tolist())
    if len(trials) == 0:
        print("  [WARN] trial 데이터가 없습니다.")
        return

    out_dir = os.path.join(OUTPUT_ROOT, method_name)
    os.makedirs(out_dir, exist_ok=True)

    total_plots = 0

    # ----------------------------
    # 1) per-trial 플롯 생성
    # ----------------------------
    for t in trials:
        dft = df[df["trial"] == t].copy()
        if dft.empty:
            continue
        dft = dft.sort_values("step")
        steps = dft["step"].to_numpy(dtype=float)

        for metric in available_metrics:
            ys = dft[metric].to_numpy(dtype=float)
            if not np.isfinite(ys).any():
                # 값이 전부 NaN이면 스킵
                continue

            y_interp = interpolate_to_progress(steps, ys, PROGRESS_GRID)
            fname = f"{metric}_trial_{int(t):03d}.png"
            fpath = os.path.join(out_dir, fname)
            plot_single_line(
                PROGRESS_GRID, y_interp,
                title=f"{method_name} | trial={int(t)} | {metric}",
                ylabel=metric,
                save_path=fpath
            )
            total_plots += 1

    print(f"  [DONE] per-trial 플롯 생성: {total_plots}개")

    # -----------------------------------
    # 2) 지표별 평균 진행도 플롯 생성(10개)
    # -----------------------------------
    avg_plots = 0
    for metric in available_metrics:
        # 각 trial에서 보간된 곡선들을 모아 평균
        curves = []
        for t in trials:
            dft = df[df["trial"] == t].copy()
            if dft.empty:
                continue
            dft = dft.sort_values("step")
            steps = dft["step"].to_numpy(dtype=float)
            ys = dft[metric].to_numpy(dtype=float)
            if np.isfinite(ys).any():
                curves.append(interpolate_to_progress(steps, ys, PROGRESS_GRID))

        if len(curves) == 0:
            continue

        M = np.vstack(curves)  # (n_trials, 101)
        mean_curve = np.nanmean(M, axis=0)  # trial 평균
        fname = f"mean_{metric}.png"
        fpath = os.path.join(out_dir, fname)
        plot_single_line(
            PROGRESS_GRID, mean_curve,
            title=f"{method_name} | mean over trials | {metric}",
            ylabel=f"{metric} (mean)",
            save_path=fpath
        )
        avg_plots += 1

    print(f"  [DONE] 평균 진행도 플롯 생성: {avg_plots}개 (메트릭 수={len(available_metrics)})")

def main():
    print(f"[OUT] 저장 경로: {os.path.abspath(OUTPUT_ROOT)}")
    for method_name, csv_path in file_info.items():
        try:
            process_one_method(method_name, csv_path)
        except Exception as e:
            print(f"[ERROR] {method_name}: {e}")

if __name__ == "__main__":
    main()