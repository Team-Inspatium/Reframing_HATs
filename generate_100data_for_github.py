# -*- coding: utf-8 -*-
# ============================================================
# BFGS + 지형 메트릭 로깅 + 결정공간 슬라이스 (로컬 실행용)
# - 매 스텝 CSV 로깅(phase 포함)
# - 3개 물리 단면(YZ/XZ/XY) + 최종 1D/2D 결정공간 슬라이스 저장
# - (간헐) Lanczos로 H의 극값 고유치 추정
# - 컬러바 스케일: XY(z=z0) 단면의 max를 YZ/XZ/XY 공통 vmax로 사용
# - 메서드 선택(input): conventional / regularized_conventional /
#                      force-equillibrium / hybrid / regularized_hybrid
#   -> 목표함수 및 출력 경로, 상수 자동 전환
# - Weighted Laplacian: U_xx, U_yy, U_zz 축별 가중치 적용
# ============================================================

import os
import time
import math
import numpy as np
import pandas as pd

# Headless backend (no X server)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

import torch
from scipy.optimize import minimize

# ============================================================
# [사용자 편집 구역] (기본값; 실제 사용값은 configure_method()에서 메서드에 따라 덮어씀)
# ============================================================
CSV_FILENAME = "optimization_steps_data_with_phases_and_terrain.csv" # per-step 로그 CSV
BASE_OUTPUT_DIR = os.path.join("results_github") # 메서드별 하위 폴더 생성
RUN_TAG = "BFGS_8x8_rand42_fixed_10000_steps" # 공통 러닝 태그

# 트래핑 목표 지점 (미터 좌표)
TARGET_COORD = (0.01, 0.01, 0.03) # (x, y, z) in meters

# 공통 가중치 심볼(목표함수 조합에 사용)
#  maximize [ w_lap*Δ²_w U  -  α*|∇U|  -  β*Penalty(|p|) ]
#  -> loss = -metric
W_LAPLACIAN = 1.0 # Δ²_w U 전체 계수  (메서드별 고정 1.0)
ALPHA_F = 500.0 # |∇U| 계수 (Force-Equilibrium/Hybrid에서 사용)
BETA_P = 5e-5 # |p| 계수 (메서드에 따라 1.0 또는 5e-5)

# Weighted Laplacian (각 방향별 가중치; 메서드에 따라 설정)
# Δ²_w U = W_XX * U_xx + W_YY * U_yy + W_ZZ * U_zz
W_LAPLACIAN_XX = 1
W_LAPLACIAN_YY = 1
W_LAPLACIAN_ZZ = 1

# Charbonnier epsilon 상대값 (로컬 RMS 기준; regularized* 에서 사용)
EPS_REL = 1e-3 # 섹션 2.2.4의 0.001 규칙

# 결과 플롯: 압력장 3개 단면(YZ@x*, XZ@y*, XY@z*)
GENERATE_SLICE_PLOTS = True

# 결정변수 공간 지형 시각화(최종점 기준
GENERATE_DECISION_SLICES = True
SLICE_HALF_RANGE = 0.30 # 각 축 ±범위 [rad]
SLICE_N_1D = 81 # 1D 샘플 개수
SLICE_N_2D = 61 # 2D 격자 한 변 샘플 개수

# 최적화(실험 반복) 설정
NUM_TRIALS = 100
NUM_OPT_STEPS = 10000

# 헤시안 분광 추정 간격/정밀도 (매 스텝 수행은 과중하므로 간헐적)
SPEC_EVERY = 200 # 200 스텝마다 시도
SPEC_K = 10 # Lanczos 반복 수(작을수록 가벼움)
HVP_EPS = 1e-4 # 유한차분 H·v epsilon
# ============================================================


# ============================================================
# 환경/출력 설정 (메서드 선택은 맨 아래에서 처리)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(random_seed)
print(f"[INFO] Using Fixed Random Seed: {random_seed}")

# ============================================================
# 물리/격자 파라미터
# ============================================================
amp_table = torch.tensor([
    10.000, 11.832, 13.509, 15.414, 17.615, 18.240, 20.172, 21.626, 22.309, 23.054, 23.820, 23.820, 23.820,
    24.650, 24.650, 26.306, 29.052, 32.140, 35.496, 39.243, 41.952, 46.411, 47.958, 49.598, 53.009, 56.657,
    58.652, 60.581, 62.610, 64.885, 67.007, 69.282, 71.624, 71.624, 76.551, 76.551, 79.183, 84.676, 84.676,
    87.579, 90.388, 93.488, 93.488, 93.488, 96.695, 100.000, 96.695, 93.488, 93.488, 93.488, 90.388, 87.579,
    84.676, 84.676, 79.183, 76.551, 76.551, 71.624, 71.624, 69.282, 67.007, 64.885, 62.610, 60.581, 58.652,
    56.657, 53.009, 49.598, 47.958, 46.411, 41.952, 39.243, 35.496, 32.140, 29.052, 26.306, 24.650, 24.650,
    23.820, 23.820, 23.820, 23.054, 22.309, 21.626, 20.172, 18.240, 17.615, 15.414, 13.509, 11.832, 10.000
], dtype=torch.float64, device=device)

frequency = 40000.0 # Hz
speed_of_sound = 343.0 # m/s
wavelength = speed_of_sound / frequency

# 8x8 배열 (1 cm pitch)
x_coords = np.arange(-3.5, 4.5, 1.0)
y_coords = np.arange(-3.5, 4.5, 1.0)
transducer_positions = torch.tensor(
    [(x * 0.01, y * 0.01, 0.0) for x in x_coords for y in y_coords],
    dtype=torch.float64, device=device
)
num_transducers = len(transducer_positions)
print(f"[INFO] Transducer grid: 8x8 ({num_transducers} transducers).")

# 관측 격자 (201^3)
grid_size = 201
x_vals = torch.linspace(-0.05, 0.05, grid_size, dtype=torch.float64, device=device)
y_vals = torch.linspace(-0.05, 0.05, grid_size, dtype=torch.float64, device=device)
z_vals = torch.linspace(0.0, 0.1, grid_size, dtype=torch.float64, device=device)
dx = (x_vals[1] - x_vals[0]).item()
dy = (y_vals[1] - y_vals[0]).item()
dz = (z_vals[1] - z_vals[0]).item()

# 목표 지점 (격자 인덱스)
target_coord_x, target_coord_y, target_coord_z = TARGET_COORD
x_idx = torch.argmin(torch.abs(x_vals - target_coord_x)).item()
y_idx = torch.argmin(torch.abs(y_vals - target_coord_y)).item()
z_idx = torch.argmin(torch.abs(z_vals - target_coord_z)).item()
target_x = x_vals[x_idx].item()
target_y = y_vals[y_idx].item()
target_z = z_vals[z_idx].item()
print(f"[INFO] Target Index: ({x_idx}, {y_idx}, {z_idx}) | Target Coord: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})")

# 로컬 윈도우(5x5x5) 반경
half = 2 # 2 -> (2*2+1)=5 포인트

# ============================================================
# 메서드-의존 파라미터 (런타임에 세팅)
# ============================================================
METHOD_NAME = None
PRESSURE_PENALTY_MODE = 'abs' # 'abs' or 'smooth_abs'
CURRENT_WEIGHTS = dict(w_lap=W_LAPLACIAN, alpha=0.0, beta=0.0)
OUTPUT_DIR = None

# ============================================================
# 계산 함수
# ============================================================
def compute_pressure_field_torch(phase_vector, amp_table, wavelength, X, Y, Z, transducer_positions):
    k_val = 2.0 * math.pi / wavelength
    grid_points = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1) # [M,3]
    delta = grid_points.unsqueeze(1) - transducer_positions.unsqueeze(0) # [M,N,3]
    R = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
    cos_theta = torch.clamp(delta[:, :, 2] / R, -1.0, 1.0)
    theta_deg = torch.rad2deg(torch.acos(cos_theta)).clamp(0.0, 90.0)

    max_idx = amp_table.shape[0] - 1
    low_index_f = theta_deg # 1 deg grid
    low_index = torch.floor(low_index_f).long()
    high_index = torch.clamp(low_index + 1, max=max_idx)
    frac = (low_index_f - low_index.to(torch.float64))
    A_low = amp_table[low_index]
    A_high = amp_table[high_index]
    A_theta = A_low + frac * (A_high - A_low)

    amplitude = A_theta / R
    propagation_phase = k_val * R
    total_phase = phase_vector.unsqueeze(0) + propagation_phase # [M,N]
    p_complex = torch.polar(amplitude, total_phase) # complex128
    p_field = torch.sum(p_complex, dim=1) # [M]
    return p_field.reshape(X.shape)

def compute_gradient_torch(U, dx, dy, dz):
    return torch.gradient(U, spacing=(dx, dy, dz), edge_order=1)

def compute_laplacian_weighted_torch(U, dx, dy, dz):
    """Δ²_w U = W_XX*U_xx + W_YY*U_yy + W_ZZ*U_zz (축별 가중)"""
    grad_U_x, grad_U_y, grad_U_z = compute_gradient_torch(U, dx, dy, dz)
    L_xx, _, _ = torch.gradient(grad_U_x, spacing=(dx, dy, dz), edge_order=1)
    _, L_yy, _ = torch.gradient(grad_U_y, spacing=(dx, dy, dz), edge_order=1)
    _, _, L_zz = torch.gradient(grad_U_z, spacing=(dx, dy, dz), edge_order=1)
    return (W_LAPLACIAN_XX * L_xx +
            W_LAPLACIAN_YY * L_yy +
            W_LAPLACIAN_ZZ * L_zz)

def compute_laplacian_unweighted_torch(U, dx, dy, dz):
    """Δ² U = U_xx + U_yy + U_zz (비가중, 로깅용)"""
    grad_U_x, grad_U_y, grad_U_z = compute_gradient_torch(U, dx, dy, dz)
    L_xx, _, _ = torch.gradient(grad_U_x, spacing=(dx, dy, dz), edge_order=1)
    _, L_yy, _ = torch.gradient(grad_U_y, spacing=(dx, dy, dz), edge_order=1)
    _, _, L_zz = torch.gradient(grad_U_z, spacing=(dx, dy, dz), edge_order=1)
    return L_xx + L_yy + L_zz

def pressure_penalty(p2_center, p_abs_center, p2_local_rms):
    if PRESSURE_PENALTY_MODE == 'smooth_abs':
        eps = EPS_REL * (p2_local_rms + 1e-32)
        return torch.sqrt(p2_center + eps*eps)
    elif PRESSURE_PENALTY_MODE == 'abs':
        return torch.sqrt(p2_center + 1e-32)
    else:
        raise ValueError(f"Unknown PRESSURE_PENALTY_MODE: {PRESSURE_PENALTY_MODE}")

def compute_gorkov_objective_local_torch(pf, dx, dy, dz):
    rho0, c0, rho_p, c_p = 1.225, 343.0, 100.0, 2400.0
    omega = 2 * math.pi * frequency
    r = 1.3e-3 / 2
    V = 4/3 * math.pi * r**3

    K1 = 0.25 * V * (1 / (c0**2 * rho0) - 1 / (c_p**2 * rho_p))
    K2 = 0.75 * V * ((rho0 - rho_p) / (omega**2 * rho0 * (rho0 + 2 * rho_p)))

    abs_p2 = (pf.real**2 + pf.imag**2)
    dpdx, dpdy, dpdz = compute_gradient_torch(pf, dx, dy, dz)
    v_sq = (dpdx.real**2 + dpdx.imag**2) + (dpdy.real**2 + dpdy.imag**2) + (dpdz.real**2 + dpdz.imag**2)

    U = K1 * abs_p2 - K2 * v_sq
    gradUx, gradUy, gradUz = compute_gradient_torch(U, dx, dy, dz)
    
    lapU_weighted = compute_laplacian_weighted_torch(U, dx, dy, dz)
    lapU_unweighted = compute_laplacian_unweighted_torch(U, dx, dy, dz)

    center_idx = (half, half, half)
    laplacian_center_weighted = lapU_weighted[center_idx]
    laplacian_center_unweighted = lapU_unweighted[center_idx]

    grad_mag_center = torch.sqrt(gradUx[center_idx]**2 + gradUy[center_idx]**2 + gradUz[center_idx]**2)

    p2_center = abs_p2[center_idx]
    p_abs_center = torch.sqrt(p2_center + 1e-32)
    p2_local_rms = torch.sqrt(torch.mean(abs_p2))

    p_pen = pressure_penalty(p2_center, p_abs_center, p2_local_rms)

    w_lap = CURRENT_WEIGHTS['w_lap']
    alpha = CURRENT_WEIGHTS['alpha']
    beta  = CURRENT_WEIGHTS['beta']

    metric = (w_lap * laplacian_center_weighted
              - alpha * grad_mag_center
              - beta  * p_pen)

    return metric, laplacian_center_weighted, grad_mag_center, p_abs_center, laplacian_center_unweighted

def objective_fn_torch(ph_tensor, x_idx, y_idx, z_idx):
    x_local = x_vals[x_idx-half:x_idx+half+1]
    y_local = y_vals[y_idx-half:y_idx+half+1]
    z_local = z_vals[z_idx-half:z_idx+half+1]
    Xl, Yl, Zl = torch.meshgrid(x_local, y_local, z_local, indexing='ij')

    pf_local = compute_pressure_field_torch(ph_tensor, amp_table, wavelength, Xl, Yl, Zl, transducer_positions)
    gorkov_metric, _, _, _, _ = compute_gorkov_objective_local_torch(pf_local, dx, dy, dz)
    return -gorkov_metric

def get_metrics_for_logging_new_method(ph_tensor, x_idx, y_idx, z_idx):
    with torch.no_grad():
        x_local = x_vals[x_idx-half:x_idx+half+1]
        y_local = y_vals[y_idx-half:y_idx+half+1]
        z_local = z_vals[z_idx-half:z_idx+half+1]
        Xl, Yl, Zl = torch.meshgrid(x_local, y_local, z_local, indexing='ij')
        pf_local = compute_pressure_field_torch(ph_tensor, amp_table, wavelength, Xl, Yl, Zl, transducer_positions)

        metric, _, gradmag_c, p_abs_c, true_lap_c = compute_gorkov_objective_local_torch(pf_local, dx, dy, dz)
        loss = -metric
        return float(true_lap_c), float(gradmag_c), float(p_abs_c), float(loss)

# ------------------------------------------------------------
# 슬라이스(단면) — 전체 3D 대신 2D 평면만 계산
# ------------------------------------------------------------
def compute_slice_YZ(phase_vector, x_fixed):
    Yl, Zl = torch.meshgrid(y_vals, z_vals, indexing='ij')
    Xl = torch.full_like(Yl, x_fixed)
    pf = compute_pressure_field_torch(phase_vector, amp_table, wavelength, Xl, Yl, Zl, transducer_positions)
    return torch.abs(pf)

def compute_slice_XZ(phase_vector, y_fixed):
    Xl, Zl = torch.meshgrid(x_vals, z_vals, indexing='ij')
    Yl = torch.full_like(Xl, y_fixed)
    pf = compute_pressure_field_torch(phase_vector, amp_table, wavelength, Xl, Yl, Zl, transducer_positions)
    return torch.abs(pf)

def compute_slice_XY(phase_vector, z_fixed):
    Xl, Yl = torch.meshgrid(x_vals, y_vals, indexing='ij')
    Zl = torch.full_like(Xl, z_fixed)
    pf = compute_pressure_field_torch(phase_vector, amp_table, wavelength, Xl, Yl, Zl, transducer_positions)
    return torch.abs(pf)

# ============================================================
# SciPy 래퍼
# ============================================================
def objective_for_scipy(phases_np):
    phases_torch = torch.tensor(phases_np, dtype=torch.float64, device=device)
    loss = objective_fn_torch(phases_torch, x_idx, y_idx, z_idx)
    return float(loss.item())

def jacobian_for_scipy(phases_np):
    phases_torch = torch.tensor(phases_np, dtype=torch.float64, device=device, requires_grad=True)
    loss = objective_fn_torch(phases_torch, x_idx, y_idx, z_idx)
    (grad,) = torch.autograd.grad(loss, phases_torch, retain_graph=False, create_graph=False)
    return grad.detach().cpu().numpy().astype(np.float64)

# ============================================================
# (간헐) H·v 및 Lanczos 극값 고유치 근사
# ============================================================
def hvp_fd(x_np, v_np, eps=HVP_EPS):
    g_plus = jacobian_for_scipy(x_np + eps*v_np)
    g_minus = jacobian_for_scipy(x_np - eps*v_np)
    return (g_plus - g_minus) / (2.0*eps)

def lanczos_extreme_eigs(hvp_fun, x_np, n, k=10):
    Q = []
    alphas, betas = [], []
    q = np.random.randn(n); q /= (np.linalg.norm(q) + 1e-16)
    beta_prev = 0.0
    for j in range(k):
        v = hvp_fun(x_np, q) if j == 0 else hvp_fun(x_np, q) - beta_prev * Q[-1]
        for qi in Q:
            v -= np.dot(v, qi) * qi
        alpha = float(np.dot(q, v))
        v -= alpha * q
        beta = float(np.linalg.norm(v) + 1e-16)

        Q.append(q.copy())
        alphas.append(alpha); betas.append(beta)
        if beta < 1e-14:
            break
        beta_prev = beta
        q = v / beta

    m = len(alphas)
    if m == 0:
        return np.nan, np.nan
    T = np.zeros((m, m))
    for i in range(m):
        T[i, i] = alphas[i]
        if i+1 < m:
            T[i, i+1] = betas[i+1]
            T[i+1, i] = betas[i+1]
    w = np.linalg.eigvalsh(T)
    return float(w[0]), float(w[-1])

# ============================================================
# 콜백 (스텝 단위 로깅 + 지형 메트릭)
# ============================================================
class StepRecorder:
    def __init__(self, trial_number, trial_start_time, log_list, total_steps, x0_np):
        self.trial_number = trial_number
        self.trial_start_time = trial_start_time
        self.all_steps_log = log_list
        self.step_counter = 0
        self.total_steps = total_steps

        self.n = len(x0_np)
        self.H = np.eye(self.n, dtype=np.float64)
        self.prev_x = x0_np.copy()
        self.prev_loss = objective_for_scipy(self.prev_x)
        self.prev_grad = jacobian_for_scipy(self.prev_x)
        self.prev_p = - self.H.dot(self.prev_grad)

    def __call__(self, xk):
        self.step_counter += 1
        elapsed_time = time.time() - self.trial_start_time

        xk = np.asarray(xk, dtype=np.float64)
        s = xk - self.prev_x
        step_norm = float(np.linalg.norm(s))

        phases_torch = torch.tensor(xk, dtype=torch.float64, device=device)
        true_laplacian, grad_mag, p_abs, loss = get_metrics_for_logging_new_method(
            phases_torch, x_idx, y_idx, z_idx
        )
        curr_loss = float(loss)

        gk = jacobian_for_scipy(xk)
        y = gk - self.prev_grad
        sTy = float(np.dot(s, y))
        valid_curv = (sTy > 1e-16)

        denom_lin = -float(np.dot(self.prev_grad, s))
        actual_reduction = float(self.prev_loss - curr_loss)
        rho_lin = actual_reduction / denom_lin if abs(denom_lin) > 1e-12 else float('nan')

        Hg = self.H.dot(self.prev_grad)
        H2g = self.H.dot(Hg)
        num = float(np.dot(self.prev_grad, Hg))
        den = float(np.dot(self.prev_grad, H2g)) if np.linalg.norm(H2g) > 0 else np.nan
        r_ratio = (num / den) if (den is not np.nan and abs(den) > 1e-300) else np.nan

        p_prev = self.prev_p
        cos_theta = float( np.dot(-self.prev_grad, p_prev) /
                           ((np.linalg.norm(self.prev_grad)+1e-16)*(np.linalg.norm(p_prev)+1e-16)) )

        denom_pp = float(np.dot(p_prev, p_prev))
        alpha_imp = float(np.dot(s, p_prev) / denom_pp) if denom_pp > 1e-300 else np.nan

        cos_phi = float( np.dot(s, y) /
                         ((np.linalg.norm(s)+1e-16)*(np.linalg.norm(y)+1e-16)) ) if np.linalg.norm(y)>0 else np.nan
        gamma_scale = float( np.dot(y, y) / sTy ) if valid_curv else np.nan

        lam_min = lam_max = kappa = np.nan
        if (self.step_counter % SPEC_EVERY == 0) and (self.step_counter > 0):
            try:
                lm, lM = lanczos_extreme_eigs(hvp_fd, xk, self.n, k=SPEC_K)
                lam_min, lam_max = lm, lM
                kappa = float(abs(lam_max) / max(abs(lam_min), 1e-16)) if np.isfinite(lam_min) else np.nan
            except Exception:
                lam_min = lam_max = kappa = np.nan

        row = {
            'trial': self.trial_number,
            'step': self.step_counter,
            'time_s': elapsed_time,
            'loss': curr_loss,
            'laplacian': float(true_laplacian),
            'gorkov_grad_mag': float(grad_mag),
            'pressure_abs': float(p_abs),
            'delta_tr_radius': step_norm,
            'rho_lin': rho_lin,
            'gnorm': float(np.linalg.norm(self.prev_grad)),
            'pnorm': float(np.linalg.norm(p_prev)),
            'alpha_imp': alpha_imp,
            'cos_theta': cos_theta,
            'r_ratio': r_ratio,
            'sTy': sTy,
            'cos_phi': cos_phi,
            'gamma_scale': gamma_scale,
            'lambda_min_est': lam_min,
            'lambda_max_est': lam_max,
            'kappa_est': kappa,
        }
        for i in range(len(xk)):
            row[f'phase_{i}'] = float(xk[i])
        self.all_steps_log.append(row)

        if (self.step_counter % 20 == 0) or (self.step_counter == self.total_steps):
            print(f"  Trial {self.trial_number}, Step {self.step_counter} | "
                  f"loss={curr_loss:.3e} | Δ(step)={step_norm:.3e} | ρ_lin={rho_lin:.3e} | "
                  f"α_imp={alpha_imp:.3e} | cosθ={cos_theta:.3f} | t={elapsed_time:.1f}s")

        if valid_curv:
            rho = 1.0 / sTy
            I = np.eye(self.n, dtype=np.float64)
            I_sy = (I - rho * np.outer(s, y))
            I_ys = (I - rho * np.outer(y, s))
            self.H = I_sy.dot(self.H).dot(I_ys) + rho * np.outer(s, s)

        self.prev_x = xk.copy()
        self.prev_loss = curr_loss
        self.prev_grad = gk.copy()
        self.prev_p = - self.H.dot(self.prev_grad)

# ============================================================
# 시각화 유틸 — 압력장 3단면 (공통 컬러바 상한: XY max)
# ============================================================
def plot_field_slices(yz_abs_np, xz_abs_np, xy_abs_np, trial_no, final_loss):
    x_vals_np = x_vals.cpu().numpy()
    y_vals_np = y_vals.cpu().numpy()
    z_vals_np = z_vals.cpu().numpy()

    common_vmax = float(np.nanmax(xy_abs_np))
    norm = colors.Normalize(vmin=0.0, vmax=common_vmax)

    fig = plt.figure(figsize=(18, 5))
    plt.suptitle(f'Trial {trial_no:03d} | Final Loss: {final_loss:.3e}', fontsize=14)

    ax = plt.subplot(1, 3, 1)
    im1 = ax.imshow(
        yz_abs_np.T,
        extent=[y_vals_np[0], y_vals_np[-1], z_vals_np[0], z_vals_np[-1]],
        origin='lower', aspect='auto', cmap='viridis', norm=norm
    )
    ax.scatter(target_y, target_z, c='red', marker='x', s=60, linewidths=1.5)
    ax.set_title(f'YZ (x={target_x:.3f} m)'); ax.set_xlabel('y (m)'); ax.set_ylabel('z (m)')
    plt.colorbar(im1, ax=ax, label='|p|')

    ax = plt.subplot(1, 3, 2)
    im2 = ax.imshow(
        xz_abs_np.T,
        extent=[x_vals_np[0], x_vals_np[-1], z_vals_np[0], z_vals_np[-1]],
        origin='lower', aspect='auto', cmap='viridis', norm=norm
    )
    ax.scatter(target_x, target_z, c='red', marker='x', s=60, linewidths=1.5)
    ax.set_title(f'XZ (y={target_y:.3f} m)'); ax.set_xlabel('x (m)'); ax.set_ylabel('z (m)')
    plt.colorbar(im2, ax=ax, label='|p|')

    ax = plt.subplot(1, 3, 3)
    im3 = ax.imshow(
        xy_abs_np.T,
        extent=[x_vals_np[0], x_vals_np[-1], y_vals_np[0], y_vals_np[-1]],
        origin='lower', aspect='auto', cmap='viridis', norm=norm
    )
    ax.scatter(target_x, target_y, c='red', marker='x', s=60, linewidths=1.5)
    ax.set_title(f'XY (z={target_z:.3f} m)'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    plt.colorbar(im3, ax=ax, label='|p|')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    return fig

# ============================================================
# 결정변수 공간 1D/2D 슬라이스
# ============================================================
def decision_space_slices(objective_fun, x_star, g_star, p_star, trial_id):
    u = -g_star
    if np.linalg.norm(u) > 0:
        u = u / (np.linalg.norm(u)+1e-16)
    else:
        u = np.random.randn(*x_star.shape); u /= np.linalg.norm(u)+1e-16

    v = p_star.copy()
    v = v - np.dot(v, u)*u
    nv = np.linalg.norm(v)
    if nv > 0:
        v = v / nv
    else:
        z = np.random.randn(*x_star.shape)
        z -= np.dot(z, u)*u
        v = z / (np.linalg.norm(v)+1e-16)

    ts = np.linspace(-SLICE_HALF_RANGE, SLICE_HALF_RANGE, SLICE_N_1D)
    f_u = [objective_fun(x_star + t*u) for t in ts]
    f_v = [objective_fun(x_star + t*v) for t in ts]

    A = np.linspace(-SLICE_HALF_RANGE, SLICE_HALF_RANGE, SLICE_N_2D)
    B = np.linspace(-SLICE_HALF_RANGE, SLICE_HALF_RANGE, SLICE_N_2D)
    F = np.zeros((SLICE_N_2D, SLICE_N_2D))
    for i,a in enumerate(A):
        for j,b in enumerate(B):
            F[j,i] = objective_fun(x_star + a*u + b*v)

    fig = plt.figure(figsize=(16,5))
    plt.suptitle(f'Decision-Space Slices (trial {trial_id:03d})', fontsize=16)

    ax1 = plt.subplot(1,3,1)
    ax1.plot(ts, f_u, linewidth=1.5)
    ax1.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax1.set_title('1D slice along -grad'); ax1.set_xlabel('t (rad)'); ax1.set_ylabel('f(x* + t u)')

    ax2 = plt.subplot(1,3,2)
    ax2.plot(ts, f_v, linewidth=1.5)
    ax2.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_title('1D slice along BFGS dir'); ax2.set_xlabel('t (rad)'); ax2.set_ylabel('f(x* + t v)')

    ax3 = plt.subplot(1,3,3)
    im = ax3.imshow(F, extent=[A[0], A[-1], B[0], B[-1]], origin='lower', aspect='auto', cmap='viridis')
    ax3.scatter(0.0, 0.0, c='red', marker='x', s=60, linewidth=2)
    ax3.set_title('2D slice on span{u,v}')
    ax3.set_xlabel('a (rad)'); ax3.set_ylabel('b (rad)')
    plt.colorbar(im, ax=ax3, label='f')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# ============================================================
# 메서드 구성 함수 (여기서 메서드별 상수 자동 세팅)
# ============================================================
def configure_method(method_raw: str):
    global METHOD_NAME, PRESSURE_PENALTY_MODE, CURRENT_WEIGHTS, OUTPUT_DIR
    global W_LAPLACIAN, ALPHA_F, BETA_P
    global W_LAPLACIAN_XX, W_LAPLACIAN_YY, W_LAPLACIAN_ZZ

    method = (method_raw or "").strip().lower()
    valid = {
        "conventional",
        "regularized_conventional",
        "force-equillibrium",
        "hybrid",
        "regularized_hybrid",
    }
    if method not in valid:
        raise ValueError(f"Unknown method: {method}. "
                         f"Choose one of {sorted(list(valid))}")

    METHOD_NAME = method

    W_LAPLACIAN = 1.0
    ALPHA_F = 500.0

    if method in {"conventional", "regularized_conventional"}:
        BETA_P = 1.0
        W_LAPLACIAN_XX, W_LAPLACIAN_YY, W_LAPLACIAN_ZZ = 1000, 10, 10
    else:
        BETA_P = 5e-5
        W_LAPLACIAN_XX = W_LAPLACIAN_YY = W_LAPLACIAN_ZZ = 1

    if method == "conventional":
        CURRENT_WEIGHTS = dict(w_lap=W_LAPLACIAN, alpha=0.0,       beta=BETA_P)
        PRESSURE_PENALTY_MODE = 'abs'
    elif method == "regularized_conventional":
        CURRENT_WEIGHTS = dict(w_lap=W_LAPLACIAN, alpha=0.0,       beta=BETA_P)
        PRESSURE_PENALTY_MODE = 'smooth_abs'
    elif method == "force-equillibrium":
        CURRENT_WEIGHTS = dict(w_lap=W_LAPLACIAN, alpha=ALPHA_F,   beta=0.0)
        PRESSURE_PENALTY_MODE = 'abs'
    elif method == "hybrid":
        CURRENT_WEIGHTS = dict(w_lap=W_LAPLACIAN, alpha=ALPHA_F,   beta=BETA_P)
        PRESSURE_PENALTY_MODE = 'abs'
    elif method == "regularized_hybrid":
        CURRENT_WEIGHTS = dict(w_lap=W_LAPLACIAN, alpha=ALPHA_F,   beta=BETA_P)
        PRESSURE_PENALTY_MODE = 'smooth_abs'

    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, METHOD_NAME, RUN_TAG)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[INFO] Method: {METHOD_NAME}")
    print(f"[INFO] Weights: {CURRENT_WEIGHTS} | Penalty mode: {PRESSURE_PENALTY_MODE}")
    print(f"[INFO] W_LAPLACIAN_XX,YY,ZZ = {W_LAPLACIAN_XX}, {W_LAPLACIAN_YY}, {W_LAPLACIAN_ZZ}")
    print(f"[INFO] Output dir: {OUTPUT_DIR}")

# ============================================================
# 메인 루프
# ============================================================
optimizer_method = 'BFGS'

def main():
    all_steps_log = []

    print(f"\n[INFO] Starting {NUM_TRIALS} optimization trials, target max iter per trial={NUM_OPT_STEPS}")
    total_start_time = time.time()

    for i in range(NUM_TRIALS):
        trial_start_time = time.time()
        trial_no = i + 1
        print(f"\n--- Running Trial {trial_no}/{NUM_TRIALS} ---")

        initial_phases_np = np.random.rand(num_transducers) * 2.0 * math.pi

        recorder = StepRecorder(
            trial_number=trial_no,
            trial_start_time=trial_start_time,
            log_list=all_steps_log,
            total_steps=NUM_OPT_STEPS,
            x0_np=initial_phases_np
        )

        res = minimize(
            fun=objective_for_scipy,
            x0=initial_phases_np,
            method=optimizer_method,
            jac=jacobian_for_scipy,
            callback=recorder,
            options={'maxiter': NUM_OPT_STEPS, 'disp': False, 'gtol': 0}
        )

        trial_total_time = time.time() - trial_start_time
        print(f"[INFO] Trial {trial_no} Finished. Elapsed: {trial_total_time:.2f}s | Final loss: {res.fun:.3e} | iters={res.nit}")

        if GENERATE_SLICE_PLOTS:
            print("  [INFO] Computing ONLY 3 slices (YZ/XZ/XY) and saving plots...")
            with torch.no_grad():
                final_phases = torch.tensor(res.x, dtype=torch.float64, device=device)
                yz_abs = compute_slice_YZ(final_phases, x_vals[x_idx])
                xz_abs = compute_slice_XZ(final_phases, y_vals[y_idx])
                xy_abs = compute_slice_XY(final_phases, z_vals[z_idx])

                fig = plot_field_slices(yz_abs.cpu().numpy(), xz_abs.cpu().numpy(), xy_abs.cpu().numpy(),
                                        trial_no, res.fun)
                filename = os.path.join(OUTPUT_DIR, f'trial_{trial_no:03d}_field_slices.png')
                fig.savefig(filename, dpi=150); plt.close(fig)
                print(f"  [INFO] Plot saved: {filename}")

        if GENERATE_DECISION_SLICES:
            print("  [INFO] Computing decision-space slices at final point ...")
            x_final = np.asarray(res.x, dtype=np.float64)
            g_final = jacobian_for_scipy(x_final)
            try:
                p_final = - recorder.H.dot(g_final)
            except Exception:
                p_final = - g_final

            fig2 = decision_space_slices(objective_for_scipy, x_final, g_final, p_final, trial_no)
            filename2 = os.path.join(OUTPUT_DIR, f"trial_{trial_no:03d}_decision_slices.png")
            fig2.savefig(filename2, dpi=170); plt.close(fig2)
            print(f"  [INFO] Decision-space slices saved: {filename2}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - total_start_time
    print(f"\n[INFO] Total execution time for {NUM_TRIALS} trials: {total_time:.2f} s")

    # --------------------------------------------------------
    # CSV 저장 (매 스텝 phase + 지형 메트릭 포함)
    # --------------------------------------------------------
    print("\n[INFO] Creating & saving per-step CSV ...")
    df_final = pd.DataFrame(all_steps_log)

    metric_cols = [
        'step','time_s','loss','laplacian','gorkov_grad_mag','pressure_abs',
        'delta_tr_radius','rho_lin',
        'gnorm','pnorm','alpha_imp','cos_theta','r_ratio','sTy','cos_phi','gamma_scale',
        'lambda_min_est','lambda_max_est','kappa_est'
    ]
    phase_cols = [f'phase_{i}' for i in range(num_transducers)]
    final_cols = ['trial'] + metric_cols + phase_cols
    df_final = df_final[final_cols]

    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    df_final.to_csv(csv_path, index=False)
    print(f"[OK] Saved CSV with {len(df_final)} rows: {csv_path}")

    # 요약: step==NUM_OPT_STEPS 우선, 없으면 각 trial의 마지막 스텝
    print("\n" + "="*60)
    print(f" Optimization Step Data Summary ({optimizer_method.upper()})")
    print("="*60)
    if 'step' in df_final.columns:
        df_summary = df_final[df_final['step'] == NUM_OPT_STEPS]
        used_last = False
        if df_summary.empty:
            used_last = True
            df_summary = (df_final.sort_values(['trial','step'])
                                  .groupby('trial', as_index=False)
                                  .tail(1))
        print(f"Trials summarized: {df_summary['trial'].nunique()} "
              f"({'last step per trial' if used_last else f'step={NUM_OPT_STEPS}'})")
        for col in ['loss','laplacian','pressure_abs','gorkov_grad_mag',
                    'delta_tr_radius','rho_lin','alpha_imp','cos_theta']:
            mu = df_summary[col].mean(); sd = df_summary[col].std(); mn = df_summary[col].min()
            print(f"{col:15s} | Mean: {mu:.4e}, Std: {sd:.4e}, Min: {mn:.4e}")
        print(f"Step Time (s)   | Mean: {df_summary['time_s'].mean():.2f}s, Std: {df_summary['time_s'].std():.2f}s")
    else:
        print("No 'step' column in final dataframe.")
    print("="*60)

if __name__ == "__main__":
    user_method = input("Select method [conventional | regularized_conventional | force-equillibrium | hybrid | regularized_hybrid]: ")
    configure_method(user_method)
    main()