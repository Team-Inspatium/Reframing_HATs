import os, time, math
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ------------------------- OUTPUT & RUN CONFIG -------------------------
# ※ 'Marzo' 표기 제거 + method별 하위 폴더 분기
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "results_conventional_methods")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

num_trials    = 100
num_opt_steps = 10000

# 헤시안 스펙트럼(간헐적) 추정 설정
SPEC_EVERY    = 200        # 몇 스텝마다 스펙트럼(Hv-Lanczos) 추정할지
SPEC_K        = 10         # Lanczos 반복 횟수(작을수록 가벼움)
HVP_EPS       = 1e-4       # H·v 유한차분 epsilon

# 결정변수 공간 슬라이스 범위/해상도
SLICE_H       = 0.30       # ±H 라디안
SLICE_N_1D    = 81
SLICE_N_2D    = 61

# ----------------- method 선택 (penalty 전환 & 출력 폴더 분기) -----------------
VALID_METHODS = {"conventional", "regularized_conventional"}
try:
    method = input("method (conventional | regularized_conventional): ").strip().lower()
except Exception:
    method = "conventional"
if method not in VALID_METHODS:
    print(f"[WARN] Unknown method '{method}'. Falling back to 'conventional'.")
    method = "conventional"

# ----------------- Pressure penalty config -----------------
# 'abs'         : 기존 |p|
# 'smooth_abs'  : ϕε(p) = sqrt(|p|^2 + ε^2)
# 'squared'     : |p|^2
if method == "conventional":
    PRESSURE_PENALTY_MODE = 'abs'
elif method == "regularized_conventional":
    PRESSURE_PENALTY_MODE = 'smooth_abs'

# ε = EPS_REL × (로컬 윈도우의 |p| 규모). (원본 값 유지)
EPS_REL = 1e-3

# 가중치 (원본 값 유지)
W_P    = 1.0
W_LXX  = 1000.0
W_LYY  = 10.0
W_LZZ  = 10.0

# 메서드별 출력 폴더 (method명이 경로에 반영되어 서로 분리 저장)
OUTPUT_DIR = os.path.join(
    BASE_OUTPUT_DIR,
    f"{method.upper()}_BFGS_8x8_rand42_steps{num_opt_steps}_trials{num_trials}"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- 랜덤 시드/디바이스 -----------------
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")
print(f"[Method] {method}  |  penalty={PRESSURE_PENALTY_MODE}")
print(f"[Output] {OUTPUT_DIR}")

# ----------------- 물리/격자/배열 파라미터 -----------------
half = 2  # 로컬 윈도우 반경 (5x5x5)
frequency = 40000.0
speed_of_sound = 343.0
wavelength = speed_of_sound / frequency
num_transducers = 64

# 방사 패턴(각도별 진폭) 테이블
amp_table = torch.tensor([
    10.000, 11.832, 13.509, 15.414, 17.615, 18.240, 20.172, 21.626, 22.309, 23.054, 23.820, 23.820, 23.820,
    24.650, 24.650, 26.306, 29.052, 32.140, 35.496, 39.243, 41.952, 46.411, 47.958, 49.598, 53.009, 56.657,
    58.652, 60.581, 62.610, 64.885, 67.007, 69.282, 71.624, 71.624, 76.551, 76.551, 79.183, 84.676, 84.676,
    87.579, 90.388, 93.488, 93.488, 93.488, 96.695, 100.000, 96.695, 93.488, 93.488, 93.488, 90.388, 87.579,
    84.676, 84.676, 79.183, 76.551, 76.551, 71.624, 71.624, 69.282, 67.007, 64.885, 62.610, 60.581, 58.652,
    56.657, 53.009, 49.598, 47.958, 46.411, 41.952, 39.243, 35.496, 32.140, 29.052, 26.306, 24.650, 24.650,
    23.820, 23.820, 23.820, 23.054, 22.309, 21.626, 20.172, 18.240, 17.615, 15.414, 13.509, 11.832, 10.000
], dtype=torch.float64, device=device)

# 8x8 트랜스듀서 좌표 (1 cm pitch)
transducer_positions = torch.tensor(
    [(x * 0.01, y * 0.01, 0.0) for x in np.arange(-3.5, 4.5, 1.0) for y in np.arange(-3.5, 4.5, 1.0)],
    dtype=torch.float64, device=device
)

# 관측 격자
grid_size = 201
x_vals = torch.linspace(-0.05, 0.05, grid_size, dtype=torch.float64, device=device)
y_vals = torch.linspace(-0.05, 0.05, grid_size, dtype=torch.float64, device=device)
z_vals = torch.linspace(0.0,  0.10, grid_size, dtype=torch.float64, device=device)
dx = float(x_vals[1]-x_vals[0]); dy = float(y_vals[1]-y_vals[0]); dz = float(z_vals[1]-z_vals[0])

# 타깃 좌표 (가장 가까운 격자 인덱스)
target_coord_x, target_coord_y, target_coord_z = 0.01, 0.01, 0.03
x_idx = int(torch.argmin(torch.abs(x_vals - target_coord_x)).item())
y_idx = int(torch.argmin(torch.abs(y_vals - target_coord_y)).item())
z_idx = int(torch.argmin(torch.abs(z_vals - target_coord_z)).item())
target_x, target_y, target_z = float(x_vals[x_idx]), float(y_vals[y_idx]), float(z_vals[z_idx])
print(f"[Target] index=({x_idx},{y_idx},{z_idx}) coord=({target_x:.4f},{target_y:.4f},{target_z:.4f})")

# ----------------------- 물리 계산 함수 -----------------------
def compute_pressure_field_torch(phase_vector, amp_table, wavelength, X, Y, Z, transducer_positions):
    k_val = 2.0 * math.pi / wavelength
    grid_points = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
    delta = grid_points.unsqueeze(1) - transducer_positions.unsqueeze(0)
    R = torch.linalg.norm(delta, dim=2)
    R = torch.clamp(R, min=1e-9)
    cos_theta = torch.clamp(delta[:, :, 2] / R, -1.0, 1.0)
    theta_deg = torch.rad2deg(torch.acos(cos_theta))
    theta_deg = torch.clamp(theta_deg, 0.0, 90.0)

    max_angle_index = amp_table.shape[0] - 1
    low_index_f = theta_deg
    low_index = torch.floor(low_index_f).long()
    high_index = torch.clamp(low_index + 1, max=max_angle_index)
    frac = (low_index_f - low_index.to(torch.float64))
    A_low = amp_table[low_index]
    A_high = amp_table[high_index]
    A_theta = A_low + frac * (A_high - A_low)

    amplitude = A_theta / R
    propagation_phase = k_val * R
    total_phase = phase_vector.unsqueeze(0) + propagation_phase
    p_complex_per_transducer = torch.polar(amplitude, total_phase)
    p_field = torch.sum(p_complex_per_transducer, dim=1)
    return p_field.reshape(X.shape)

def compute_pressure_field_batched(phase_vector, amp_table, wavelength, x_vals, y_vals, z_vals, transducer_positions, batch_size=65536):
    n = x_vals.numel()
    n_pts = n**3
    out = torch.zeros((n_pts,), dtype=torch.complex128, device='cpu')

    # 인덱스 전개
    all_idx = torch.arange(n_pts, device=device)
    ix = torch.div(all_idx, (n*n), rounding_mode='floor')
    iy = torch.div(all_idx % (n*n), n,    rounding_mode='floor')
    iz = all_idx % n

    for i in range(0, n_pts, batch_size):
        b = slice(i, min(i+batch_size, n_pts))
        bx = x_vals[ix[b]].view(-1,1,1)
        by = y_vals[iy[b]].view(-1,1,1)
        bz = z_vals[iz[b]].view(-1,1,1)
        pf = compute_pressure_field_torch(phase_vector, amp_table, wavelength, bx, by, bz, transducer_positions)
        out[b] = pf.detach().cpu().reshape(-1)
    return out.reshape(n, n, n)

def compute_gradient_torch(U, dx, dy, dz):
    return torch.gradient(U, spacing=(dx, dy, dz), edge_order=1)

# ------------------- Loss & metrics -------------------
def calculate_loss_tensor(ph_tensor, x_idx, y_idx, z_idx, return_components=False):
    """
    Loss = W_P * pressure_penalty - (W_LXX * L_xx + W_LYY * L_yy + W_LZZ * L_zz)
    pressure_penalty는 설정에 따라 |p|, sqrt(|p|^2 + ε^2), 또는 |p|^2.
    ε는 로컬 윈도우의 |p| 규모(RMS)를 기준으로 산정.
    """
    # 로컬 윈도우
    x_local = x_vals[x_idx-half : x_idx+half+1]
    y_local = y_vals[y_idx-half : y_idx+half+1]
    z_local = z_vals[z_idx-half : z_idx+half+1]
    Xl, Yl, Zl = torch.meshgrid(x_local, y_local, z_local, indexing='ij')

    # 로컬 복소 압력장
    pf_local = compute_pressure_field_torch(
        ph_tensor, amp_table, wavelength, Xl, Yl, Zl, transducer_positions
    )

    # Gor'kov 계수
    rho0, c0, rho_p, c_p = 1.225, 343.0, 100.0, 2400.0
    omega = 2 * math.pi * frequency
    r = 1.3e-3 / 2
    V = 4/3 * math.pi * r**3
    K1 = 0.25 * V * (1 / (c0**2 * rho0) - 1 / (c_p**2 * rho_p))
    K2 = 0.75 * V * ((rho0 - rho_p) / (omega**2 * rho0 * (rho0 + 2 * rho_p)))

    # |p|, |p|^2
    p2_local = pf_local.real**2 + pf_local.imag**2    # |p|^2
    p_abs_local = torch.sqrt(p2_local + 1e-32)        # |p|
    center = (half, half, half)
    p2_center = p2_local[center]
    p_abs_center = p_abs_local[center]

    # ε 산정 (로컬 RMS(|p|) × EPS_REL)
    typical_p = torch.sqrt(torch.mean(p2_local))      # RMS(|p|)
    eps = EPS_REL * (typical_p + 1e-32)

    # pressure penalty 선택 (method에 의해 PRESSURE_PENALTY_MODE가 결정됨)
    if PRESSURE_PENALTY_MODE == 'smooth_abs':
        p_penalty = torch.sqrt(p2_center + eps*eps)          # ϕε(p)
    elif PRESSURE_PENALTY_MODE == 'squared':
        p_penalty = p2_center                                 # |p|^2
    elif PRESSURE_PENALTY_MODE == 'abs':
        p_penalty = torch.sqrt(p2_center + 1e-32)             # |p|
    else:
        raise ValueError(f"Unknown PRESSURE_PENALTY_MODE: {PRESSURE_PENALTY_MODE}")

    # Gor'kov U 및 가중 라플라시안
    abs_p2 = p2_local
    dpdx, dpdy, dpdz = compute_gradient_torch(pf_local, dx, dy, dz)
    v_sq = torch.abs(dpdx)**2 + torch.abs(dpdy)**2 + torch.abs(dpdz)**2
    U = K1 * abs_p2 - K2 * v_sq

    gradUx, gradUy, gradUz = compute_gradient_torch(U, dx, dy, dz)
    L_xx, _, _ = torch.gradient(gradUx, spacing=(dx, dy, dz), edge_order=1)
    _, L_yy, _ = torch.gradient(gradUy, spacing=(dx, dy, dz), edge_order=1)
    _, _, L_zz = torch.gradient(gradUz, spacing=(dx, dy, dz), edge_order=1)

    laplacian_center = W_LXX * L_xx[center] + W_LYY * L_yy[center] + W_LZZ * L_zz[center]

    # 최종 loss
    loss = W_P * p_penalty - laplacian_center

    if return_components:
        grad_mag_center = torch.sqrt(gradUx[center]**2 + gradUy[center]**2 + gradUz[center]**2)
        return laplacian_center, grad_mag_center, p_abs_center, loss
    return loss

def calculate_all_metrics_torch(ph_tensor):
    with torch.no_grad():
        lap, gmag, pabs, loss = calculate_loss_tensor(ph_tensor, x_idx, y_idx, z_idx, return_components=True)
    return float(lap), float(gmag), float(pabs), float(loss)

# ------------------- Scipy objective & jacobian -------------------
def objective_for_scipy(phases_np):
    phases_torch = torch.tensor(phases_np, dtype=torch.float64, device=device)
    loss = calculate_loss_tensor(phases_torch, x_idx, y_idx, z_idx)
    return float(loss.detach().cpu().item())

def jacobian_for_scipy(phases_np):
    phases_torch = torch.tensor(phases_np, dtype=torch.float64, device=device, requires_grad=True)
    loss = calculate_loss_tensor(phases_torch, x_idx, y_idx, z_idx)
    (grad,) = torch.autograd.grad(loss, phases_torch, retain_graph=False, create_graph=False)
    return grad.detach().cpu().numpy().astype(np.float64)

# ------------------- H·v (finite-difference) ----------------------
def hvp_fd(x_np, v_np, eps=HVP_EPS):
    # 중앙차분: H v ≈ (∇f(x+eps v) - ∇f(x-eps v)) / (2 eps)
    g_plus  = jacobian_for_scipy(x_np + eps*v_np)
    g_minus = jacobian_for_scipy(x_np - eps*v_np)
    return (g_plus - g_minus) / (2.0*eps)

# ---------------------- Lanczos (symmetric) -----------------------
def lanczos_extreme_eigs(hvp_fun, x_np, n, k=10):
    # 간단 재직교 포함
    Q = []
    alphas, betas = [], []
    q = np.random.randn(n); q /= np.linalg.norm(q) + 1e-16
    beta_prev = 0.0
    for j in range(k):
        if j == 0:
            v = hvp_fun(x_np, q)
        else:
            v = hvp_fun(x_np, q) - beta_prev * Q[-1]
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
    lam_min = float(w[0])
    lam_max = float(w[-1])
    return lam_min, lam_max

# ------------------------ Step Recorder ---------------------------
class StepRecorder:
    def __init__(self, trial_number, x0_np, g0_np, total_steps):
        self.trial = trial_number
        self.total_steps = total_steps
        self.t0 = time.time()

        # BFGS 상태 추적(H_k 근사)
        self.H = np.eye(len(x0_np), dtype=np.float64)   # H_0
        self.x_prev = x0_np.copy()
        self.g_prev = g0_np.copy()
        self.p_prev = - self.H.dot(self.g_prev)

        self.step = 0
        self.logs = []

    def _metrics_for_completed_step(self, xk_np):
        s = xk_np - self.x_prev
        gk = jacobian_for_scipy(xk_np)
        y = gk - self.g_prev

        sTy = float(np.dot(s, y))
        valid_curv = (sTy > 1e-16)

        Hg  = self.H.dot(self.g_prev)
        H2g = self.H.dot(Hg)
        num = float(np.dot(self.g_prev, Hg))
        den = float(np.dot(self.g_prev, H2g)) if np.linalg.norm(H2g) > 0 else np.nan
        r_ratio = (num / den) if (den is not None and abs(den) > 1e-300) else np.nan

        p = self.p_prev
        cos_theta = float( np.dot(-self.g_prev, p) / ((np.linalg.norm(self.g_prev)+1e-16)*(np.linalg.norm(p)+1e-16)) )

        denom_pp = float(np.dot(p, p))
        alpha_imp = float(np.dot(s, p) / denom_pp) if denom_pp > 1e-300 else np.nan

        cos_phi = float( np.dot(s, y) / ((np.linalg.norm(s)+1e-16)*(np.linalg.norm(y)+1e-16)) ) if np.linalg.norm(y)>0 else np.nan
        gamma_sc = float( np.dot(y, y) / sTy ) if valid_curv else np.nan

        lap, gmag, pabs, loss = calculate_all_metrics_torch(torch.tensor(xk_np, dtype=torch.float64, device=device))

        lam_min = lam_max = kappa = np.nan
        if (self.step % SPEC_EVERY == 0) and (self.step > 0):
            try:
                lam_min, lam_max = lanczos_extreme_eigs(hvp_fd, xk_np, len(xk_np), k=SPEC_K)
                if np.isfinite(lam_min) and np.isfinite(lam_max):
                    if abs(lam_min) < 1e-16:
                        kappa = np.inf
                    else:
                        kappa = float(abs(lam_max)/max(abs(lam_min), 1e-16))
            except Exception:
                lam_min = lam_max = kappa = np.nan

        self.logs.append({
            'trial': self.trial,
            'step': self.step,
            'time_s': time.time()-self.t0,
            'loss': loss,
            'laplacian': lap,
            'gorkov_grad_mag': gmag,
            'pressure_abs': pabs,
            'gnorm': float(np.linalg.norm(self.g_prev)),
            'pnorm': float(np.linalg.norm(p)),
            'alpha_imp': alpha_imp,
            'cos_theta': cos_theta,
            'r_ratio': r_ratio,
            'sTy': sTy,
            'cos_phi': cos_phi,
            'gamma_scale': gamma_sc,
            'lambda_min_est': lam_min,
            'lambda_max_est': lam_max,
            'kappa_est': kappa
        })

        # BFGS 갱신
        if valid_curv:
            rho = 1.0 / sTy
            I = np.eye(self.H.shape[0], dtype=np.float64)
            I_sy = (I - rho * np.outer(s, y))
            I_ys = (I - rho * np.outer(y, s))
            self.H = I_sy.dot(self.H).dot(I_ys) + rho * np.outer(s, s)

        # 상태 업데이트
        self.x_prev = xk_np.copy()
        self.g_prev = gk.copy()
        self.p_prev = - self.H.dot(self.g_prev)

    def __call__(self, xk):
        self.step += 1
        xk_np = np.asarray(xk, dtype=np.float64)
        try:
            self._metrics_for_completed_step(xk_np)
        except Exception:
            pass
        if (self.step % 20 == 0) or (self.step == self.total_steps):
            print(f"  [trial {self.trial:02d}] step {self.step}/{self.total_steps} ...")

# --------------------- 시각화 유틸 -----------------------
def plot_field_slices(pf_full_abs, x_idx, y_idx, z_idx, trial_id, final_loss):
    xv = x_vals.detach().cpu().numpy()
    yv = y_vals.detach().cpu().numpy()
    zv = z_vals.detach().cpu().numpy()

    fig = plt.figure(figsize=(18,5))
    plt.suptitle(f'Trial {trial_id:03d} | Final Loss: {final_loss:.3e}', fontsize=16)

    ax1 = plt.subplot(1,3,1)
    im1 = ax1.imshow(pf_full_abs[x_idx,:,:].T, extent=[yv[0], yv[-1], zv[0], zv[-1]],
                     origin='lower', aspect='auto', cmap='viridis')
    ax1.scatter(target_y, target_z, c='red', marker='x', s=80, linewidths=2)
    ax1.set_title(f'YZ-plane (x={target_x:.3f} m)'); ax1.set_xlabel('y (m)'); ax1.set_ylabel('z (m)')
    plt.colorbar(im1, ax=ax1, label='|p|')

    ax2 = plt.subplot(1,3,2)
    im2 = ax2.imshow(pf_full_abs[:,y_idx,:].T, extent=[xv[0], xv[-1], zv[0], zv[-1]],
                     origin='lower', aspect='auto', cmap='viridis')
    ax2.scatter(target_x, target_z, c='red', marker='x', s=80, linewidths=2)
    ax2.set_title(f'XZ-plane (y={target_y:.3f} m)'); ax2.set_xlabel('x (m)'); ax2.set_ylabel('z (m)')
    plt.colorbar(im2, ax=ax2, label='|p|')

    ax3 = plt.subplot(1,3,3)
    im3 = ax3.imshow(pf_full_abs[:,:,z_idx].T, extent=[xv[0], xv[-1], yv[0], yv[-1]],
                     origin='lower', aspect='auto', cmap='viridis')
    ax3.scatter(target_x, target_y, c='red', marker='x', s=80, linewidths=2)
    ax3.set_title(f'XY-plane (z={target_z:.3f} m)'); ax3.set_xlabel('x (m)'); ax3.set_ylabel('y (m)')
    plt.colorbar(im3, ax=ax3, label='|p|')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def decision_space_slices(objective_fun, x_star, g_star, p_star, trial_id):
    # u = -g, v = p (u ⟂ v로 직교화)
    u = -g_star
    if np.linalg.norm(u) > 0:
        u = u / (np.linalg.norm(u)+1e-16)
    else:
        u = np.random.randn(*x_star.shape); u /= np.linalg.norm(u)+1e-16

    v = p_star
    if np.linalg.norm(v) > 0:
        v = v - np.dot(v, u)*u
        nv = np.linalg.norm(v)
        if nv > 0:
            v = v / nv
        else:
            z = np.random.randn(*x_star.shape); z -= np.dot(z, u)*u
            v = z / (np.linalg.norm(z)+1e-16)
    else:
        z = np.random.randn(*x_star.shape); z -= np.dot(z, u)*u
        v = z / (np.linalg.norm(z)+1e-16)

    # 1D: along u and v
    ts1 = np.linspace(-SLICE_H, SLICE_H, SLICE_N_1D)
    f_u = [objective_fun(x_star + t*u) for t in ts1]
    ts2 = np.linspace(-SLICE_H, SLICE_H, SLICE_N_1D)
    f_v = [objective_fun(x_star + t*v) for t in ts2]

    # 2D: on span {u, v}
    A = np.linspace(-SLICE_H, SLICE_H, SLICE_N_2D)
    B = np.linspace(-SLICE_H, SLICE_H, SLICE_N_2D)
    F = np.zeros((SLICE_N_2D, SLICE_N_2D))
    for i,a in enumerate(A):
        for j,b in enumerate(B):
            F[j,i] = objective_fun(x_star + a*u + b*v)  # (j,i) to align imshow

    fig = plt.figure(figsize=(16,5))
    plt.suptitle(f'Decision-Space Slices (trial {trial_id:03d})', fontsize=16)

    ax1 = plt.subplot(1,3,1)
    ax1.plot(ts1, f_u, linewidth=1.5)
    ax1.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax1.set_title('1D slice along -grad'); ax1.set_xlabel('t (rad)'); ax1.set_ylabel('f(x* + t u)')

    ax2 = plt.subplot(1,3,2)
    ax2.plot(ts2, f_v, linewidth=1.5)
    ax2.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_title('1D slice along BFGS dir'); ax2.set_xlabel('t (rad)'); ax2.set_ylabel('f(x* + t v)')

    ax3 = plt.subplot(1,3,3)
    im = ax3.imshow(F, extent=[A[0], A[-1], B[0], B[-1]], origin='lower', aspect='auto', cmap='viridis')
    ax3.scatter(0.0, 0.0, c='red', marker='x', s=60, linewidths=2)
    ax3.set_title('2D slice on span{u,v}')
    ax3.set_xlabel('a (rad)'); ax3.set_ylabel('b (rad)')
    plt.colorbar(im, ax=ax3, label='f')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# =========================== MAIN ===========================
def main():
    all_logs = []
    total_t0 = time.time()

    print(f"\n[Run] trials={num_trials}, steps/trial={num_opt_steps}, penalty={PRESSURE_PENALTY_MODE}")
    for trial in range(1, num_trials+1):
        print(f"\n--- Trial {trial:02d} ---")
        x0 = np.random.rand(num_transducers) * 2.0 * math.pi
        g0 = jacobian_for_scipy(x0)

        recorder = StepRecorder(trial_number=trial, x0_np=x0, g0_np=g0, total_steps=num_opt_steps)

        res = minimize(
            fun=objective_for_scipy,
            x0=x0,
            method='BFGS',
            jac=jacobian_for_scipy,
            callback=recorder,
            options={'maxiter': num_opt_steps, 'disp': False, 'gtol': -1.0}  # 원본 유지
        )

        all_logs.extend(recorder.logs)

        x_final = np.asarray(res.x, dtype=np.float64)
        g_final = jacobian_for_scipy(x_final)
        p_final = -recorder.H.dot(g_final)
        final_loss = objective_for_scipy(x_final)

        print("  [+] Computing full field slices ...")
        with torch.no_grad():
            pf_full = compute_pressure_field_batched(
                torch.tensor(x_final, dtype=torch.float64, device=device),
                amp_table, wavelength, x_vals, y_vals, z_vals, transducer_positions
            )
            pf_abs = torch.abs(pf_full).cpu().numpy()

        fig1 = plot_field_slices(pf_abs, x_idx, y_idx, z_idx, trial, final_loss)
        save1 = os.path.join(OUTPUT_DIR, f"trial_{trial:03d}_field_slices.png")
        fig1.savefig(save1, dpi=180); plt.close(fig1)
        print(f"  [saved] {save1}")

        print("  [+] Computing decision-space slices ...")
        fig2 = decision_space_slices(objective_for_scipy, x_final, g_final, p_final, trial)
        save2 = os.path.join(OUTPUT_DIR, f"trial_{trial:03d}_decision_slices.png")
        fig2.savefig(save2, dpi=180); plt.close(fig2)
        print(f"  [saved] {save2}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # CSV 저장
    df = pd.DataFrame(all_logs)
    cols = [
        'trial','step','time_s','loss','laplacian','gorkov_grad_mag','pressure_abs',
        'gnorm','pnorm','alpha_imp','cos_theta','r_ratio','sTy','cos_phi','gamma_scale',
        'lambda_min_est','lambda_max_est','kappa_est'
    ]
    df = df[cols]
    csv_path = os.path.join(OUTPUT_DIR, "optimization_steps_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV] saved: {csv_path}")

    # --- 최종 요약 ---
    df_final = df[df['step'] == num_opt_steps]
    if not df_final.empty:
        print("\n" + "="*50)
        print(f"Final-step summary (step={num_opt_steps}) over {df_final['trial'].nunique()} trials")
        print("="*50)
        for col in ['loss','laplacian','pressure_abs','gorkov_grad_mag','time_s']:
            mu = df_final[col].mean(); sd = df_final[col].std(); mn = df_final[col].min()
            print(f"{col:15s} | mean: {mu:.4e} | std: {sd:.4e} | min: {mn:.4e}")
        print("="*50)
    else:
        print("No final-step rows to summarize.")

    print(f"[Done] total time: {time.time()-total_t0:.2f}s")

if __name__ == "__main__":
    main()
