"""
PINN v4 — ensemble with soft IC/BC constraints.

Strategy to reduce variance:
  - 10 loss terms: PDE residual + IC value + IC velocity + BC + Sensor data
    + energy dissipation identity + monotonic energy decay + (1,1)-modal ODE
    + spatial symmetry penalty + higher-modes suppression (modes ≠ (1,1)
    projected to zero). All physics-only; no κ_true / no analytical solution
    leaks into training.
  - Fixed empirical weights (LOSS_WEIGHTS) — data and IC must dominate to
    avoid the trivial u≡0 solution. Adaptive schemes (Kendall uncertainty,
    GradNorm) collapse here because PDE is trivially satisfiable by u=0,
    and both methods then under-weight the hard constraints (data, IC).
  - Ensemble of 8 models with varied hyperparameters:
        4 vanilla MLPs (small + medium) + 4 ResNet blocks (deeper)
        × 2 kappa_init values = 8 configurations
  - 10 sensors (was 5 in v3), noise = 1e-3 (K-thermocouple precision)
  - L-BFGS phase reuses ALL 10 constraints on PRE-SAMPLED fixed batches
    (line search requires deterministic objective)
  - Aggregation by Boltzmann-weighted average + bootstrap CI

All outputs saved to  ./results_v4/

Run:  .venv/Scripts/python.exe example/pinn_homogeneous_v4_ensemble.py
"""

import math, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import autograd, optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


# ── Output folder & logger ────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results_v4"
PLOTS_DIR = RESULTS_DIR / "plots"
RUNS_DIR = RESULTS_DIR / "runs"
for d in (RESULTS_DIR, PLOTS_DIR, RUNS_DIR):
    d.mkdir(exist_ok=True, parents=True)

LOG_PATH = RESULTS_DIR / "log.txt"
LOG_FILE = open(LOG_PATH, "w", encoding="utf-8")


def log(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=LOG_FILE)
    LOG_FILE.flush()


# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
log(f"Results dir: {RESULTS_DIR}")


# ── Constants ─────────────────────────────────────────────────────────────────
PI = math.pi
KAPPA_TRUE = 1.2e-4
T_MAX = 500.0
LAM11 = 2 * PI ** 2
OMEGA0 = math.sqrt(LAM11)

KAPPA_MIN, KAPPA_MAX = 1e-7, 1e-2  # kept for reference only; not used in parametrization

# 10 sensors: 3×3 grid + 1 near-corner
SENSOR_XY = [
    (0.25, 0.25), (0.25, 0.50), (0.25, 0.75),
    (0.50, 0.25), (0.50, 0.50), (0.50, 0.75),
    (0.75, 0.25), (0.75, 0.50), (0.75, 0.75),
    (0.10, 0.10),
]
N_TIME = 500
NOISE_LEVEL = 1e-3   # 0.1% — realistic K-thermocouple

# Fixed loss weights — empirically tuned (v3 baseline + new physics terms).
# Data and IC must dominate to avoid the trivial u≡0 solution that satisfies
# every other constraint. PDE is reference at 1.
LOSS_NAMES = ['pde', 'ic1', 'ic2', 'bc', 'data', 'en', 'mo', 'md', 'sy', 'hm']
LOSS_WEIGHTS = [1.0, 200.0, 200.0, 200.0, 2000.0, 50.0, 20.0, 200.0, 50.0, 500.0]
N_LOSS = len(LOSS_NAMES)

# Modes other than (1,1) that are allowed by central+diagonal symmetry —
# their projections should be identically zero for the true solution
# (linear PDE preserves modes; IC excites only (1,1)).
FORBIDDEN_MODES = [(2, 2), (1, 3), (3, 1), (3, 3)]

N_T_ENERGY, N_XY_ENERGY = 4, 1500   # sub-sampling for energy integrals
N_T_MODAL,  N_XY_MODAL  = 4, 1500   # sub-sampling for modal-ODE integrals
N_T_HM,     N_XY_HM     = 4, 1500   # sub-sampling for higher-modes projection
N_SYM = 2000                        # batch for symmetry penalty
ADAM_EPOCHS = 20000
LBFGS_STEPS = 300

# Ensemble: 4 vanilla MLPs + 4 ResNet variants × 2 κ_init = 8 models
ENSEMBLE_CONFIGS = [
    # Vanilla MLP (fast convergence, smaller nets)
    {'id': 'M1', 'hidden': 64,  'layers': 3, 'kappa_init': 1e-4, 'residual': False, 'seed': 1},
    {'id': 'M2', 'hidden': 64,  'layers': 3, 'kappa_init': 1e-3, 'residual': False, 'seed': 2},
    {'id': 'M3', 'hidden': 128, 'layers': 4, 'kappa_init': 1e-4, 'residual': False, 'seed': 3},
    {'id': 'M4', 'hidden': 128, 'layers': 4, 'kappa_init': 1e-3, 'residual': False, 'seed': 4},
    # ResNet (stable gradients in deeper nets)
    {'id': 'M5', 'hidden': 64,  'layers': 4, 'kappa_init': 1e-4, 'residual': True,  'seed': 5},
    {'id': 'M6', 'hidden': 64,  'layers': 4, 'kappa_init': 1e-3, 'residual': True,  'seed': 6},
    {'id': 'M7', 'hidden': 128, 'layers': 6, 'kappa_init': 1e-4, 'residual': True,  'seed': 7},
    {'id': 'M8', 'hidden': 128, 'layers': 6, 'kappa_init': 1e-3, 'residual': True,  'seed': 8},
]


# ── Analytical solution (data generation + validation only) ───────────────────
class AnalyticalSolution:
    def __init__(self, kappa=KAPPA_TRUE):
        lam = LAM11
        disc = (kappa * lam) ** 2 - 4 * lam
        self.alpha = kappa * lam / 2
        self.beta = math.sqrt(-disc) / 2

    def amplitude(self, t):
        return np.exp(-self.alpha * t) * (
            np.cos(self.beta * t) + (self.alpha / self.beta) * np.sin(self.beta * t))

    def __call__(self, x, y, t):
        if torch.is_tensor(x):
            x, y, t = (v.detach().cpu().numpy().flatten() for v in (x, y, t))
        return self.amplitude(np.asarray(t)) * np.sin(PI * x) * np.sin(PI * y)


analytical = AnalyticalSolution()


# ── ResNet block (used when cfg['residual'] = True) ───────────────────────────
class ResBlock(nn.Module):
    """
    Pre-activation residual block (He et al. 2016 style):
        out = tanh(x + W2 · tanh(W1 · x))
    Skip-connection bypasses two linears; tanh wraps the sum to keep values bounded.
    """
    def __init__(self, hidden):
        super().__init__()
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        for m in (self.lin1, self.lin2):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = torch.tanh(self.lin1(x))
        h = self.lin2(h)
        return torch.tanh(x + h)


# ── PINN with soft IC/BC + fixed-weight loss ──────────────────────────────────
class PINN(nn.Module):
    """
    Standard MLP output θ(x, y, t) ∈ ℝ.
    IC/BC enforced via soft loss terms with fixed weights (LOSS_WEIGHTS).
    No use of κ_true or analytical time evolution.

    If `residual=True`, the (n_layers-1) hidden blocks are ResBlocks (each has
    2 linears + skip connection). Otherwise, plain Linear+Tanh layers.
    """

    def __init__(self, hidden=128, n_layers=6, kappa_init=1e-3,
                 residual=False, t_max=T_MAX):
        super().__init__()
        self.t_max = t_max
        self.omega0 = OMEGA0
        self.residual = residual

        in_dim = 5
        self.input_layer = nn.Linear(in_dim, hidden)
        if residual:
            self.hidden_blocks = nn.ModuleList(
                [ResBlock(hidden) for _ in range(n_layers - 1)])
        else:
            self.hidden_blocks = nn.ModuleList(
                [nn.Linear(hidden, hidden) for _ in range(n_layers - 1)])
        self.output_layer = nn.Linear(hidden, 1)

        # Xavier init for all linears in vanilla parts (ResBlock self-initializes)
        for m in (self.input_layer, self.output_layer):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        if not residual:
            for lin in self.hidden_blocks:
                nn.init.xavier_normal_(lin.weight)
                nn.init.zeros_(lin.bias)

        self.log_kappa = nn.Parameter(torch.tensor(math.log(kappa_init), dtype=torch.float32))

    def forward(self, x, y, t):
        s = torch.sin(self.omega0 * t)
        c = torch.cos(self.omega0 * t)
        tau = t / self.t_max
        h = torch.tanh(self.input_layer(torch.cat([x, y, s, c, tau], dim=1)))
        for block in self.hidden_blocks:
            h = block(h) if self.residual else torch.tanh(block(h))
        return self.output_layer(h)

    def get_kappa(self):
        return torch.exp(self.log_kappa)


# ── PDE residual ──────────────────────────────────────────────────────────────
def _g(u, v):
    return autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                         create_graph=True, retain_graph=True)[0]


def pde_res(model, x, y, t):
    u = model(x, y, t)
    ux, uy, ut = _g(u, x), _g(u, y), _g(u, t)
    uxx, uyy, utt = _g(ux, x), _g(uy, y), _g(ut, t)
    uxxt, uyyt = _g(uxx, t), _g(uyy, t)
    return utt - model.get_kappa() * (uxxt + uyyt) - (uxx + uyy)


def _energy_one(model, x, y, t, kappa, er_list, mo_list):
    u = model(x, y, t)
    u_t = _g(u, t)
    u_x = _g(u, x); u_y = _g(u, y)
    u_xt = _g(u_x, t); u_yt = _g(u_y, t)
    E = 0.5 * (u_t.pow(2) + u_x.pow(2) + u_y.pow(2)).mean()
    D = (u_xt.pow(2) + u_yt.pow(2)).mean()
    dEdt = autograd.grad(E, t, create_graph=True, retain_graph=True)[0].sum()
    er_list.append((dEdt + kappa * D).pow(2))
    mo_list.append(torch.relu(dEdt).pow(2))


def energy_residual(model, fixed=None, n_t=N_T_ENERGY, n_xy=N_XY_ENERGY):
    """
    Global physical constraints derived from PDE + BC (θ=0 on ∂Ω):

        E(t) = ½∫(θ_t² + |∇θ|²) dΩ      — total energy
        D(t) = ∫|∇θ_t|² dΩ ≥ 0           — dissipation rate
        Identity (multiplying PDE by θ_t and integrating by parts):
            dE/dt + κ·D(t) = 0
        Consequence (since κ > 0, D ≥ 0):
            dE/dt ≤ 0                    — monotonic energy decay

    Returns:
        l_energy : MSE of (dE/dt + κ·D)  — exact algebraic constraint on κ
        l_mono   : MSE of relu(dE/dt)    — energy non-increasing penalty

    Both use only the problem statement, no κ_true / no analytical solution.
    Estimates each integral by Monte-Carlo over Ω at a fixed t.
    """
    er_list, mo_list = [], []
    kappa = model.get_kappa()
    if fixed is None:
        for _ in range(n_t):
            tv = float(torch.rand(1, device=device).item()) * T_MAX
            x = torch.rand(n_xy, 1, requires_grad=True, device=device)
            y = torch.rand(n_xy, 1, requires_grad=True, device=device)
            t = torch.full((n_xy, 1), tv, device=device, requires_grad=True)
            _energy_one(model, x, y, t, kappa, er_list, mo_list)
    else:
        for (tv, x_fix, y_fix) in fixed:
            x = x_fix.detach().requires_grad_(True)
            y = y_fix.detach().requires_grad_(True)
            t = torch.full((x.shape[0], 1), tv, device=device, requires_grad=True)
            _energy_one(model, x, y, t, kappa, er_list, mo_list)
    return torch.stack(er_list).mean(), torch.stack(mo_list).mean()


def _modal_one(model, x, y, t, kappa, res):
    u = model(x, y, t)
    phi = torch.sin(PI * x) * torch.sin(PI * y)
    A = 4.0 * (u * phi).mean()
    dA_vec = autograd.grad(A, t, create_graph=True, retain_graph=True)[0]
    dA = dA_vec.sum()
    d2A_vec = autograd.grad(dA, t, create_graph=True, retain_graph=True)[0]
    d2A = d2A_vec.sum()
    res.append((d2A + kappa * LAM11 * dA + LAM11 * A).pow(2))


def modal_ode_residual(model, fixed=None, n_t=N_T_MODAL, n_xy=N_XY_MODAL):
    """
    Modal (1,1) coefficient
        A(t) = 4 · ∫∫ u(x,y,t) · sin(πx) sin(πy) dx dy
    must satisfy the 1-D ODE (linear PDE → modes decouple):
        A''(t) + κ · LAM11 · A'(t) + LAM11 · A(t) = 0
    with A(0)=1, A'(0)=0 (from the IC).

    LAM11 = 2π² is a pure geometric eigenvalue of −Δ on (0,1)² with
    Dirichlet BC; nothing to do with κ_true or the analytical solution.
    """
    res = []
    kappa = model.get_kappa()
    if fixed is None:
        for _ in range(n_t):
            tv = float(torch.rand(1, device=device).item()) * T_MAX
            x = torch.rand(n_xy, 1, device=device)
            y = torch.rand(n_xy, 1, device=device)
            t = torch.full((n_xy, 1), tv, device=device, requires_grad=True)
            _modal_one(model, x, y, t, kappa, res)
    else:
        for (tv, x_fix, y_fix) in fixed:
            x = x_fix.detach()
            y = y_fix.detach()
            t = torch.full((x.shape[0], 1), tv, device=device, requires_grad=True)
            _modal_one(model, x, y, t, kappa, res)
    return torch.stack(res).mean()


def symmetry_residual(model, fixed=None, n=N_SYM):
    """
    For homogeneous IC sin(πx)sin(πy) on (0,1)² the solution preserves:
        (i)  central symmetry  u(x, y, t) = u(1−x, 1−y, t)
        (ii) diagonal symmetry u(x, y, t) = u(y, x, t)
    Penalty halves the effective solution manifold, free regularizer.
    """
    if fixed is None:
        x = torch.rand(n, 1, device=device)
        y = torch.rand(n, 1, device=device)
        t = torch.rand(n, 1, device=device) * T_MAX
    else:
        x, y, t = fixed
    u = model(x, y, t)
    u_cs = model(1.0 - x, 1.0 - y, t)
    u_dg = model(y, x, t)
    return ((u - u_cs).pow(2) + (u - u_dg).pow(2)).mean()


def higher_modes_residual(model, fixed=None, n_t=N_T_HM, n_xy=N_XY_HM):
    """
    Project u(x,y,t) onto every (m,n) ∈ FORBIDDEN_MODES (modes other than
    (1,1) that are allowed by central + diagonal symmetry). For the true
    solution all these projections are identically zero — linear PDE keeps
    each mode in its own ODE, and IC excites only (1,1).

        B_{mn}(t) := 4 · ∫∫ u(x,y,t) · sin(mπx) sin(nπy) dx dy

    Penalty: mean over modes and t of B_{mn}². Algebraic (no autograd).
    """
    res = []
    if fixed is None:
        batches = [(float(torch.rand(1, device=device).item()) * T_MAX,
                    torch.rand(n_xy, 1, device=device),
                    torch.rand(n_xy, 1, device=device))
                   for _ in range(n_t)]
    else:
        batches = fixed
    for (tv, x, y) in batches:
        t = torch.full((x.shape[0], 1), tv, device=device)
        u = model(x, y, t)
        for (m, n) in FORBIDDEN_MODES:
            phi = torch.sin(m * PI * x) * torch.sin(n * PI * y)
            B = 4.0 * (u * phi).mean()
            res.append(B.pow(2))
    return torch.stack(res).mean()


# ── Pre-samplers for L-BFGS fixed batches ─────────────────────────────────────
def sample_fixed_energy():
    return [(float(torch.rand(1, device=device).item()) * T_MAX,
             torch.rand(N_XY_ENERGY, 1, device=device),
             torch.rand(N_XY_ENERGY, 1, device=device))
            for _ in range(N_T_ENERGY)]


def sample_fixed_modal():
    return [(float(torch.rand(1, device=device).item()) * T_MAX,
             torch.rand(N_XY_MODAL, 1, device=device),
             torch.rand(N_XY_MODAL, 1, device=device))
            for _ in range(N_T_MODAL)]


def sample_fixed_sym():
    return (torch.rand(N_SYM, 1, device=device),
            torch.rand(N_SYM, 1, device=device),
            torch.rand(N_SYM, 1, device=device) * T_MAX)


def sample_fixed_higher_modes():
    return [(float(torch.rand(1, device=device).item()) * T_MAX,
             torch.rand(N_XY_HM, 1, device=device),
             torch.rand(N_XY_HM, 1, device=device))
            for _ in range(N_T_HM)]


# ── Samplers ──────────────────────────────────────────────────────────────────
def spde(n):
    x = torch.rand(n, 1, requires_grad=True, device=device)
    y = torch.rand(n, 1, requires_grad=True, device=device)
    t = torch.rand(n, 1, requires_grad=True, device=device) * T_MAX
    return x, y, t


def sic(n):
    x = torch.rand(n, 1, requires_grad=True, device=device)
    y = torch.rand(n, 1, requires_grad=True, device=device)
    t = torch.zeros(n, 1, requires_grad=True, device=device)
    return x, y, t


def sbc(n):
    k = n // 4
    randt = lambda: torch.rand(k, 1, requires_grad=True, device=device) * T_MAX
    xL = torch.zeros(k, 1, requires_grad=True, device=device)
    xR = torch.ones(k, 1, requires_grad=True, device=device)
    y0 = torch.zeros(k, 1, requires_grad=True, device=device)
    y1 = torch.ones(k, 1, requires_grad=True, device=device)
    return (
        torch.cat([xL, xR,
                   torch.rand(k, 1, requires_grad=True, device=device),
                   torch.rand(k, 1, requires_grad=True, device=device)]),
        torch.cat([torch.rand(k, 1, requires_grad=True, device=device),
                   torch.rand(k, 1, requires_grad=True, device=device), y0, y1]),
        torch.cat([randt(), randt(), randt(), randt()])
    )


# ── Loss (PDE + IC + BC + Data) ───────────────────────────────────────────────
mse = nn.MSELoss()


def compute_individual_losses(model, xd, yd, td, dn,
                              fp=None, fi=None, fb=None,
                              fe=None, fm=None, fs=None, fh=None,
                              include_global=True):
    """
    Returns the N_LOSS raw loss components (no weighting). When
    `include_global` is False, the last 5 (en, mo, md, sy, hm) are 0 tensors.
    Pre-sampled batches (fp,fi,fb,fe,fm,fs,fh) make the call deterministic
    — used by L-BFGS so that strong-Wolfe line search sees a fixed objective.
    """
    if fp is None: xp, yp, tp = spde(10000)
    else:          xp, yp, tp = (v.detach().requires_grad_(True) for v in fp)
    if fi is None: xi, yi, ti = sic(3000)
    else:          xi, yi, ti = (v.detach().requires_grad_(True) for v in fi)
    if fb is None: xb, yb, tb = sbc(3000)
    else:          xb, yb, tb = (v.detach().requires_grad_(True) for v in fb)

    l_pde = pde_res(model, xp, yp, tp).pow(2).mean()
    u_ic = model(xi, yi, ti)
    l_ic1 = mse(u_ic, torch.sin(PI * xi) * torch.sin(PI * yi))
    l_ic2 = _g(u_ic, ti).pow(2).mean()
    l_bc = model(xb, yb, tb).pow(2).mean()
    l_data = mse(model(xd, yd, td), dn)

    if include_global:
        l_energy, l_mono = energy_residual(model, fixed=fe)
        l_modal = modal_ode_residual(model, fixed=fm)
        l_sym = symmetry_residual(model, fixed=fs)
        l_hm = higher_modes_residual(model, fixed=fh)
    else:
        zero = torch.zeros((), device=device)
        l_energy = l_mono = l_modal = l_sym = l_hm = zero

    return (l_pde, l_ic1, l_ic2, l_bc, l_data,
            l_energy, l_mono, l_modal, l_sym, l_hm)


_WEIGHTS_T = torch.tensor(LOSS_WEIGHTS, dtype=torch.float32, device=device)


def weighted_total(losses, include_global=True):
    idx = list(range(N_LOSS)) if include_global else [0, 1, 2, 3, 4]
    stacked = torch.stack([losses[i] for i in idx])
    return (_WEIGHTS_T[idx] * stacked).sum()


def compute_loss(model, xd, yd, td, dn,
                 fp=None, fi=None, fb=None,
                 fe=None, fm=None, fs=None, fh=None,
                 include_global=True):
    losses = compute_individual_losses(model, xd, yd, td, dn,
                                       fp=fp, fi=fi, fb=fb,
                                       fe=fe, fm=fm, fs=fs, fh=fh,
                                       include_global=include_global)
    total = weighted_total(losses, include_global=include_global)
    return (total, *losses)


# ── Sensor data ───────────────────────────────────────────────────────────────
def make_sensor_data(seed=0):
    rng = np.random.default_rng(seed)
    t_s = np.linspace(0.0, T_MAX, N_TIME)
    xs, ys, ts, ds = [], [], [], []
    for sx, sy in SENSOR_XY:
        vals = analytical(np.full(N_TIME, sx), np.full(N_TIME, sy), t_s)
        xs.append(np.full(N_TIME, sx)); ys.append(np.full(N_TIME, sy))
        ts.append(t_s); ds.append(vals)
    xs = np.concatenate(xs).reshape(-1, 1)
    ys = np.concatenate(ys).reshape(-1, 1)
    ts = np.concatenate(ts).reshape(-1, 1)
    ds = np.concatenate(ds).reshape(-1, 1)
    noise = rng.standard_normal(ds.shape) * NOISE_LEVEL * np.abs(ds)
    ds_noisy = ds + noise
    return (torch.tensor(xs, dtype=torch.float32, device=device),
            torch.tensor(ys, dtype=torch.float32, device=device),
            torch.tensor(ts, dtype=torch.float32, device=device),
            torch.tensor(ds_noisy, dtype=torch.float32, device=device))


# ── Single model training ─────────────────────────────────────────────────────
def train_one(cfg, xd, yd, td, dn, run_dir):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])

    model = PINN(hidden=cfg['hidden'], n_layers=cfg['layers'],
                 kappa_init=cfg['kappa_init'],
                 residual=cfg.get('residual', False)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    arch = f"{cfg['hidden']}×{cfg['layers']}{'/Res' if cfg.get('residual') else ''}"
    label = f"{cfg['id']} ({arch}, κ_init={cfg['kappa_init']:.0e})"
    log(f"\n[{label}]  params={n_params}  seed={cfg['seed']}  κ_init={model.get_kappa().item():.4e}")

    loss_hist, kappa_hist = [], []
    pde_hist, data_hist = [], []
    best_loss = float('inf')
    best_sd = None
    t0 = time.time()

    # Adam
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1000, min_lr=1e-6)

    for ep in range(ADAM_EPOCHS):
        opt.zero_grad()
        L, l_pde, l_ic1, l_ic2, l_bc, l_data, l_en, l_mo, l_md, l_sy, l_hm = compute_loss(model, xd, yd, td, dn)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(L)

        loss_hist.append(L.item())
        kappa_hist.append(model.get_kappa().item())
        pde_hist.append(l_pde.item())
        data_hist.append(l_data.item())

        if L.item() < best_loss:
            best_loss = L.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 1000 == 0:
            log(f"  [Adam]   ep={ep:5d}  loss={L.item():.4e}  "
                f"pde={l_pde.item():.2e}  ic={l_ic1.item():.2e}  "
                f"data={l_data.item():.2e}  en={l_en.item():.2e}  "
                f"md={l_md.item():.2e}  sy={l_sy.item():.2e}  hm={l_hm.item():.2e}  "
                f"κ={model.get_kappa().item():.4e}  lr={opt.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_sd)

    # L-BFGS — full 10-term loss on FIXED batches (line search needs
    # deterministic objective).
    fp = spde(10000); fi = sic(3000); fb = sbc(3000)
    fe = sample_fixed_energy()
    fm = sample_fixed_modal()
    fs = sample_fixed_sym()
    fh = sample_fixed_higher_modes()

    lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=50,
                        history_size=100, line_search_fn='strong_wolfe')

    def closure():
        lbfgs.zero_grad()
        L, *_ = compute_loss(model, xd, yd, td, dn,
                             fp=fp, fi=fi, fb=fb,
                             fe=fe, fm=fm, fs=fs, fh=fh,
                             include_global=True)
        L.backward()
        return L

    for step in range(LBFGS_STEPS):
        try:
            L = lbfgs.step(closure)
        except Exception as e:
            log(f"  [LBFGS]  step={step:3d}  EXCEPTION: {type(e).__name__}: {e} — stopping L-BFGS")
            break
        if L is None or not math.isfinite(L.item()):
            log(f"  [LBFGS]  step={step:3d}  non-finite/None loss — stopping L-BFGS")
            break
        loss_hist.append(L.item())
        kappa_hist.append(model.get_kappa().item())
        if L.item() < best_loss:
            best_loss = L.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if step % 20 == 0:
            log(f"  [LBFGS]  step={step:3d}  loss={L.item():.4e}  "
                f"κ={model.get_kappa().item():.4e}")

    model.load_state_dict(best_sd)
    elapsed = time.time() - t0

    # Final metrics
    kf = model.get_kappa().item()
    err = abs(kf - KAPPA_TRUE) / KAPPA_TRUE * 100

    model.eval()
    with torch.no_grad():
        xt = torch.rand(5000, 1, device=device)
        yt = torch.rand(5000, 1, device=device)
        tt = torch.rand(5000, 1, device=device) * T_MAX
        up = model(xt, yt, tt)
    ue = torch.tensor(analytical(xt.cpu(), yt.cpu(), tt.cpu()).reshape(-1, 1),
                      dtype=torch.float32)
    l2 = (torch.norm(up.cpu() - ue) / torch.norm(ue)).item()
    linf = torch.max(torch.abs(up.cpu() - ue)).item()

    # Persist
    torch.save(best_sd, run_dir / "state_dict.pth")
    np.save(run_dir / "loss_history.npy", np.array(loss_hist))
    np.save(run_dir / "kappa_history.npy", np.array(kappa_hist))
    np.save(run_dir / "pde_history.npy", np.array(pde_hist))
    np.save(run_dir / "data_history.npy", np.array(data_hist))
    metrics = {**cfg, 'n_params': n_params, 'loss': best_loss,
               'kappa': kf, 'err_pct': err, 'l2_rel': l2, 'linf': linf,
               'time_min': elapsed / 60}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log(f"[{label}] DONE  loss={best_loss:.4e}  κ={kf:.4e}  err={err:.2f}%  "
        f"L2={l2:.3e}  t={elapsed/60:.1f}min")

    return {**metrics, 'model_state': best_sd,
            'loss_hist': loss_hist, 'kappa_hist': kappa_hist,
            'pde_hist': pde_hist, 'data_hist': data_hist}


# ── Ensemble aggregation ──────────────────────────────────────────────────────
def boltzmann_weights(losses, T_ratio=1.0):
    """
    w_i = exp(-(L_i - L_min) / T),   T = T_ratio * (L_max - L_min) or median.
    Smaller loss -> larger weight. Normalizes to sum=1.
    """
    losses = np.asarray(losses)
    L_shift = losses - losses.min()
    T = T_ratio * (losses.max() - losses.min()) if losses.max() > losses.min() else losses.min()
    if T <= 0:
        T = 1e-12
    w = np.exp(-L_shift / T)
    return w / w.sum()


def bootstrap_ci(kappas, weights, n_boot=5000, alpha=0.05):
    """Bootstrap percentile CI for the weighted mean."""
    n = len(kappas)
    rng = np.random.default_rng(42)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        w = weights[idx]; k = kappas[idx]
        boot[b] = (w * k).sum() / w.sum()
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return lo, hi, boot


# ── Plotting ──────────────────────────────────────────────────────────────────
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def _label(r):
    suffix = '/Res' if r.get('residual') else ''
    return f"{r['id']} ({r['hidden']}×{r['layers']}{suffix}, κ₀={r['kappa_init']:.0e})"


def plot_loss_history(results):
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(results):
        ax.semilogy(r['loss_hist'], lw=0.7, color=COLORS[i], label=_label(r))
    ax.axvline(ADAM_EPOCHS, color='k', ls='--', lw=0.8, alpha=0.5, label='Adam → L-BFGS')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Total loss')
    ax.set_title('Loss convergence — ensemble of 8 models')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_history.png", dpi=150); plt.close()


def plot_kappa_history(results):
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(results):
        ax.semilogy(r['kappa_hist'], lw=0.8, color=COLORS[i], label=_label(r))
    ax.axhline(KAPPA_TRUE, color='k', ls=':', lw=1.5, label=f'κ_true = {KAPPA_TRUE:.2e}')
    ax.axhline(KAPPA_MIN, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(KAPPA_MAX, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('Iteration'); ax.set_ylabel('κ (log scale)')
    ax.set_title('κ trajectory across ensemble')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kappa_history.png", dpi=150); plt.close()


def plot_ensemble_estimate(results, weights, k_mean, ci_low, ci_high, boot_samples):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-model kappa with weights
    ax = axes[0]
    kappas = np.array([r['kappa'] for r in results])
    xs = np.arange(1, len(kappas) + 1)
    sc = ax.scatter(xs, kappas, s=200*np.array(weights)+20, c=weights,
                    cmap='viridis', zorder=3, edgecolor='k')
    for x, k, w, r in zip(xs, kappas, weights, results):
        ax.text(x, k * 1.06, f'κ={k:.2e}\nw={w:.2f}',
                ha='center', fontsize=8)
    ax.axhline(KAPPA_TRUE, color='k', ls=':', lw=1.2, label=f'κ_true = {KAPPA_TRUE:.3e}')
    ax.axhline(k_mean, color='r', ls='-', lw=1.4, label=f'Boltzmann mean = {k_mean:.3e}')
    ax.fill_between([0.5, len(kappas) + 0.5], ci_low, ci_high,
                    color='r', alpha=0.15,
                    label=f'95% CI [{ci_low:.2e}, {ci_high:.2e}]')
    ax.set_xticks(xs)
    ax.set_xticklabels([r['id'] for r in results])
    ax.set_ylabel('κ')
    ax.set_title('Ensemble κ estimates (point size = weight)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='loss-weight')

    # Right: bootstrap distribution
    ax = axes[1]
    ax.hist(boot_samples, bins=60, alpha=0.7, color='steelblue', edgecolor='k')
    ax.axvline(KAPPA_TRUE, color='k', ls=':', lw=2, label=f'κ_true = {KAPPA_TRUE:.3e}')
    ax.axvline(k_mean, color='r', ls='-', lw=2, label=f'mean = {k_mean:.3e}')
    ax.axvline(ci_low, color='r', ls='--', lw=1.5, alpha=0.7)
    ax.axvline(ci_high, color='r', ls='--', lw=1.5, alpha=0.7, label='95% CI')
    ax.set_xlabel('κ'); ax.set_ylabel('Bootstrap density')
    ax.set_title('Bootstrap posterior of ensemble κ (5000 samples)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kappa_ensemble.png", dpi=150); plt.close()


def plot_weights(results, weights):
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [f"{r['id']}\n{r['hidden']}×{r['layers']}"
              f"{'/Res' if r.get('residual') else ''}"
              f"\nκ₀={r['kappa_init']:.0e}" for r in results]
    bars = ax.bar(range(len(weights)), weights, color='steelblue', edgecolor='k')
    for b, w in zip(bars, weights):
        ax.text(b.get_x() + b.get_width()/2, w + 0.005, f'{w:.3f}',
                ha='center', fontsize=9)
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Loss-weight (softmax)')
    ax.set_title('Ensemble model weights (higher = better physics fit)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ensemble_weights.png", dpi=150); plt.close()


def plot_amplitude(best_model):
    t_plot = np.linspace(0, T_MAX, 1000)
    A_exact = analytical.amplitude(t_plot)
    x05 = torch.full((1000, 1), 0.5, device=device)
    y05 = torch.full((1000, 1), 0.5, device=device)
    tp = torch.tensor(t_plot.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        u = best_model(x05, y05, tp).cpu().numpy().flatten()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t_plot, A_exact, 'k-', lw=1.3, label='Analytical A(t)')
    axes[0].plot(t_plot, u, 'r--', lw=1.0, label='PINN A(t)')
    axes[0].set_ylabel('A(t) at (0.5, 0.5)')
    axes[0].set_title('Temporal amplitude — best ensemble model (by loss)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(t_plot, np.abs(u - A_exact) + 1e-12, 'b-', lw=0.7)
    axes[1].set_xlabel('t'); axes[1].set_ylabel('|A_PINN - A_exact|')
    axes[1].grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "amplitude_compare.png", dpi=150); plt.close()


def plot_field_snapshots(best_model):
    t_snaps = [1.0, 10.0, 50.0, 100.0, 250.0, 500.0]
    n_sp = 60
    x_sp = torch.linspace(0, 1, n_sp, device=device)
    y_sp = torch.linspace(0, 1, n_sp, device=device)
    X, Y = torch.meshgrid(x_sp, y_sp, indexing='ij')
    Xf, Yf = X.reshape(-1, 1), Y.reshape(-1, 1)

    fig, axes = plt.subplots(3, len(t_snaps), figsize=(20, 9))
    for i, t_val in enumerate(t_snaps):
        Tf = torch.full_like(Xf, t_val)
        with torch.no_grad():
            up = best_model(Xf, Yf, Tf).reshape(n_sp, n_sp).cpu().numpy()
        ua = analytical(Xf.cpu(), Yf.cpu(), Tf.cpu()).reshape(n_sp, n_sp)
        diff = up - ua
        vmin, vmax = min(up.min(), ua.min()), max(up.max(), ua.max())
        dlim = max(abs(diff.min()), abs(diff.max()), 1e-12)

        for r, (data, title) in enumerate(zip([up, ua, diff],
                                              [f'PINN t={t_val:g}',
                                               f'Exact t={t_val:g}',
                                               f'Diff max={dlim:.2e}'])):
            kwargs = {'cmap': 'RdBu_r'}
            if r < 2: kwargs.update(vmin=vmin, vmax=vmax)
            else:     kwargs.update(vmin=-dlim, vmax=dlim)
            c = axes[r, i].contourf(X.cpu(), Y.cpu(), data, 30, **kwargs)
            axes[r, i].set_title(title)
            plt.colorbar(c, ax=axes[r, i], fraction=0.046, pad=0.04)
            axes[r, i].set_xlabel('x'); axes[r, i].set_ylabel('y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "field_snapshots.png", dpi=130); plt.close()


def plot_pde_residual(best_model):
    n_sp = 60
    x_sp = torch.linspace(0, 1, n_sp, device=device)
    y_sp = torch.linspace(0, 1, n_sp, device=device)
    X, Y = torch.meshgrid(x_sp, y_sp, indexing='ij')
    Xf = X.reshape(-1, 1).requires_grad_(True)
    Yf = Y.reshape(-1, 1).requires_grad_(True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, t_val in enumerate([10.0, 100.0, 400.0]):
        Tf = torch.full_like(Xf, t_val).requires_grad_(True)
        res = pde_res(best_model, Xf, Yf, Tf).detach().reshape(n_sp, n_sp).cpu().numpy()
        cb = axes[i].contourf(X.cpu(), Y.cpu(), np.abs(res), 30, cmap='hot')
        axes[i].set_title(f'|PDE residual|  t={t_val:g}')
        axes[i].set_xlabel('x'); axes[i].set_ylabel('y')
        plt.colorbar(cb, ax=axes[i], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pde_residual.png", dpi=150); plt.close()


def plot_time_error(best_model):
    times = np.linspace(1.0, T_MAX, 50)
    l2s, linfs = [], []
    for tval in times:
        with torch.no_grad():
            xt = torch.rand(2000, 1, device=device)
            yt = torch.rand(2000, 1, device=device)
            tt = torch.full_like(xt, float(tval))
            up = best_model(xt, yt, tt)
        ue = torch.tensor(analytical(xt.cpu(), yt.cpu(), tt.cpu()).reshape(-1, 1),
                          dtype=torch.float32)
        l2s.append((torch.norm(up.cpu() - ue) / (torch.norm(ue) + 1e-12)).item())
        linfs.append(torch.max((up.cpu() - ue).abs()).item())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(times, l2s, 'b-', label='L2 relative')
    ax.semilogy(times, linfs, 'r--', label='L∞ absolute')
    ax.set_xlabel('t'); ax.set_ylabel('Error')
    ax.set_title('Field error vs time (best ensemble model)')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "time_error.png", dpi=150); plt.close()


def plot_arch_comparison(results):
    """Group models by architecture (incl. residual flag); show kappa range per group."""
    groups = {}
    for r in results:
        key = f"{r['hidden']}×{r['layers']}{'/Res' if r.get('residual') else ''}"
        groups.setdefault(key, []).append(r['kappa'])

    fig, ax = plt.subplots(figsize=(9, 5))
    keys = list(groups.keys())
    for i, k in enumerate(keys):
        vals = groups[k]
        ax.scatter([i]*len(vals), vals, s=80, alpha=0.7)
        ax.hlines(np.mean(vals), i - 0.2, i + 0.2, color='r', lw=2)
    ax.axhline(KAPPA_TRUE, color='k', ls=':', lw=1.5, label=f'κ_true = {KAPPA_TRUE:.2e}')
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys)
    ax.set_xlabel('Architecture (hidden × layers)')
    ax.set_ylabel('Identified κ')
    ax.set_title('Architecture sensitivity of κ identification')
    ax.set_yscale('log')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "architecture_comparison.png", dpi=150); plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
log("\n" + "="*72)
log("PINN v4 — ENSEMBLE  (soft IC/BC constraints, 10 sensors, noise=1e-3)")
log(f"  Sensors: {len(SENSOR_XY)},  N_TIME: {N_TIME},  noise: {NOISE_LEVEL:.0e}")
log(f"  κ in [{KAPPA_MIN:.0e}, {KAPPA_MAX:.0e}]")
log(f"  Adam={ADAM_EPOCHS}, LBFGS={LBFGS_STEPS}")
log(f"  Ensemble size: {len(ENSEMBLE_CONFIGS)}")
log("="*72)

xd, yd, td, dn = make_sensor_data(seed=0)
log(f"Sensor data: {dn.shape[0]} points, "
    f"|signal| range [{dn.abs().min().item():.3e}, {dn.abs().max().item():.3e}]")

results = []
for cfg in ENSEMBLE_CONFIGS:
    run_dir = RUNS_DIR / cfg['id']
    run_dir.mkdir(exist_ok=True)
    res = train_one(cfg, xd, yd, td, dn, run_dir)
    results.append(res)

# ── Ensemble aggregation ──────────────────────────────────────────────────────
losses = np.array([r['loss'] for r in results])
kappas = np.array([r['kappa'] for r in results])
errs = np.array([r['err_pct'] for r in results])
l2s = np.array([r['l2_rel'] for r in results])

weights = boltzmann_weights(losses, T_ratio=1.0)
k_mean = float((weights * kappas).sum())
k_std = float(np.sqrt((weights * (kappas - k_mean) ** 2).sum()))
k_err = abs(k_mean - KAPPA_TRUE) / KAPPA_TRUE * 100

ci_low, ci_high, boot_samples = bootstrap_ci(kappas, weights, n_boot=5000)

best = min(results, key=lambda r: r['loss'])
best_idx = results.index(best)

# ── Summary ───────────────────────────────────────────────────────────────────
log("\n" + "="*72)
log("ENSEMBLE SUMMARY")
log("="*72)
log(f"  {'ID':<5}{'arch':<13}{'κ₀':<10}{'loss':<14}{'κ':<14}{'err%':<10}{'L2':<12}{'weight':<10}")
for i, r in enumerate(results):
    mark = "  *BEST" if r is best else ""
    arch = f"{r['hidden']}×{r['layers']}{'/Res' if r.get('residual') else ''}"
    log(f"  {r['id']:<5}{arch:<13}{r['kappa_init']:<10.0e}"
        f"{r['loss']:<14.4e}{r['kappa']:<14.4e}{r['err_pct']:<10.2f}"
        f"{r['l2_rel']:<12.3e}{weights[i]:<10.3f}{mark}")

log(f"\nBoltzmann-weighted ensemble κ:")
log(f"  κ_ensemble = {k_mean:.4e}   error = {k_err:.2f}%")
log(f"  weighted std = {k_std:.4e}")
log(f"  95% bootstrap CI: [{ci_low:.4e}, {ci_high:.4e}]")
log(f"\nBest single by loss: {best['id']}  κ={best['kappa']:.4e}  err={best['err_pct']:.2f}%")
log(f"κ_true (reference) = {KAPPA_TRUE:.4e}")

# Save summary
with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump({
        'kappa_true': KAPPA_TRUE,
        'ensemble_size': len(results),
        'kappas': kappas.tolist(),
        'losses': losses.tolist(),
        'weights': weights.tolist(),
        'errors_pct': errs.tolist(),
        'l2_rel': l2s.tolist(),
        'ensemble_kappa': k_mean,
        'ensemble_err_pct': k_err,
        'ensemble_std': k_std,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'best_id': best['id'],
        'best_kappa': best['kappa'],
        'best_err_pct': best['err_pct'],
        'config': {
            'N_TIME': N_TIME, 'NOISE_LEVEL': NOISE_LEVEL,
            'N_SENSORS': len(SENSOR_XY),
            'KAPPA_MIN': KAPPA_MIN, 'KAPPA_MAX': KAPPA_MAX,
            'ADAM_EPOCHS': ADAM_EPOCHS, 'LBFGS_STEPS': LBFGS_STEPS,
            'LOSS_NAMES': LOSS_NAMES,
            'LOSS_WEIGHTS': LOSS_WEIGHTS,
            'ENSEMBLE_CONFIGS': ENSEMBLE_CONFIGS,
            'sensors': SENSOR_XY,
        }
    }, f, indent=2)

# ── Plots ─────────────────────────────────────────────────────────────────────
log("\nGenerating plots ...")
plot_loss_history(results)
plot_kappa_history(results)
plot_ensemble_estimate(results, weights, k_mean, ci_low, ci_high, boot_samples)
plot_weights(results, weights)
plot_arch_comparison(results)

# Best model plots
best_model = PINN(hidden=best['hidden'], n_layers=best['layers'],
                  kappa_init=best['kappa_init'],
                  residual=best.get('residual', False)).to(device)
best_model.load_state_dict(best['model_state'])
best_model.eval()
plot_amplitude(best_model)
plot_field_snapshots(best_model)
plot_pde_residual(best_model)
plot_time_error(best_model)

log(f"All plots saved to: {PLOTS_DIR}")
log(f"All artifacts saved to: {RESULTS_DIR}")
log("\nDone.")
LOG_FILE.close()
