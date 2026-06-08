"""
PINN v3 — production version for diploma report.

Inverse problem: identify kappa in 2D GN-III homogeneous heat equation
                   u_tt - kappa*(u_xxt + u_yyt) - (u_xx + u_yy) = 0

Strategy (everything stays within inverse-problem physics):
  - Sensor data: 5 sensors x 500 time points (realistic ADC rate)
  - Noise level: 1e-5 (configurable)
  - kappa bounded to physical range [1e-7, 1e-2] via sigmoid (knowledge of
    material class — metal — but NOT of specific value 1.2e-4)
  - Multi-restart x3 with seeds 1,2,3; selection by lowest final LOSS
  - Adam(20000) -> L-BFGS(100); no kappa freezing (was harmful in v3_test)

All outputs saved to  ./results_v3/

Run:  .venv/Scripts/python.exe example/pinn_homogeneous_v3.py
"""

import math, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import autograd, optim
import matplotlib
matplotlib.use('Agg')   # no display, save only
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


# ── Output folder & logger ────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results_v3"
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
KAPPA_TRUE = 1.2e-4          # used only for data generation and final validation
T_MAX = 500.0
LAM11 = 2 * PI ** 2
OMEGA0 = math.sqrt(LAM11)

KAPPA_MIN, KAPPA_MAX = 1e-7, 1e-2   # physical bounds (material class = metal)
KAPPA_INIT = 1e-3                   # neutral starting point

SENSOR_XY = [(0.25, 0.25), (0.25, 0.75), (0.50, 0.50), (0.75, 0.25), (0.75, 0.75)]
N_TIME = 500
NOISE_LEVEL = 1e-3   # 0.1% — typical K-type thermocouple / PT100 sensor

W_PDE, W_IC1, W_IC2, W_BC, W_DATA = 1.0, 200.0, 200.0, 200.0, 2000.0
ADAM_EPOCHS = 20000
LBFGS_STEPS = 100
SEEDS = [1, 2, 3]


# ── Analytical solution (data + validation only) ──────────────────────────────
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


# ── PINN ──────────────────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, hidden=128, n_layers=6, t_max=T_MAX):
        super().__init__()
        self.t_max = t_max
        self.omega0 = OMEGA0
        in_dim = 5
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # kappa via sigmoid in [KMIN, KMAX]; init at KAPPA_INIT
        ratio = (KAPPA_INIT - KAPPA_MIN) / (KAPPA_MAX - KAPPA_MIN)
        theta_init = math.log(ratio / (1.0 - ratio))
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))

    def forward(self, x, y, t):
        s = torch.sin(self.omega0 * t)
        c = torch.cos(self.omega0 * t)
        return self.net(torch.cat([x, y, s, c, t / self.t_max], dim=1))

    def get_kappa(self):
        return KAPPA_MIN + (KAPPA_MAX - KAPPA_MIN) * torch.sigmoid(self.theta)


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
    def rt(): return torch.rand(k, 1, requires_grad=True, device=device) * T_MAX
    x0 = torch.zeros(k, 1, requires_grad=True, device=device)
    x1 = torch.ones(k, 1, requires_grad=True, device=device)
    y0 = torch.zeros(k, 1, requires_grad=True, device=device)
    y1 = torch.ones(k, 1, requires_grad=True, device=device)
    xL, yL, tL = x0, torch.rand(k, 1, requires_grad=True, device=device), rt()
    xR, yR, tR = x1, torch.rand(k, 1, requires_grad=True, device=device), rt()
    xB, yB, tB = torch.rand(k, 1, requires_grad=True, device=device), y0, rt()
    xT, yT, tT = torch.rand(k, 1, requires_grad=True, device=device), y1, rt()
    return (torch.cat([xL, xR, xB, xT]),
            torch.cat([yL, yR, yB, yT]),
            torch.cat([tL, tR, tB, tT]))


def _g(u, v):
    return autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                         create_graph=True, retain_graph=True)[0]


def pde_res(model, x, y, t):
    u = model(x, y, t)
    ux, uy, ut = _g(u, x), _g(u, y), _g(u, t)
    uxx, uyy, utt = _g(ux, x), _g(uy, y), _g(ut, t)
    uxxt, uyyt = _g(uxx, t), _g(uyy, t)
    return utt - model.get_kappa() * (uxxt + uyyt) - (uxx + uyy)


mse = nn.MSELoss()


def loss_components(model, xd, yd, td, dn,
                    fp=None, fi=None, fb=None):
    if fp is None:
        xp, yp, tp = spde(10000); xi, yi, ti = sic(3000); xb, yb, tb = sbc(3000)
    else:
        xp, yp, tp = (v.detach().requires_grad_(True) for v in fp)
        xi, yi, ti = (v.detach().requires_grad_(True) for v in fi)
        xb, yb, tb = (v.detach().requires_grad_(True) for v in fb)
    l_pde = pde_res(model, xp, yp, tp).pow(2).mean()
    u_ic = model(xi, yi, ti)
    l_ic1 = mse(u_ic, torch.sin(PI * xi) * torch.sin(PI * yi))
    l_ic2 = _g(u_ic, ti).pow(2).mean()
    l_bc = model(xb, yb, tb).pow(2).mean()
    l_data = mse(model(xd, yd, td), dn)
    return l_pde, l_ic1, l_ic2, l_bc, l_data


# ── Sensor data (one experiment for all restarts) ─────────────────────────────
def make_sensor_data(seed=0):
    rng = np.random.default_rng(seed)
    t_s = np.linspace(0.0, T_MAX, N_TIME)
    xs, ys, ts, ds = [], [], [], []
    for sx, sy in SENSOR_XY:
        vals = analytical(np.full(N_TIME, sx), np.full(N_TIME, sy), t_s)
        xs.append(np.full(N_TIME, sx))
        ys.append(np.full(N_TIME, sy))
        ts.append(t_s)
        ds.append(vals)
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


# ── Single training run ───────────────────────────────────────────────────────
def train_one(seed, xd, yd, td, dn, label, run_dir):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = PINN().to(device)
    log(f"\n[{label}] seed={seed}  kappa_init={model.get_kappa().item():.4e}")

    loss_hist, kappa_hist = [], []
    comp_hist = {'pde': [], 'ic1': [], 'ic2': [], 'bc': [], 'data': []}
    best_loss = float('inf')
    best_sd = None
    t0 = time.time()

    # Phase 1: Adam (joint, no warm-up)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1000, min_lr=1e-6)

    for ep in range(ADAM_EPOCHS):
        opt.zero_grad()
        l_pde, l_ic1, l_ic2, l_bc, l_data = loss_components(model, xd, yd, td, dn)
        L = (W_PDE*l_pde + W_IC1*l_ic1 + W_IC2*l_ic2 + W_BC*l_bc + W_DATA*l_data)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(L)

        loss_hist.append(L.item())
        kappa_hist.append(model.get_kappa().item())
        comp_hist['pde'].append(l_pde.item())
        comp_hist['ic1'].append(l_ic1.item())
        comp_hist['ic2'].append(l_ic2.item())
        comp_hist['bc'].append(l_bc.item())
        comp_hist['data'].append(l_data.item())

        if L.item() < best_loss:
            best_loss = L.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 1000 == 0:
            lr_now = opt.param_groups[0]['lr']
            log(f"  [Adam]   ep={ep:5d}  loss={L.item():.4e}  "
                f"pde={l_pde.item():.3e}  data={l_data.item():.3e}  "
                f"kappa={model.get_kappa().item():.4e}  lr={lr_now:.2e}")

    model.load_state_dict(best_sd)

    # Phase 2: L-BFGS (fixed collocation)
    fp = spde(10000); fi = sic(3000); fb = sbc(3000)
    lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=50,
                        history_size=100, line_search_fn='strong_wolfe')

    def closure():
        lbfgs.zero_grad()
        lp, l1, l2, lb, ld = loss_components(model, xd, yd, td, dn, fp, fi, fb)
        L = W_PDE*lp + W_IC1*l1 + W_IC2*l2 + W_BC*lb + W_DATA*ld
        L.backward()
        return L

    for step in range(LBFGS_STEPS):
        L = lbfgs.step(closure)
        loss_hist.append(L.item())
        kappa_hist.append(model.get_kappa().item())
        if L is not None and L.item() < best_loss:
            best_loss = L.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if step % 20 == 0:
            log(f"  [LBFGS]  step={step:3d}  loss={L.item():.4e}  "
                f"kappa={model.get_kappa().item():.4e}")

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

    # Persist artifacts
    torch.save(best_sd, run_dir / "state_dict.pth")
    np.save(run_dir / "loss_history.npy", np.array(loss_hist))
    np.save(run_dir / "kappa_history.npy", np.array(kappa_hist))
    for k, v in comp_hist.items():
        np.save(run_dir / f"loss_{k}.npy", np.array(v))
    metrics = {'seed': seed, 'loss': best_loss, 'kappa': kf, 'err_pct': err,
               'l2_rel': l2, 'linf': linf, 'time_min': elapsed / 60}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log(f"[{label}] DONE  loss={best_loss:.4e}  kappa={kf:.4e}  err={err:.2f}%  "
        f"L2={l2:.3e}  Linf={linf:.3e}  t={elapsed/60:.1f}min")

    return {**metrics, 'model_state': best_sd,
            'loss_hist': loss_hist, 'kappa_hist': kappa_hist, 'comp_hist': comp_hist}


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_loss_history(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results, 1):
        ax.semilogy(r['loss_hist'], lw=0.7, label=f"R{i}  (seed={r['seed']})")
    ax.axvline(ADAM_EPOCHS, color='k', ls='--', lw=0.8, alpha=0.6, label='Adam → L-BFGS')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total loss')
    ax.set_title('Loss convergence (3 restarts)')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_history.png", dpi=150); plt.close()


def plot_kappa_history(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results, 1):
        ax.semilogy(r['kappa_hist'], lw=0.8, label=f"R{i}  (seed={r['seed']})")
    ax.axhline(KAPPA_TRUE, color='k', ls=':', lw=1.2, label=f'κ_true = {KAPPA_TRUE:.2e}')
    ax.axhline(KAPPA_MIN, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(KAPPA_MAX, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('κ (log scale)')
    ax.set_title('κ trajectory across restarts')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kappa_history.png", dpi=150); plt.close()


def plot_kappa_distribution(results, ci_low, ci_high, mean):
    kappas = [r['kappa'] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(1, len(kappas) + 1)
    ax.scatter(xs, kappas, s=80, c='b', zorder=3, label='restart')
    for x, k in zip(xs, kappas):
        ax.text(x, k * 1.05, f'{k:.3e}', ha='center', fontsize=9)
    ax.axhline(KAPPA_TRUE, color='k', ls=':', lw=1.2, label=f'κ_true = {KAPPA_TRUE:.3e}')
    ax.axhline(mean, color='r', ls='-', lw=1.2, label=f'mean = {mean:.3e}')
    ax.fill_between([0.5, len(kappas) + 0.5], ci_low, ci_high, color='r', alpha=0.15,
                    label=f'95% CI [{ci_low:.3e}, {ci_high:.3e}]')
    ax.set_xticks(xs)
    ax.set_xticklabels([f"R{i}" for i in xs])
    ax.set_ylabel('κ')
    ax.set_title('Identified κ — point estimates + 95% confidence interval')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kappa_distribution.png", dpi=150); plt.close()


def plot_amplitude(best_model):
    t_plot = np.linspace(0, T_MAX, 1000)
    A_exact = analytical.amplitude(t_plot)
    x05 = torch.full((1000, 1), 0.5, device=device)
    y05 = torch.full((1000, 1), 0.5, device=device)
    tp = torch.tensor(t_plot.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        u = best_model(x05, y05, tp).cpu().numpy().flatten()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t_plot, A_exact, 'k-', lw=1.4, label='Analytical')
    axes[0].plot(t_plot, u, 'r--', lw=1.0, label='PINN')
    axes[0].set_ylabel('A(t) at (0.5, 0.5)')
    axes[0].set_title('Temporal amplitude profile (best restart)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(t_plot, np.abs(u - A_exact) + 1e-12, 'b-', lw=0.7)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('|A_PINN - A_exact|')
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
        dlim = max(abs(diff.min()), abs(diff.max()))

        c0 = axes[0, i].contourf(X.cpu(), Y.cpu(), up, 30, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'PINN  t={t_val:g}')
        plt.colorbar(c0, ax=axes[0, i], fraction=0.046, pad=0.04)
        c1 = axes[1, i].contourf(X.cpu(), Y.cpu(), ua, 30, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Exact  t={t_val:g}')
        plt.colorbar(c1, ax=axes[1, i], fraction=0.046, pad=0.04)
        c2 = axes[2, i].contourf(X.cpu(), Y.cpu(), diff, 30, cmap='RdBu_r', vmin=-dlim, vmax=dlim)
        axes[2, i].set_title(f'Diff  max={dlim:.2e}')
        plt.colorbar(c2, ax=axes[2, i], fraction=0.046, pad=0.04)
        for r in range(3):
            axes[r, i].set_xlabel('x'); axes[r, i].set_ylabel('y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "field_snapshots.png", dpi=130); plt.close()


def plot_pde_residual(best_model):
    n_sp = 60
    x_sp = torch.linspace(0, 1, n_sp, device=device)
    y_sp = torch.linspace(0, 1, n_sp, device=device)
    X, Y = torch.meshgrid(x_sp, y_sp, indexing='ij')
    Xf, Yf = X.reshape(-1, 1).requires_grad_(True), Y.reshape(-1, 1).requires_grad_(True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, t_val in enumerate([10.0, 100.0, 400.0]):
        Tf = torch.full_like(Xf, t_val).requires_grad_(True)
        res = pde_res(best_model, Xf, Yf, Tf).detach().reshape(n_sp, n_sp).cpu().numpy()
        ax = axes[i]
        cb = ax.contourf(X.cpu(), Y.cpu(), np.abs(res), 30, cmap='hot')
        ax.set_title(f'|PDE residual|  t={t_val:g}')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(cb, ax=ax, fraction=0.046, pad=0.04)
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
    ax.set_title('Field error vs time (best restart)')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "time_error.png", dpi=150); plt.close()


def plot_loss_breakdown(best_result):
    comp = best_result['comp_hist']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(comp['pde'],  lw=0.6, label='L_pde')
    ax.semilogy(comp['ic1'],  lw=0.6, label='L_ic1 (value)')
    ax.semilogy(comp['ic2'],  lw=0.6, label='L_ic2 (velocity)')
    ax.semilogy(comp['bc'],   lw=0.6, label='L_bc')
    ax.semilogy(comp['data'], lw=0.6, label='L_data')
    ax.set_xlabel('Adam epoch')
    ax.set_ylabel('Loss component (unweighted)')
    ax.set_title('Loss components — best restart')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_breakdown.png", dpi=150); plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
log("\n" + "="*72)
log("PINN v3  —  MULTI-RESTART (no warm-up, sigmoid-bounded κ)")
log(f"  N_TIME={N_TIME}, sensors={len(SENSOR_XY)}, noise={NOISE_LEVEL:.0e}")
log(f"  κ in [{KAPPA_MIN:.0e}, {KAPPA_MAX:.0e}], κ_init={KAPPA_INIT:.0e}")
log(f"  Adam={ADAM_EPOCHS}, LBFGS={LBFGS_STEPS}, seeds={SEEDS}")
log("="*72)

xd, yd, td, dn = make_sensor_data(seed=0)
log(f"Sensor data: {dn.shape[0]} points, "
    f"|signal| range [{dn.abs().min().item():.3e}, {dn.abs().max().item():.3e}]")

results = []
for i, seed in enumerate(SEEDS, 1):
    run_dir = RUNS_DIR / f"R{i}_seed{seed}"
    run_dir.mkdir(exist_ok=True)
    res = train_one(seed, xd, yd, td, dn, label=f"R{i}", run_dir=run_dir)
    results.append(res)

# ── Statistics and confidence interval ────────────────────────────────────────
kappas = np.array([r['kappa'] for r in results])
errs = np.array([r['err_pct'] for r in results])
losses = np.array([r['loss'] for r in results])
l2s = np.array([r['l2_rel'] for r in results])

mean_k = float(np.mean(kappas))
std_k = float(np.std(kappas, ddof=1)) if len(kappas) > 1 else 0.0
n = len(kappas)
ci_half = float(student_t.ppf(0.975, df=max(n-1, 1)) * std_k / math.sqrt(n)) if n > 1 else 0.0
ci_low, ci_high = mean_k - ci_half, mean_k + ci_half

best = min(results, key=lambda r: r['loss'])
best_idx = results.index(best)

# ── Summary log ───────────────────────────────────────────────────────────────
log("\n" + "="*72)
log("SUMMARY")
log("="*72)
log(f"  {'Run':<5}{'seed':<6}{'loss':<14}{'kappa':<14}{'err%':<10}{'L2':<12}{'time min':<10}")
for i, r in enumerate(results, 1):
    mark = "  *BEST" if r is best else ""
    log(f"  R{i:<4}{r['seed']:<6}{r['loss']:<14.4e}{r['kappa']:<14.4e}"
        f"{r['err_pct']:<10.2f}{r['l2_rel']:<12.3e}{r['time_min']:<10.1f}{mark}")

log(f"\nκ statistics over {n} restarts:")
log(f"  mean = {mean_k:.4e}")
log(f"  std  = {std_k:.4e}")
log(f"  95% CI: [{ci_low:.4e}, {ci_high:.4e}]   (t-distribution, df={n-1})")
log(f"  median = {float(np.median(kappas)):.4e}")
log(f"  range  = [{kappas.min():.4e}, {kappas.max():.4e}]")
log(f"\nBest by loss: R{best_idx+1}  κ={best['kappa']:.4e}  err={best['err_pct']:.2f}%")
log(f"κ_true (reference) = {KAPPA_TRUE:.4e}")

# Save summary JSON
with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump({
        'kappa_true': KAPPA_TRUE,
        'n_restarts': n,
        'kappas': kappas.tolist(),
        'errors_pct': errs.tolist(),
        'losses': losses.tolist(),
        'l2_rel': l2s.tolist(),
        'mean_kappa': mean_k,
        'std_kappa': std_k,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'best_run': best_idx + 1,
        'best_kappa': best['kappa'],
        'best_err_pct': best['err_pct'],
        'config': {
            'N_TIME': N_TIME, 'NOISE_LEVEL': NOISE_LEVEL,
            'KAPPA_MIN': KAPPA_MIN, 'KAPPA_MAX': KAPPA_MAX,
            'ADAM_EPOCHS': ADAM_EPOCHS, 'LBFGS_STEPS': LBFGS_STEPS,
            'SEEDS': SEEDS, 'sensors': SENSOR_XY,
            'weights': {'pde': W_PDE, 'ic1': W_IC1, 'ic2': W_IC2,
                        'bc': W_BC, 'data': W_DATA},
        }
    }, f, indent=2)

# ── Plots ─────────────────────────────────────────────────────────────────────
log("\nGenerating plots ...")
plot_loss_history(results)
plot_kappa_history(results)
plot_kappa_distribution(results, ci_low, ci_high, mean_k)
plot_loss_breakdown(best)

# For per-field plots, reload best model
best_model = PINN().to(device)
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
