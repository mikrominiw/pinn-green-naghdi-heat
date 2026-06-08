"""
Diploma parametric study.

Runs both homogeneous (q=0) and inhomogeneous (q=2xt) cases across:
  - noise levels: 1e-5, 1e-4, 1e-3, 1e-2
  - seeds: 1, 2, 3  (reproducibility check)

For each run saves metrics and generates all diploma-quality figures:
  1. kappa convergence curve (kappa vs epoch)
  2. loss convergence (log scale)
  3. field snapshots PINN vs Exact at t = [1, 10, 50, 100, 250, 500]
  4. temporal profile u(0.5, 0.5, t) PINN vs Exact
  5. summary table: noise x seed -> kappa_found, err%, L2, Linf, time

Results saved to: ./diploma_results/
"""

import math
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import autograd, optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Output ────────────────────────────────────────────────────────────────────
OUT = Path(__file__).parent.parent / "diploma_results"
OUT.mkdir(exist_ok=True)
LOG_FILE = open(OUT / "log.txt", "w", encoding="utf-8")


def log(*a, **kw):
    print(*a, **kw)
    print(*a, **kw, file=LOG_FILE)
    LOG_FILE.flush()


# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Constants ─────────────────────────────────────────────────────────────────
PI = math.pi
KAPPA_TRUE = 1.2e-4
T_MAX = 500.0
LAM11 = 2 * PI ** 2
OMEGA1 = math.sqrt(LAM11)        # pi*sqrt(2) — mode (1,1)
OMEGA2 = math.sqrt(5) * PI       # pi*sqrt(5) — mode (2,1), inhomogeneous only

ADAM_EPOCHS = 20000
LBFGS_STEPS = 300

NOISE_LEVELS = [1e-5, 1e-4, 1e-3, 1e-2]
SEEDS = [1, 2, 3]

SENSOR_XY = [(0.25, 0.25), (0.25, 0.75), (0.50, 0.50), (0.75, 0.25), (0.75, 0.75)]
N_TIME = 200

W_PDE  = 1.0
W_IC1  = 200.0
W_IC2  = 200.0
W_BC   = 200.0
W_DATA = 2000.0


# ── Analytical solutions ──────────────────────────────────────────────────────
class AnalyticalHomogeneous:
    """Single-mode exact solution for q=0."""
    def __init__(self, kappa=KAPPA_TRUE):
        lam = LAM11
        self.alpha = kappa * lam / 2
        self.beta = math.sqrt(4 * lam - (kappa * lam) ** 2) / 2

    def amplitude(self, t):
        return np.exp(-self.alpha * t) * (
            np.cos(self.beta * t) + (self.alpha / self.beta) * np.sin(self.beta * t))

    def __call__(self, x, y, t):
        if torch.is_tensor(x):
            x, y, t = (v.detach().cpu().numpy().flatten() for v in (x, y, t))
        return self.amplitude(np.asarray(t)) * np.sin(PI * x) * np.sin(PI * y)


class AnalyticalInhomogeneous:
    """Multi-mode exact solution for q=2xt."""
    def __init__(self, kappa=KAPPA_TRUE, n_max=30, m_max=30):
        self.kappa = kappa
        self.n_max = n_max
        self.m_max = m_max

    def __call__(self, x, y, t):
        if torch.is_tensor(x):
            x, y, t = (v.detach().cpu().numpy().flatten() for v in (x, y, t))
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        t = np.asarray(t, dtype=float)
        result = np.zeros_like(x)
        for n in range(1, self.n_max + 1):
            for m in range(1, self.m_max + 1):
                lam = PI ** 2 * (n ** 2 + m ** 2)
                A_nm = 1.0 if (n == 1 and m == 1) else 0.0
                int_x = (-1) ** (n + 1) / (n * PI)
                int_y = (1 - (-1) ** m) / (m * PI)
                f_nm = 8 * int_x * int_y
                if abs(A_nm) < 1e-12 and abs(f_nm) < 1e-12:
                    continue
                disc = (self.kappa * lam) ** 2 - 4 * lam
                if disc < 0:
                    real = -self.kappa * lam / 2
                    imag = math.sqrt(-disc) / 2
                    D1 = A_nm - f_nm / lam
                    D2 = -real * D1 / imag
                    T_t = (np.exp(real * t) * (D1 * np.cos(imag * t) + D2 * np.sin(imag * t))
                           + f_nm / lam)
                else:
                    r1 = (-self.kappa * lam + math.sqrt(disc)) / 2
                    r2 = (-self.kappa * lam - math.sqrt(disc)) / 2
                    diff = r1 - r2
                    C1 = -r2 * (A_nm - f_nm / lam) / diff if abs(diff) > 1e-12 else (A_nm - f_nm / lam) / 2
                    C2 = r1 * (A_nm - f_nm / lam) / diff if abs(diff) > 1e-12 else (A_nm - f_nm / lam) / 2
                    T_t = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t) + f_nm / lam
                result += T_t * np.sin(n * PI * x) * np.sin(m * PI * y)
        return result


# ── PINN architectures ────────────────────────────────────────────────────────
class PINNHomogeneous(nn.Module):
    """5-feature input: [x, y, sin(w1*t), cos(w1*t), t/T]."""
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(5, 128), nn.Tanh()]
        for _ in range(5):
            layers += [nn.Linear(128, 128), nn.Tanh()]
        layers.append(nn.Linear(128, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(1e-3), dtype=torch.float32))

    def forward(self, x, y, t):
        inp = torch.cat([x, y,
                         torch.sin(OMEGA1 * t), torch.cos(OMEGA1 * t),
                         t / T_MAX], dim=1)
        return self.net(inp)

    def get_kappa(self):
        return torch.exp(self.log_kappa)


class PINNInhomogeneous(nn.Module):
    """7-feature input: [x, y, sin(w1*t), cos(w1*t), sin(w2*t), cos(w2*t), t/T]."""
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(7, 128), nn.Tanh()]
        for _ in range(5):
            layers += [nn.Linear(128, 128), nn.Tanh()]
        layers.append(nn.Linear(128, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(1e-3), dtype=torch.float32))

    def forward(self, x, y, t):
        inp = torch.cat([x, y,
                         torch.sin(OMEGA1 * t), torch.cos(OMEGA1 * t),
                         torch.sin(OMEGA2 * t), torch.cos(OMEGA2 * t),
                         t / T_MAX], dim=1)
        return self.net(inp)

    def get_kappa(self):
        return torch.exp(self.log_kappa)


# ── Collocation samplers ──────────────────────────────────────────────────────
def spde(n):
    return (torch.rand(n, 1, requires_grad=True, device=device),
            torch.rand(n, 1, requires_grad=True, device=device),
            torch.rand(n, 1, requires_grad=True, device=device) * T_MAX)


def sic(n):
    return (torch.rand(n, 1, requires_grad=True, device=device),
            torch.rand(n, 1, requires_grad=True, device=device),
            torch.zeros(n, 1, requires_grad=True, device=device))


def sbc(n):
    k = n // 4
    rt = lambda: torch.rand(k, 1, requires_grad=True, device=device) * T_MAX
    x0 = torch.zeros(k, 1, requires_grad=True, device=device)
    x1 = torch.ones(k, 1, requires_grad=True, device=device)
    y0 = torch.zeros(k, 1, requires_grad=True, device=device)
    y1 = torch.ones(k, 1, requires_grad=True, device=device)
    return (
        torch.cat([x0, x1, torch.rand(k,1,requires_grad=True,device=device), torch.rand(k,1,requires_grad=True,device=device)]),
        torch.cat([torch.rand(k,1,requires_grad=True,device=device), torch.rand(k,1,requires_grad=True,device=device), y0, y1]),
        torch.cat([rt(), rt(), rt(), rt()])
    )


# ── Autograd helper ───────────────────────────────────────────────────────────
def _g(u, v):
    return autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                         create_graph=True, retain_graph=True)[0]


def pde_res(model, x, y, t, source_x=None):
    """GN-III residual. source_x=x for inhomogeneous (rhs=2x), None for homogeneous."""
    u = model(x, y, t)
    ux, uy, ut = _g(u, x), _g(u, y), _g(u, t)
    uxx, uyy, utt = _g(ux, x), _g(uy, y), _g(ut, t)
    uxxt, uyyt = _g(uxx, t), _g(uyy, t)
    r = utt - model.get_kappa() * (uxxt + uyyt) - (uxx + uyy)
    if source_x is not None:
        r = r - 2.0 * source_x
    return r


mse = nn.MSELoss()


def compute_loss(model, x_data, y_data, t_data, data_noisy, inhomogeneous,
                 fp=None, fi=None, fb=None):
    xp, yp, tp = spde(10000) if fp is None else (v.detach().requires_grad_(True) for v in fp)
    xi, yi, ti = sic(3000)   if fi is None else (v.detach().requires_grad_(True) for v in fi)
    xb, yb, tb = sbc(3000)   if fb is None else (v.detach().requires_grad_(True) for v in fb)

    src = xp if inhomogeneous else None
    l_pde = pde_res(model, xp, yp, tp, source_x=src).pow(2).mean()

    u_ic = model(xi, yi, ti)
    l_ic1 = mse(u_ic, torch.sin(PI * xi) * torch.sin(PI * yi))
    l_ic2 = _g(u_ic, ti).pow(2).mean()
    l_bc = model(xb, yb, tb).pow(2).mean()
    l_data = mse(model(x_data, y_data, t_data), data_noisy)

    total = W_PDE*l_pde + W_IC1*l_ic1 + W_IC2*l_ic2 + W_BC*l_bc + W_DATA*l_data
    return total, l_pde, l_ic1, l_ic2, l_bc, l_data


# ── Sensor data builder ───────────────────────────────────────────────────────
def make_sensor_data(analytical, noise_level, seed):
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
    ds_noisy = ds + rng.standard_normal(ds.shape) * noise_level * np.abs(ds)
    return (torch.tensor(xs, dtype=torch.float32, device=device),
            torch.tensor(ys, dtype=torch.float32, device=device),
            torch.tensor(ts, dtype=torch.float32, device=device),
            torch.tensor(ds_noisy, dtype=torch.float32, device=device))


# ── Single run ────────────────────────────────────────────────────────────────
def train_one(case, noise_level, seed, run_dir):
    """
    case: 'homo' or 'inhomo'
    Returns dict with metrics and histories.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    inhomogeneous = (case == 'inhomo')
    analytical = AnalyticalInhomogeneous() if inhomogeneous else AnalyticalHomogeneous()
    model_cls = PINNInhomogeneous if inhomogeneous else PINNHomogeneous
    model = model_cls().to(device)

    x_data, y_data, t_data, data_noisy = make_sensor_data(analytical, noise_level, seed)

    label = f"{case} | noise={noise_level:.0e} | seed={seed}"
    log(f"\n{'─'*60}")
    log(f"  {label}")
    log(f"  params={sum(p.numel() for p in model.parameters()):,}  "
        f"kappa_init={model.get_kappa().item():.2e}")
    log(f"{'─'*60}")

    loss_hist, kappa_hist = [], []
    best_loss = float('inf')
    best_sd = None
    t0 = time.time()

    # Adam
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1000, min_lr=1e-6)

    for ep in range(ADAM_EPOCHS):
        opt.zero_grad()
        L, l_pde, l_ic1, l_ic2, l_bc, l_data = compute_loss(
            model, x_data, y_data, t_data, data_noisy, inhomogeneous)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(L)
        loss_hist.append(L.item())
        kappa_hist.append(model.get_kappa().item())
        if L.item() < best_loss:
            best_loss = L.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 2000 == 0:
            log(f"  Adam {ep:5d}  loss={L.item():.4e}  "
                f"pde={l_pde.item():.2e}  data={l_data.item():.2e}  "
                f"kappa={model.get_kappa().item():.4e}  lr={opt.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_sd)

    # L-BFGS
    fp = spde(10000); fi = sic(3000); fb = sbc(3000)
    lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=50,
                        history_size=100, line_search_fn='strong_wolfe')

    def closure():
        lbfgs.zero_grad()
        L, *_ = compute_loss(model, x_data, y_data, t_data, data_noisy,
                              inhomogeneous, fp=fp, fi=fi, fb=fb)
        L.backward()
        return L

    for step in range(LBFGS_STEPS):
        try:
            L = lbfgs.step(closure)
        except Exception as e:
            log(f"  L-BFGS step {step}: {type(e).__name__} — stopping"); break
        if L is None or not math.isfinite(L.item()):
            log(f"  L-BFGS step {step}: non-finite — stopping"); break
        loss_hist.append(L.item())
        kappa_hist.append(model.get_kappa().item())
        if L.item() < best_loss:
            best_loss = L.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if step % 100 == 0:
            log(f"  LBFGS {step:3d}  loss={L.item():.4e}  kappa={model.get_kappa().item():.4e}")

    model.load_state_dict(best_sd)
    elapsed = time.time() - t0

    kf = model.get_kappa().item()
    err = abs(kf - KAPPA_TRUE) / KAPPA_TRUE * 100

    # Field metrics
    model.eval()
    with torch.no_grad():
        xt = torch.rand(5000, 1, device=device)
        yt = torch.rand(5000, 1, device=device)
        tt = torch.rand(5000, 1, device=device) * T_MAX
        up = model(xt, yt, tt)
    ue = torch.tensor(analytical(xt.cpu(), yt.cpu(), tt.cpu()).reshape(-1, 1), dtype=torch.float32)
    l2 = (torch.norm(up.cpu() - ue) / torch.norm(ue)).item()
    linf = torch.max((up.cpu() - ue).abs()).item()

    log(f"  RESULT  kappa={kf:.4e}  err={err:.2f}%  L2={l2:.3e}  Linf={linf:.3e}  "
        f"t={elapsed/60:.1f}min")

    # Save
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_sd, run_dir / "state_dict.pth")
    np.save(run_dir / "loss_history.npy", np.array(loss_hist))
    np.save(run_dir / "kappa_history.npy", np.array(kappa_hist))
    metrics = {'case': case, 'noise': noise_level, 'seed': seed,
               'kappa_found': kf, 'err_pct': err,
               'l2_rel': l2, 'linf': linf, 'time_min': elapsed / 60}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {**metrics, 'model': model, 'analytical': analytical,
            'loss_hist': loss_hist, 'kappa_hist': kappa_hist,
            'inhomogeneous': inhomogeneous}


# ── Per-run plots ─────────────────────────────────────────────────────────────
def plot_run(r, run_dir):
    model = r['model']
    analytical = r['analytical']

    # 1. Loss + kappa convergence
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.semilogy(r['loss_hist'], 'b-', lw=0.7, label='Total loss')
    ax1.axvline(ADAM_EPOCHS, color='gray', ls='--', lw=0.8, label='Adam→L-BFGS')
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('Loss (log)', color='b')
    ax1.tick_params(axis='y', colors='b')
    ax1.grid(True, which='both', alpha=0.3)
    ax2 = ax1.twinx()
    ax2.semilogy(r['kappa_hist'], 'r-', lw=0.9, alpha=0.8, label='kappa')
    ax2.axhline(KAPPA_TRUE, color='darkred', ls=':', lw=1.2, label=f'kappa_true={KAPPA_TRUE:.1e}')
    ax2.set_ylabel('kappa (log)', color='r')
    ax2.tick_params(axis='y', colors='r')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
    ax1.set_title(f"{r['case']} | noise={r['noise']:.0e} | seed={r['seed']} | "
                  f"kappa={r['kappa_found']:.3e} (err={r['err_pct']:.1f}%)")
    plt.tight_layout()
    plt.savefig(run_dir / "convergence.png", dpi=150); plt.close()

    # 2. Temporal profile u(0.5, 0.5, t)
    t_plot = np.linspace(0, T_MAX, 1000)
    u_ex = analytical(np.full(1000, 0.5), np.full(1000, 0.5), t_plot)
    x05 = torch.full((1000, 1), 0.5, device=device)
    y05 = torch.full((1000, 1), 0.5, device=device)
    tp = torch.tensor(t_plot.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        u_pinn = model(x05, y05, tp).cpu().numpy().flatten()

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t_plot, u_ex, 'b-', lw=1.4, label='Analytical')
    axes[0].plot(t_plot, u_pinn, 'r--', lw=1.0, label=f'PINN (kappa={r["kappa_found"]:.3e})')
    axes[0].set_ylabel('u(0.5, 0.5, t)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Temporal profile  |  err={r['err_pct']:.2f}%  L2={r['l2_rel']:.3e}")
    axes[1].semilogy(t_plot, np.abs(u_pinn - u_ex) + 1e-15, 'g-', lw=0.8)
    axes[1].set_xlabel('t'); axes[1].set_ylabel('|PINN - Analytical|')
    axes[1].grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "temporal_profile.png", dpi=150); plt.close()

    # 3. Field snapshots
    t_snaps = [1.0, 10.0, 50.0, 100.0, 250.0, 500.0]
    n_sp = 50
    xs = torch.linspace(0, 1, n_sp, device=device)
    ys = torch.linspace(0, 1, n_sp, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing='ij')
    Xf, Yf = X.reshape(-1, 1), Y.reshape(-1, 1)

    fig, axes = plt.subplots(3, len(t_snaps), figsize=(20, 9))
    for i, tv in enumerate(t_snaps):
        Tf = torch.full_like(Xf, tv)
        with torch.no_grad():
            up = model(Xf, Yf, Tf).reshape(n_sp, n_sp).cpu().numpy()
        ua = analytical(Xf.cpu(), Yf.cpu(), Tf.cpu()).reshape(n_sp, n_sp)
        diff = up - ua
        vlim = max(abs(up).max(), abs(ua).max(), 1e-12)
        dlim = max(abs(diff).max(), 1e-12)

        for row, (data, title, cmap, vmin, vmax) in enumerate([
            (up,   f'PINN  t={tv:g}',    'RdBu_r', -vlim,  vlim),
            (ua,   f'Exact t={tv:g}',    'RdBu_r', -vlim,  vlim),
            (diff, f'Diff  max={dlim:.1e}', 'seismic', -dlim, dlim),
        ]):
            c = axes[row, i].contourf(X.cpu(), Y.cpu(), data, 30,
                                      cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, i].set_title(title, fontsize=9)
            plt.colorbar(c, ax=axes[row, i], fraction=0.046, pad=0.04)
            axes[row, i].set_xlabel('x'); axes[row, i].set_ylabel('y')
    plt.suptitle(f"{r['case']} q={'2xt' if r['inhomogeneous'] else '0'}  |  "
                 f"kappa={r['kappa_found']:.3e} (true {KAPPA_TRUE:.1e}, err {r['err_pct']:.1f}%)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(run_dir / "field_snapshots.png", dpi=120); plt.close()


# ── Summary plots (across all runs) ──────────────────────────────────────────
def plot_noise_study(all_results, case, case_dir):
    """kappa_found vs noise level, averaged over seeds with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    noise_vals = sorted(set(r['noise'] for r in all_results if r['case'] == case))
    kappas_mean, kappas_std, errs_mean, errs_std = [], [], [], []

    for nl in noise_vals:
        subset = [r for r in all_results if r['case'] == case and r['noise'] == nl]
        kf = np.array([r['kappa_found'] for r in subset])
        er = np.array([r['err_pct'] for r in subset])
        kappas_mean.append(kf.mean()); kappas_std.append(kf.std())
        errs_mean.append(er.mean()); errs_std.append(er.std())

    x_pos = range(len(noise_vals))
    noise_labels = [f'{nl:.0e}' for nl in noise_vals]

    ax = axes[0]
    ax.errorbar(x_pos, kappas_mean, yerr=kappas_std, fmt='o-', capsize=5,
                color='steelblue', lw=1.5, ms=7, label='kappa_found (mean±std)')
    ax.axhline(KAPPA_TRUE, color='r', ls='--', lw=1.5, label=f'kappa_true={KAPPA_TRUE:.1e}')
    ax.set_yscale('log')
    ax.set_xticks(x_pos); ax.set_xticklabels(noise_labels)
    ax.set_xlabel('Noise level'); ax.set_ylabel('Identified kappa')
    ax.set_title(f'{case}: kappa identification vs noise')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    ax.errorbar(x_pos, errs_mean, yerr=errs_std, fmt='s-', capsize=5,
                color='darkorange', lw=1.5, ms=7, label='err% (mean±std)')
    ax.set_xticks(x_pos); ax.set_xticklabels(noise_labels)
    ax.set_xlabel('Noise level'); ax.set_ylabel('Relative error, %')
    ax.set_title(f'{case}: identification error vs noise')
    ax.set_yscale('log')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(case_dir / "noise_study.png", dpi=150); plt.close()


def plot_seed_study(all_results, case, case_dir):
    """kappa trajectory across seeds for each noise level."""
    noise_vals = sorted(set(r['noise'] for r in all_results if r['case'] == case))
    fig, axes = plt.subplots(1, len(noise_vals), figsize=(5 * len(noise_vals), 4), sharey=True)
    if len(noise_vals) == 1:
        axes = [axes]

    colors = ['steelblue', 'darkorange', 'green']
    for ax, nl in zip(axes, noise_vals):
        subset = [r for r in all_results if r['case'] == case and r['noise'] == nl]
        for r, col in zip(subset, colors):
            hist = r.get('kappa_hist')
            if not hist:
                continue
            ax.semilogy(hist, color=col, lw=0.8, alpha=0.9,
                        label=f"seed={r['seed']} kappa={r['kappa_found']:.2e}")
        ax.axhline(KAPPA_TRUE, color='k', ls=':', lw=1.5, label=f'kappa_true')
        ax.axvline(ADAM_EPOCHS, color='gray', ls='--', lw=0.7)
        ax.set_title(f'noise={nl:.0e}')
        ax.set_xlabel('Iteration')
        ax.legend(fontsize=7); ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('kappa (log)')
    plt.suptitle(f'{case}: kappa convergence across seeds', fontsize=11)
    plt.tight_layout()
    plt.savefig(case_dir / "seed_reproducibility.png", dpi=150); plt.close()


def plot_homo_vs_inhomo(all_results, out_dir):
    """Side-by-side error comparison: homogeneous vs inhomogeneous."""
    fig, ax = plt.subplots(figsize=(10, 5))
    noise_vals = sorted(set(r['noise'] for r in all_results))
    x = np.arange(len(noise_vals))
    width = 0.35

    for case, offset, color, label in [
        ('homo', -width/2, 'steelblue', 'Homogeneous (q=0)'),
        ('inhomo',  width/2, 'darkorange', 'Inhomogeneous (q=2xt)'),
    ]:
        means, stds = [], []
        for nl in noise_vals:
            subset = [r for r in all_results if r['case'] == case and r['noise'] == nl]
            errs = [r['err_pct'] for r in subset]
            means.append(np.mean(errs)); stds.append(np.std(errs))
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      color=color, alpha=0.8, label=label, edgecolor='k')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{nl:.0e}' for nl in noise_vals])
    ax.set_xlabel('Noise level')
    ax.set_ylabel('kappa relative error, %')
    ax.set_title('Homogeneous vs Inhomogeneous: kappa identification error')
    ax.set_yscale('log')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "homo_vs_inhomo.png", dpi=150); plt.close()


def save_summary_table(all_results, out_dir):
    """Print and save a LaTeX-friendly summary table."""
    header = f"{'Case':<8} {'Noise':<8} {'Seed':<5} {'kappa_found':<14} {'err%':<8} {'L2':<10} {'Linf':<10} {'time_min':<10}"
    log("\n" + "="*72)
    log("SUMMARY TABLE")
    log("="*72)
    log(header)
    log("-" * len(header))
    rows = []
    for r in sorted(all_results, key=lambda x: (x['case'], x['noise'], x['seed'])):
        row = (f"{r['case']:<8} {r['noise']:<8.0e} {r['seed']:<5} "
               f"{r['kappa_found']:<14.4e} {r['err_pct']:<8.2f} "
               f"{r['l2_rel']:<10.3e} {r['linf']:<10.3e} {r['time_min']:<10.1f}")
        log(row)
        rows.append({**r})

    with open(out_dir / "summary.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k not in ('model', 'analytical', 'loss_hist', 'kappa_hist')}
                   for r in all_results], f, indent=2)

    # LaTeX table
    with open(out_dir / "table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llllllll}\n\\hline\n")
        f.write("Case & Noise & Seed & $\\hat{\\kappa}$ & err\\% & $L^2$ rel & $L^\\infty$ & time, min \\\\\n\\hline\n")
        for r in sorted(all_results, key=lambda x: (x['case'], x['noise'], x['seed'])):
            f.write(f"{r['case']} & {r['noise']:.0e} & {r['seed']} & "
                    f"{r['kappa_found']:.3e} & {r['err_pct']:.1f} & "
                    f"{r['l2_rel']:.3e} & {r['linf']:.3e} & {r['time_min']:.1f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    log(f"\nLaTeX table saved: {out_dir / 'table.tex'}")


# ── Main ──────────────────────────────────────────────────────────────────────
log("\n" + "="*72)
log("DIPLOMA PARAMETRIC STUDY")
log(f"  Cases: homogeneous (q=0) + inhomogeneous (q=2xt)")
log(f"  Noise levels: {NOISE_LEVELS}")
log(f"  Seeds: {SEEDS}")
log(f"  Adam={ADAM_EPOCHS}, LBFGS={LBFGS_STEPS}")
log(f"  Total runs: {2 * len(NOISE_LEVELS) * len(SEEDS)}")
log("="*72)

all_results = []

def load_completed(run_dir, case, noise, seed):
    """Load metrics from a completed run. Returns dict or None."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            m = json.load(f)
        # Minimal sanity check
        assert 'kappa_found' in m and 'err_pct' in m
        return m
    except Exception:
        return None


for case in ['homo', 'inhomo']:
    case_dir = OUT / case
    case_dir.mkdir(exist_ok=True)
    for noise in NOISE_LEVELS:
        for seed in SEEDS:
            run_dir = case_dir / f"noise_{noise:.0e}_seed{seed}"
            existing = load_completed(run_dir, case, noise, seed)
            if existing is not None:
                log(f"\n[SKIP] {case} noise={noise:.0e} seed={seed} — "
                    f"already done (kappa={existing['kappa_found']:.4e} err={existing['err_pct']:.2f}%)")
                # Still need model + histories for summary plots — skip heavy plots,
                # but record metrics for the summary table.
                all_results.append(existing)
                continue
            r = train_one(case, noise, seed, run_dir)
            plot_run(r, run_dir)
            all_results.append(r)

# Summary plots
log("\nGenerating summary plots...")
for case in ['homo', 'inhomo']:
    case_dir = OUT / case
    plot_noise_study(all_results, case, case_dir)
    plot_seed_study(all_results, case, case_dir)

plot_homo_vs_inhomo(all_results, OUT)
save_summary_table(all_results, OUT)

log(f"\nAll results saved to: {OUT}")
log("Done.")
LOG_FILE.close()
