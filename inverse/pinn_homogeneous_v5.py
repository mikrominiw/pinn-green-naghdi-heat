"""
PINN v5 - kappa identification with curriculum in T and ResNet architecture.

1. Curriculum in T: train on progressively longer time windows.
   Schedule: T in [10, 50, 100, 250, 500].

2. ResNet architecture: skip-connections through ResBlocks for stable
   gradient flow in deeper networks.

3. Profile scan: after training, sweep kappa on a log grid with frozen network
   weights and plot L_data(kappa). Visualises whether one or two minima remain.
"""

import math
import time
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import autograd, optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- Output dir + logging -----------------------------------------------------
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(__file__).parent.parent / "results_v5" / RUN_ID
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("pinn_v5")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
_fh = logging.FileHandler(RESULTS_DIR / "log.txt", mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_sh)


def log(msg=""):
    logger.info(msg)


# -- Device -------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Run ID: {RUN_ID}")
log(f"Results: {RESULTS_DIR}")
log(f"Device: {device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# -- Constants ----------------------------------------------------------------
PI = math.pi
KAPPA_TRUE = 1.2e-4
T_MAX = 500.0
LAM11 = 2 * PI ** 2
OMEGA0 = math.sqrt(LAM11)   # pi*sqrt(2) -- undamped frequency, geometry only

NOISE_LEVEL = 1e-3
SEED = 1

SENSOR_XY = [(0.25, 0.25), (0.25, 0.75), (0.50, 0.50), (0.75, 0.25), (0.75, 0.75)]
N_TIME = 200

# Base weights: relative scales. Adaptive scheme below rescales the
# non-data terms inversely proportional to their gradient norms (NTK-style
# heuristic) so no term is starved. W_DATA stays fixed -- data is the
# information source for kappa and must always dominate.
W_PDE_BASE = 1.0
W_IC1_BASE = 200.0
W_IC2_BASE = 200.0
W_BC_BASE  = 200.0
W_DATA     = 2000.0

# Adaptive weights (mutable, updated every ADAPT_EVERY epochs)
W = {'pde': W_PDE_BASE, 'ic1': W_IC1_BASE, 'ic2': W_IC2_BASE, 'bc': W_BC_BASE}

# Adaptive weighting disabled by default: experiments showed that
# NTK-style balancing weakens already-satisfied IC/BC terms (their
# gradient becomes small as they are satisfied), creating a positive
# feedback loop that strips kappa of its constraint signal. Fixed
# empirical weights (the base values above) outperformed adaptation
# in our setting -- kappa drifted by 175% with adaptation enabled.
ADAPT_ENABLED = False
ADAPT_EVERY = 200
ADAPT_ALPHA = 0.9
ADAPT_CLIP = (0.1, 1000.0)

# Curriculum schedule: (T_stage, n_adam_epochs)
CURRICULUM = [
    (10.0,  2000),
    (50.0,  2000),
    (100.0, 3000),
    (250.0, 5000),
    (500.0, 8000),
]
LBFGS_STEPS = 300


# -- Analytical solution ------------------------------------------------------
class AnalyticalSolution:
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


analytical = AnalyticalSolution()


# -- Sensor data --------------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

rng = np.random.default_rng(SEED)
t_sensor_np = np.linspace(0.0, T_MAX, N_TIME)

xs, ys, ts, ds = [], [], [], []
for sx, sy in SENSOR_XY:
    vals = analytical(np.full(N_TIME, sx), np.full(N_TIME, sy), t_sensor_np)
    xs.append(np.full(N_TIME, sx)); ys.append(np.full(N_TIME, sy))
    ts.append(t_sensor_np); ds.append(vals)

xs = np.concatenate(xs).reshape(-1, 1)
ys = np.concatenate(ys).reshape(-1, 1)
ts = np.concatenate(ts).reshape(-1, 1)
ds = np.concatenate(ds).reshape(-1, 1)
ds_noisy = ds + rng.standard_normal(ds.shape) * NOISE_LEVEL * np.abs(ds)

x_data = torch.tensor(xs, dtype=torch.float32, device=device)
y_data = torch.tensor(ys, dtype=torch.float32, device=device)
t_data = torch.tensor(ts, dtype=torch.float32, device=device)
data_noisy = torch.tensor(ds_noisy, dtype=torch.float32, device=device)


KAPPA_INIT = 1e-3


# -- PINN with skip-connections (ResNet style) -------------------------------
class ResBlock(nn.Module):
    """
    Pre-activation residual block:
        out = tanh(x + W2 . tanh(W1 . x))
    Skip-connection bypasses two linears; outer tanh keeps values bounded.
    Stabilises gradient flow for deeper nets and helps optimisation of kappa.
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


class PINN(nn.Module):
    """
    Input layer -> N_BLOCKS ResBlocks -> output layer.
    Each ResBlock contains 2 linears (so 6 blocks ~ 12 hidden linears,
    matching the depth of the original 6-layer MLP).
    """
    def __init__(self, kappa_init, hidden=64, n_blocks=3):
        super().__init__()
        self.input_layer = nn.Linear(5, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(hidden, 1)

        for m in (self.input_layer, self.output_layer):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

        self.log_kappa = nn.Parameter(torch.tensor(math.log(kappa_init), dtype=torch.float32))

    def forward(self, x, y, t):
        inp = torch.cat([x, y,
                         torch.sin(OMEGA0 * t), torch.cos(OMEGA0 * t),
                         t / T_MAX], dim=1)
        h = torch.tanh(self.input_layer(inp))
        for block in self.blocks:
            h = block(h)
        N = self.output_layer(h)
        # Hard constraints (ansatz):
        #   IC1:  u(x,y,0)   = sin(pi x) sin(pi y)
        #   IC2:  u_t(x,y,0) = 0
        #   BC:   u = 0 on dOmega  (sin(pi x) sin(pi y) vanishes there)
        # u(x,y,t) = sin(pi x) sin(pi y) * [cos(beta_0 t) + (t/T_MAX)^2 * N(x,y,t)]
        # At t=0: factor = 1 -> matches IC1. d/dt at t=0 = 0 -> matches IC2.
        # The sin*sin envelope kills u on all four edges of [0,1]^2.
        # (t/T_MAX)^2 is bounded in [0,1] so N is not amplified by huge t^2.
        envelope = torch.sin(PI * x) * torch.sin(PI * y)
        tn = t / T_MAX
        return envelope * (torch.cos(OMEGA0 * t) + tn * tn * N)

    def get_kappa(self):
        return torch.exp(self.log_kappa)


# -- Helpers ------------------------------------------------------------------
def _g(u, v):
    return autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                         create_graph=True, retain_graph=True)[0]


def pde_res(model, x, y, t):
    u = model(x, y, t)
    ux, uy, ut = _g(u, x), _g(u, y), _g(u, t)
    uxx, uyy, utt = _g(ux, x), _g(uy, y), _g(ut, t)
    return utt - model.get_kappa() * (_g(uxx, t) + _g(uyy, t)) - (uxx + uyy)


mse = nn.MSELoss()


def spde(n, t_max):
    return (torch.rand(n, 1, requires_grad=True, device=device),
            torch.rand(n, 1, requires_grad=True, device=device),
            torch.rand(n, 1, requires_grad=True, device=device) * t_max)


def sic(n):
    return (torch.rand(n, 1, requires_grad=True, device=device),
            torch.rand(n, 1, requires_grad=True, device=device),
            torch.zeros(n, 1, requires_grad=True, device=device))


def sbc(n, t_max):
    k = n // 4
    rt = lambda: torch.rand(k, 1, requires_grad=True, device=device) * t_max
    x0 = torch.zeros(k, 1, requires_grad=True, device=device)
    x1 = torch.ones(k, 1, requires_grad=True, device=device)
    y0 = torch.zeros(k, 1, requires_grad=True, device=device)
    y1 = torch.ones(k, 1, requires_grad=True, device=device)
    return (
        torch.cat([x0, x1,
                   torch.rand(k, 1, requires_grad=True, device=device),
                   torch.rand(k, 1, requires_grad=True, device=device)]),
        torch.cat([torch.rand(k, 1, requires_grad=True, device=device),
                   torch.rand(k, 1, requires_grad=True, device=device), y0, y1]),
        torch.cat([rt(), rt(), rt(), rt()])
    )


def compute_loss(model, t_max, fp=None, fi=None, fb=None):
    # IC/BC enforced exactly by the ansatz in PINN.forward -- only PDE + data here.
    xp, yp, tp = spde(10000, t_max) if fp is None else tuple(v.detach().requires_grad_(True) for v in fp)

    l_pde = pde_res(model, xp, yp, tp).pow(2).mean()
    l_data = mse(model(x_data, y_data, t_data), data_noisy)

    zero = torch.zeros((), device=device)
    total = W['pde'] * l_pde + W_DATA * l_data
    return total, l_pde, zero, zero, zero, l_data


def grad_norm(loss, params):
    """L2 norm of the loss gradient w.r.t. given parameters."""
    grads = autograd.grad(loss, params, retain_graph=True,
                          create_graph=False, allow_unused=True)
    total_sq = 0.0
    for g in grads:
        if g is not None:
            total_sq += g.detach().pow(2).sum().item()
    return math.sqrt(total_sq)


def update_weights(model, t_max):
    """
    NTK-inspired loss balancing (Wang et al. 2022, "When and why PINNs fail").
    Rescales each non-data weight so that all loss-terms produce gradients
    of comparable norm. W_DATA is held fixed (information source for kappa).

    W_i_new = ||grad(L_data)|| / ||grad(L_i)||  (per-term)
    EMA-smoothed and clipped to prevent collapse to trivial u=0.
    """
    xp, yp, tp = spde(10000, t_max)
    xi, yi, ti = sic(3000)
    xb, yb, tb = sbc(3000, t_max)

    params = [p for p in model.parameters() if p.requires_grad and p is not model.log_kappa]

    l_pde  = pde_res(model, xp, yp, tp).pow(2).mean()
    u_ic   = model(xi, yi, ti)
    l_ic1  = mse(u_ic, torch.sin(PI * xi) * torch.sin(PI * yi))
    l_ic2  = _g(u_ic, ti).pow(2).mean()
    l_bc   = model(xb, yb, tb).pow(2).mean()
    l_data = mse(model(x_data, y_data, t_data), data_noisy)

    g_data = grad_norm(l_data, params)
    if g_data < 1e-12:
        return  # data gradient vanished -- skip update

    new_w = {}
    for name, lval in [('pde', l_pde), ('ic1', l_ic1), ('ic2', l_ic2), ('bc', l_bc)]:
        gi = grad_norm(lval, params)
        if gi < 1e-12:
            new_w[name] = W[name]  # keep current
        else:
            w_target = g_data / gi
            w_target = max(ADAPT_CLIP[0], min(ADAPT_CLIP[1], w_target))
            new_w[name] = ADAPT_ALPHA * W[name] + (1 - ADAPT_ALPHA) * w_target

    for k in new_w:
        W[k] = new_w[k]


# -- Training -----------------------------------------------------------------
model = PINN(kappa_init=KAPPA_INIT).to(device)
log(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")
log(f"kappa init: {model.get_kappa().item():.4e}  (true: {KAPPA_TRUE:.4e})")

loss_history = []
kappa_history = []
stage_boundaries = []
best_loss = float('inf')
best_sd = None
t0 = time.time()

# Two parameter groups: network weights and log_kappa.
# log_kappa lives in [log(1e-6), log(1e-2)] ~ [-13.8, -4.6]; one log-unit move
# is huge in kappa, so it warrants a larger nominal lr than the network weights.
LR_NET_BASE = 1e-3
LR_KAPPA_BASE = 1e-2

net_params = [p for n, p in model.named_parameters() if n != 'log_kappa']
opt = optim.Adam([
    {'params': net_params,         'lr': LR_NET_BASE,   'name': 'net'},
    {'params': [model.log_kappa],  'lr': LR_KAPPA_BASE, 'name': 'kappa'},
])
# ReduceLROnPlateau acts on all param_groups: both lrs decay together.
sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=500,
                                            threshold=0.01, min_lr=1e-6)


def reset_lr_and_scheduler():
    """Restore base lrs and rebuild scheduler -- called at each curriculum stage.
    Each curriculum stage changes T (problem geometry of the loss landscape),
    so the optimiser should not carry over plateau detection from the previous stage."""
    global sch
    for g in opt.param_groups:
        g['lr'] = LR_NET_BASE if g.get('name') == 'net' else LR_KAPPA_BASE
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=500,
                                                threshold=0.01, min_lr=1e-6)


epoch_total = 0
for stage_idx, (T_stage, n_epochs) in enumerate(CURRICULUM):
    log(f"\n=== Stage {stage_idx+1}/{len(CURRICULUM)}: T={T_stage:.0f}  epochs={n_epochs} ===")
    if stage_idx > 0:
        reset_lr_and_scheduler()
        log(f"  lr reset: net={LR_NET_BASE:.1e}  kappa={LR_KAPPA_BASE:.1e}")
    stage_boundaries.append(epoch_total)

    for ep in range(n_epochs):
        if ADAPT_ENABLED and ep % ADAPT_EVERY == 0 and (epoch_total + ep) > 0:
            update_weights(model, T_stage)

        opt.zero_grad()
        L, l_pde, l_ic1, l_ic2, l_bc, l_data = compute_loss(model, T_stage)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(L)

        loss_history.append(L.item())
        kappa_history.append(model.get_kappa().item())

        if L.item() < best_loss:
            best_loss = L.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 500 == 0:
            lr_net = opt.param_groups[0]['lr']
            lr_kap = opt.param_groups[1]['lr']
            log(f"  ep={ep:5d}  loss={L.item():.4e}  "
                f"pde={l_pde.item():.2e}  "
                f"ic1={l_ic1.item():.2e}  ic2={l_ic2.item():.2e}  "
                f"bc={l_bc.item():.2e}  data={l_data.item():.2e}  "
                f"kappa={model.get_kappa().item():.4e}  "
                f"lr_net={lr_net:.2e}  lr_kap={lr_kap:.2e}  "
                f"W=[pde={W['pde']:.1f} ic1={W['ic1']:.1f} "
                f"ic2={W['ic2']:.1f} bc={W['bc']:.1f}]")

    epoch_total += n_epochs

model.load_state_dict(best_sd)
log(f"\nAfter curriculum: kappa={model.get_kappa().item():.4e}"
      f"  err={abs(model.get_kappa().item()-KAPPA_TRUE)/KAPPA_TRUE*100:.2f}%")

# -- L-BFGS on full T_MAX -----------------------------------------------------
log(f"\n=== L-BFGS ({LBFGS_STEPS} steps, T={T_MAX}) ===")
fp = spde(10000, T_MAX); fi = sic(3000); fb = sbc(3000, T_MAX)
lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=50,
                    history_size=100, line_search_fn='strong_wolfe')


# Components from the latest closure call (filled in for logging)
last_components = {}


def closure():
    lbfgs.zero_grad()
    L, l_pde, l_ic1, l_ic2, l_bc, l_data = compute_loss(model, T_MAX, fp=fp, fi=fi, fb=fb)
    L.backward()
    last_components['pde']  = l_pde.item()
    last_components['ic1']  = l_ic1.item()
    last_components['ic2']  = l_ic2.item()
    last_components['bc']   = l_bc.item()
    last_components['data'] = l_data.item()
    return L


for step in range(LBFGS_STEPS):
    try:
        L = lbfgs.step(closure)
    except Exception as e:
        log(f"  L-BFGS step {step}: {type(e).__name__} -- stopping"); break
    if L is None or not math.isfinite(L.item()):
        log(f"  L-BFGS step {step}: non-finite -- stopping"); break
    loss_history.append(L.item())
    kappa_history.append(model.get_kappa().item())
    if L.item() < best_loss:
        best_loss = L.item()
        best_sd = {k: v.clone() for k, v in model.state_dict().items()}
    if step % 50 == 0:
        log(f"  step={step:3d}  loss={L.item():.4e}  "
            f"pde={last_components['pde']:.2e}  "
            f"ic1={last_components['ic1']:.2e}  ic2={last_components['ic2']:.2e}  "
            f"bc={last_components['bc']:.2e}  data={last_components['data']:.2e}  "
            f"kappa={model.get_kappa().item():.4e}")

model.load_state_dict(best_sd)
elapsed = time.time() - t0

# -- Metrics ------------------------------------------------------------------
kappa_found = model.get_kappa().item()
err_pct = abs(kappa_found - KAPPA_TRUE) / KAPPA_TRUE * 100

model.eval()
with torch.no_grad():
    xt = torch.rand(5000, 1, device=device)
    yt = torch.rand(5000, 1, device=device)
    tt = torch.rand(5000, 1, device=device) * T_MAX
    up = model(xt, yt, tt)
ue = torch.tensor(analytical(xt.cpu(), yt.cpu(), tt.cpu()).reshape(-1, 1), dtype=torch.float32)
l2_rel = (torch.norm(up.cpu() - ue) / torch.norm(ue)).item()
linf   = torch.max((up.cpu() - ue).abs()).item()

log(f"\n{'='*55}")
log(f"True  kappa = {KAPPA_TRUE:.4e}")
log(f"Init  kappa = {KAPPA_INIT:.4e}")
log(f"Found kappa = {kappa_found:.4e}")
log(f"Relative error = {err_pct:.2f}%")
log(f"L2 rel error (field) = {l2_rel:.3e}")
log(f"L-inf error  (field) = {linf:.3e}")
log(f"Time = {elapsed/60:.1f} min")
log(f"{'='*55}")


# -- Profile scan: L_data(kappa) with frozen network -------------------------
log("\nProfile scan: sweeping kappa with frozen network...")
kappa_scan = np.logspace(-5, -3, 80)
data_losses_scan = []
model.eval()
for k in kappa_scan:
    with torch.no_grad():
        model.log_kappa.data = torch.tensor(math.log(k), device=device)
        l = mse(model(x_data, y_data, t_data), data_noisy).item()
    data_losses_scan.append(l)
model.log_kappa.data = torch.tensor(math.log(kappa_found), device=device)


# -- Plots --------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 1. Loss + kappa convergence
ax1 = axes[0, 0]
ax1.semilogy(loss_history, 'b-', lw=0.7, label='Total loss')
for i, (sb, (T_s, _)) in enumerate(zip(stage_boundaries, CURRICULUM)):
    ax1.axvline(sb, color='green', ls=':', lw=0.8, alpha=0.7,
                label=f'T={T_s:.0f}' if i < 3 else None)
ax1.axvline(epoch_total, color='gray', ls='--', lw=0.8, label='Adam->L-BFGS')
ax1.set_xlabel('Iteration'); ax1.set_ylabel('Loss (log)', color='b')
ax1.tick_params(axis='y', colors='b')
ax1.grid(True, which='both', alpha=0.3)
ax2 = ax1.twinx()
ax2.semilogy(kappa_history, 'r-', lw=0.9, alpha=0.8)
ax2.axhline(KAPPA_TRUE, color='darkred', ls=':', lw=1.5, label=f'kappa_true={KAPPA_TRUE:.1e}')
ax2.axhline(KAPPA_INIT,  color='orange',  ls='--', lw=1.0, label=f'KAPPA_INIT={KAPPA_INIT:.1e}')
ax2.set_ylabel('kappa (log)', color='r')
ax2.tick_params(axis='y', colors='r')
lines1, l1 = ax1.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, l1 + l2, fontsize=8, loc='upper right')
ax1.set_title(f'Curriculum convergence  kappa={kappa_found:.3e}  err={err_pct:.1f}%')

# 2. Profile L_data(kappa)
ax = axes[0, 1]
ax.semilogy(kappa_scan, data_losses_scan, 'b-', lw=1.5)
ax.axvline(KAPPA_TRUE,  color='k',      ls=':',  lw=1.5, label=f'kappa_true={KAPPA_TRUE:.1e}')
ax.axvline(kappa_found, color='r',      ls='--', lw=1.2, label=f'kappa_found={kappa_found:.1e}')
ax.axvline(KAPPA_INIT,  color='orange', ls='--', lw=1.0, label=f'KAPPA_INIT={KAPPA_INIT:.1e}')
ax.set_xscale('log')
ax.set_xlabel('kappa'); ax.set_ylabel('L_data (frozen network)')
ax.set_title('Profile: data loss vs kappa (frozen network)')
ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

# 3. Temporal profile
ax = axes[1, 0]
t_plot = np.linspace(0, T_MAX, 1000)
A_exact = analytical.amplitude(t_plot)
x05 = torch.full((1000, 1), 0.5, device=device)
y05 = torch.full((1000, 1), 0.5, device=device)
tp  = torch.tensor(t_plot.reshape(-1, 1), dtype=torch.float32, device=device)
with torch.no_grad():
    u_center = model(x05, y05, tp).cpu().numpy().flatten()
ax.plot(t_plot, A_exact,  'b-',  lw=1.5, label='Analytical A(t)')
ax.plot(t_plot, u_center, 'r--', lw=1.0, label=f'PINN  kappa={kappa_found:.3e}')
ax.set_xlabel('t'); ax.set_ylabel('u(0.5, 0.5, t)')
ax.set_title(f'Temporal profile  err={err_pct:.2f}%  L2={l2_rel:.3e}')
ax.legend(); ax.grid(True, alpha=0.3)

# 4. Pointwise error over time
ax = axes[1, 1]
ax.semilogy(t_plot, np.abs(u_center - A_exact) + 1e-15, 'g-', lw=0.8)
ax.set_xlabel('t'); ax.set_ylabel('|PINN - Analytical|')
ax.set_title('Pointwise error at (0.5, 0.5)')
ax.grid(True, which='both', alpha=0.3)

plt.suptitle(f'PINN v5 -- Curriculum + ResNet  noise={NOISE_LEVEL:.0e}  seed={SEED}',
             fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'results.png', dpi=150)
log(f"Saved: {RESULTS_DIR / 'results.png'}")
plt.close()

# Field snapshots
t_snaps = [1.0, 10.0, 50.0, 100.0, 250.0, 500.0]
n_sp = 50
xg = torch.linspace(0, 1, n_sp, device=device)
yg = torch.linspace(0, 1, n_sp, device=device)
X, Y = torch.meshgrid(xg, yg, indexing='ij')
Xf, Yf = X.reshape(-1, 1), Y.reshape(-1, 1)

fig2, axes2 = plt.subplots(3, len(t_snaps), figsize=(20, 9))
for i, tv in enumerate(t_snaps):
    Tf = torch.full_like(Xf, tv)
    with torch.no_grad():
        up = model(Xf, Yf, Tf).reshape(n_sp, n_sp).cpu().numpy()
    ua   = analytical(Xf.cpu(), Yf.cpu(), Tf.cpu()).reshape(n_sp, n_sp)
    diff = up - ua
    vlim = max(abs(up).max(), abs(ua).max(), 1e-12)
    dlim = max(abs(diff).max(), 1e-12)
    for row, (data, title, cmap, vmin, vmax) in enumerate([
        (up,   f'PINN  t={tv:g}',       'RdBu_r',  -vlim, vlim),
        (ua,   f'Exact t={tv:g}',       'RdBu_r',  -vlim, vlim),
        (diff, f'Diff  max={dlim:.1e}', 'seismic', -dlim, dlim),
    ]):
        c = axes2[row, i].contourf(X.cpu(), Y.cpu(), data, 30,
                                   cmap=cmap, vmin=vmin, vmax=vmax)
        axes2[row, i].set_title(title, fontsize=9)
        plt.colorbar(c, ax=axes2[row, i], fraction=0.046, pad=0.04)
        axes2[row, i].set_xlabel('x'); axes2[row, i].set_ylabel('y')

plt.suptitle(f'Field snapshots  kappa={kappa_found:.3e}'
             f'  (true {KAPPA_TRUE:.1e}  err {err_pct:.1f}%)', fontsize=11)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'snapshots.png', dpi=120)
log(f"Saved: {RESULTS_DIR / 'snapshots.png'}")
plt.close()

torch.save(best_sd, RESULTS_DIR / 'best.pth')
log("\nDone.")
