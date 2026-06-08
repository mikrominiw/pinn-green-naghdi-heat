"""
Quick test for v3 improvements over pinn_homogeneous_v2.py.

Changes (all within inverse-problem physics — no kappa_true leakage):
  1. N_TIME: 200 -> 500     (denser sensor sampling, realistic ADC rate)
  2. kappa = sigmoid bounded to [1e-7, 1e-2]
     (we know material is metal -> 5-orders physical range)
  3. kappa warm-up: frozen for first 3000 epochs (algorithmic)
  4. Multi-restart x3, pick run with lowest LOSS (no kappa_true used)
  5. Reduced training per restart: 10000 Adam + 150 L-BFGS
     (total: 3 x ~25min = ~75min vs 50min single full run)

If this gives stable kappa < 20% error across all 3 restarts, we
base study_parametric.py on this template.

Run:  .venv/Scripts/python.exe example/pinn_homogeneous_v3_test.py
"""

import math, time
import numpy as np
import torch
import torch.nn as nn
from torch import autograd, optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Constants ─────────────────────────────────────────────────────────────────
PI = math.pi
KAPPA_TRUE = 1.2e-4          # used ONLY for data generation + final validation
T_MAX = 500.0
LAM11 = 2 * PI ** 2
OMEGA0 = math.sqrt(LAM11)

# Physical bounds for kappa (knowledge of material class, not its value)
KAPPA_MIN = 1e-7
KAPPA_MAX = 1e-2
KAPPA_INIT = 1e-3            # neutral starting point (in middle of log-range)


# ── Analytical solution (data generation only) ────────────────────────────────
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


# ── PINN with bounded kappa (sigmoid) ─────────────────────────────────────────
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

        # theta -> kappa via sigmoid map: kappa = KMIN + (KMAX - KMIN) * sigmoid(theta)
        # Init theta so kappa = KAPPA_INIT
        ratio = (KAPPA_INIT - KAPPA_MIN) / (KAPPA_MAX - KAPPA_MIN)
        theta_init = math.log(ratio / (1.0 - ratio))   # logit
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))

    def forward(self, x, y, t):
        s = torch.sin(self.omega0 * t)
        c = torch.cos(self.omega0 * t)
        tau = t / self.t_max
        return self.net(torch.cat([x, y, s, c, tau], dim=1))

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
    xL = x0; yL = torch.rand(k, 1, requires_grad=True, device=device); tL = rt()
    xR = x1; yR = torch.rand(k, 1, requires_grad=True, device=device); tR = rt()
    xB = torch.rand(k, 1, requires_grad=True, device=device); yB = y0; tB = rt()
    xT = torch.rand(k, 1, requires_grad=True, device=device); yT = y1; tT = rt()
    return (torch.cat([xL, xR, xB, xT]),
            torch.cat([yL, yR, yB, yT]),
            torch.cat([tL, tR, tB, tT]))


def _g(u, v):
    return autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                         create_graph=True, retain_graph=True)[0]


def pde_res(model, x, y, t):
    u = model(x, y, t)
    ux = _g(u, x); uy = _g(u, y); ut = _g(u, t)
    uxx = _g(ux, x); uyy = _g(uy, y); utt = _g(ut, t)
    uxxt = _g(uxx, t); uyyt = _g(uyy, t)
    return utt - model.get_kappa() * (uxxt + uyyt) - (uxx + uyy)


mse = nn.MSELoss()
W_PDE, W_IC1, W_IC2, W_BC, W_DATA = 1.0, 200.0, 200.0, 200.0, 2000.0


# ── Sensors (realistic: denser time grid) ─────────────────────────────────────
SENSOR_XY = [(0.25, 0.25), (0.25, 0.75), (0.50, 0.50), (0.75, 0.25), (0.75, 0.75)]
N_TIME = 500        # increased from 200 (still well within ADC rate)
NOISE_LEVEL = 1e-5


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
    # Realistic measurement noise (multiplicative)
    noise = rng.standard_normal(ds.shape) * NOISE_LEVEL * np.abs(ds)
    ds_noisy = ds + noise
    return (torch.tensor(xs, dtype=torch.float32, device=device),
            torch.tensor(ys, dtype=torch.float32, device=device),
            torch.tensor(ts, dtype=torch.float32, device=device),
            torch.tensor(ds_noisy, dtype=torch.float32, device=device))


# ── Single training run with warm-up ──────────────────────────────────────────
ADAM_EPOCHS = 10000
WARMUP_EPOCHS = 3000     # kappa frozen for first 3000 epochs
LBFGS_STEPS = 150


def train_one(seed, xd, yd, td, dn, label):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = PINN().to(device)
    print(f"\n[{label}] seed={seed}  kappa_init={model.get_kappa().item():.4e}")

    # Phase 1: warm-up (kappa frozen)
    model.theta.requires_grad_(False)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1000, min_lr=1e-6)

    best_loss = float('inf')
    best_sd = None
    t0 = time.time()

    def compute_loss():
        xp, yp, tp = spde(10000)
        xi, yi, ti = sic(3000)
        xb, yb, tb = sbc(3000)
        l_pde = pde_res(model, xp, yp, tp).pow(2).mean()
        u_ic = model(xi, yi, ti)
        l_ic1 = mse(u_ic, torch.sin(PI * xi) * torch.sin(PI * yi))
        l_ic2 = _g(u_ic, ti).pow(2).mean()
        l_bc = model(xb, yb, tb).pow(2).mean()
        l_data = mse(model(xd, yd, td), dn)
        return W_PDE*l_pde + W_IC1*l_ic1 + W_IC2*l_ic2 + W_BC*l_bc + W_DATA*l_data

    # Warm-up
    for ep in range(WARMUP_EPOCHS):
        opt.zero_grad()
        L = compute_loss()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(L)
        if L.item() < best_loss:
            best_loss = L.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 500 == 0:
            print(f"  [warm-up] ep={ep:5d}  loss={L.item():.4e}  kappa={model.get_kappa().item():.4e}")

    # Unfreeze kappa, rebuild optimizer to include theta
    model.theta.requires_grad_(True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1000, min_lr=1e-6)

    # Phase 2: joint Adam
    for ep in range(WARMUP_EPOCHS, ADAM_EPOCHS):
        opt.zero_grad()
        L = compute_loss()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(L)
        if L.item() < best_loss:
            best_loss = L.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 500 == 0:
            print(f"  [joint]   ep={ep:5d}  loss={L.item():.4e}  kappa={model.get_kappa().item():.4e}")

    model.load_state_dict(best_sd)

    # Phase 3: L-BFGS
    fp = spde(10000); fi = sic(3000); fb = sbc(3000)
    lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=50,
                        history_size=100, line_search_fn='strong_wolfe')

    def closure():
        lbfgs.zero_grad()
        xp, yp, tp = (v.detach().requires_grad_(True) for v in fp)
        xi, yi, ti = (v.detach().requires_grad_(True) for v in fi)
        xb, yb, tb = (v.detach().requires_grad_(True) for v in fb)
        l_pde = pde_res(model, xp, yp, tp).pow(2).mean()
        u_ic = model(xi, yi, ti)
        l_ic1 = mse(u_ic, torch.sin(PI * xi) * torch.sin(PI * yi))
        l_ic2 = _g(u_ic, ti).pow(2).mean()
        l_bc = model(xb, yb, tb).pow(2).mean()
        l_data = mse(model(xd, yd, td), dn)
        L = W_PDE*l_pde + W_IC1*l_ic1 + W_IC2*l_ic2 + W_BC*l_bc + W_DATA*l_data
        L.backward()
        return L

    for step in range(LBFGS_STEPS):
        L = lbfgs.step(closure)
        if L is not None and L.item() < best_loss:
            best_loss = L.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if step % 30 == 0:
            print(f"  [L-BFGS]  step={step:3d}  loss={L.item():.4e}  kappa={model.get_kappa().item():.4e}")

    model.load_state_dict(best_sd)
    elapsed = time.time() - t0

    # Final metrics
    kf = model.get_kappa().item()
    err = abs(kf - KAPPA_TRUE) / KAPPA_TRUE * 100

    # Field validation (only for diploma report, not used for selection)
    model.eval()
    with torch.no_grad():
        xt = torch.rand(2000, 1, device=device)
        yt = torch.rand(2000, 1, device=device)
        tt = torch.rand(2000, 1, device=device) * T_MAX
        up = model(xt, yt, tt)
    ue = torch.tensor(analytical(xt.cpu(), yt.cpu(), tt.cpu()).reshape(-1, 1),
                      dtype=torch.float32)
    l2 = (torch.norm(up.cpu() - ue) / torch.norm(ue)).item()

    print(f"[{label}] DONE  loss={best_loss:.4e}  kappa={kf:.4e}  err={err:.2f}%  "
          f"L2={l2:.3e}  t={elapsed/60:.1f}min")
    return {'seed': seed, 'loss': best_loss, 'kappa': kf, 'err': err, 'l2': l2, 'time': elapsed}


# ── Multi-restart ─────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("MULTI-RESTART TEST  (3 seeds, selection by final LOSS)")
print(f"  N_TIME={N_TIME}, noise={NOISE_LEVEL:.0e}, kappa in [{KAPPA_MIN:.0e},{KAPPA_MAX:.0e}]")
print(f"  warm-up={WARMUP_EPOCHS}, adam={ADAM_EPOCHS}, lbfgs={LBFGS_STEPS}")
print("="*72)

# Generate sensor data ONCE (same data for all restarts — realistic experiment)
xd, yd, td, dn = make_sensor_data(seed=0)

results = []
for r, seed in enumerate([1, 2, 3], 1):
    res = train_one(seed, xd, yd, td, dn, label=f"R{r}")
    results.append(res)

# Selection by loss
best = min(results, key=lambda r: r['loss'])

print("\n" + "="*72)
print("SUMMARY")
print("="*72)
print(f"  {'Run':<6}{'seed':<6}{'loss':<14}{'kappa':<14}{'err%':<10}{'L2':<12}{'time':<10}")
for i, r in enumerate(results, 1):
    mark = "  <-- BEST (by loss)" if r is best else ""
    print(f"  R{i:<5}{r['seed']:<6}{r['loss']:<14.4e}{r['kappa']:<14.4e}"
          f"{r['err']:<10.2f}{r['l2']:<12.3e}{r['time']/60:<10.1f}{mark}")

print(f"\nSelected by lowest loss: kappa={best['kappa']:.4e}  "
      f"err={best['err']:.2f}%  L2={best['l2']:.3e}")
print(f"True kappa = {KAPPA_TRUE:.4e}")
