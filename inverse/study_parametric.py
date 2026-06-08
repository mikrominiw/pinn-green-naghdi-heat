"""
Parametric study for the diploma:
  - Noise sensitivity:   NOISE_LEVEL in {1e-5, 1e-4, 1e-3, 1e-2}
  - Sensor count:        N_SENSORS   in {3, 5, 7, 10}

Uses the homogeneous GN-III case (q=0) with reduced training
(10 000 Adam + 100 L-BFGS per run) to keep total wall-time reasonable
while showing reliable trends.

Results are printed to console and saved to  diploma_parametric.txt
Run:  .venv/Scripts/python.exe example/study_parametric.py
"""

import math, time, sys
import numpy as np
import torch
import torch.nn as nn
from torch import autograd, optim

# ── redirect print to both console and file ───────────────────────────────────
LOG_FILE = open("diploma_parametric.txt", "w", encoding="utf-8")

def log(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=LOG_FILE)
    LOG_FILE.flush()

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Physical constants ────────────────────────────────────────────────────────
PI       = math.pi
KAPPA_TRUE = 1.2e-4
T_MAX    = 500.0
LAM11    = 2 * PI ** 2
OMEGA0   = math.sqrt(LAM11)   # encoding frequency, known from geometry


# ── Analytical solution (homogeneous) ────────────────────────────────────────
class AnalyticalSolution:
    def __init__(self, kappa=KAPPA_TRUE):
        lam  = LAM11
        disc = (kappa * lam) ** 2 - 4 * lam
        self.alpha = kappa * lam / 2
        self.beta  = math.sqrt(-disc) / 2

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
    def __init__(self, hidden=128, n_layers=6, t_max=T_MAX, kappa_init=1e-3):
        super().__init__()
        self.t_max  = t_max
        self.omega0 = OMEGA0
        in_dim = 5
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.log_kappa = nn.Parameter(
            torch.tensor(math.log(kappa_init), dtype=torch.float32))

    def forward(self, x, y, t):
        inp = torch.cat([x, y,
                         torch.sin(self.omega0 * t),
                         torch.cos(self.omega0 * t),
                         t / self.t_max], dim=1)
        return self.net(inp)

    def get_kappa(self):
        return torch.exp(self.log_kappa)


# ── Samplers ──────────────────────────────────────────────────────────────────
def spde(n):
    x = torch.rand(n,1,requires_grad=True,device=device)
    y = torch.rand(n,1,requires_grad=True,device=device)
    t = torch.rand(n,1,requires_grad=True,device=device)*T_MAX
    return x, y, t

def sic(n):
    x = torch.rand(n,1,requires_grad=True,device=device)
    y = torch.rand(n,1,requires_grad=True,device=device)
    t = torch.zeros(n,1,requires_grad=True,device=device)
    return x, y, t

def sbc(n):
    k = n//4
    randt = lambda: torch.rand(k,1,requires_grad=True,device=device)*T_MAX
    x0=torch.zeros(k,1,requires_grad=True,device=device)
    x1=torch.ones(k,1,requires_grad=True,device=device)
    y0=torch.zeros(k,1,requires_grad=True,device=device)
    y1=torch.ones(k,1,requires_grad=True,device=device)
    xL=x0; yL=torch.rand(k,1,requires_grad=True,device=device); tL=randt()
    xR=x1; yR=torch.rand(k,1,requires_grad=True,device=device); tR=randt()
    xB=torch.rand(k,1,requires_grad=True,device=device); yB=y0; tB=randt()
    xT=torch.rand(k,1,requires_grad=True,device=device); yT=y1; tT=randt()
    return (torch.cat([xL,xR,xB,xT]),
            torch.cat([yL,yR,yB,yT]),
            torch.cat([tL,tR,tB,tT]))

def _g(u, v):
    return autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                         create_graph=True, retain_graph=True)[0]

def pde_res(model, x, y, t):
    u=model(x,y,t); ux=_g(u,x); uy=_g(u,y); ut=_g(u,t)
    uxx=_g(ux,x); uyy=_g(uy,y); utt=_g(ut,t)
    uxxt=_g(uxx,t); uyyt=_g(uyy,t)
    return utt - model.get_kappa()*(uxxt+uyyt) - (uxx+uyy)

mse = nn.MSELoss()
W_PDE=1.; W_IC1=200.; W_IC2=200.; W_BC=200.; W_DATA=2000.


# ── Build sensor data ─────────────────────────────────────────────────────────
SENSOR_PRESETS = {
    3:  [(0.25,0.25),(0.50,0.50),(0.75,0.75)],
    5:  [(0.25,0.25),(0.25,0.75),(0.50,0.50),(0.75,0.25),(0.75,0.75)],
    7:  [(0.25,0.25),(0.25,0.75),(0.50,0.50),(0.75,0.25),(0.75,0.75),
         (0.25,0.50),(0.75,0.50)],
    10: [(0.25,0.25),(0.25,0.75),(0.50,0.50),(0.75,0.25),(0.75,0.75),
         (0.25,0.50),(0.75,0.50),(0.50,0.25),(0.50,0.75),(0.50,0.50)],
}
N_TIME = 200
t_s = np.linspace(0, T_MAX, N_TIME)

def make_sensor_tensors(n_sensors, noise_level):
    sensors = SENSOR_PRESETS[n_sensors]
    xs, ys, ts, ds = [], [], [], []
    for sx, sy in sensors:
        vals = analytical(np.full(N_TIME,sx), np.full(N_TIME,sy), t_s)
        xs.append(np.full(N_TIME,sx)); ys.append(np.full(N_TIME,sy))
        ts.append(t_s); ds.append(vals)
    xs = np.concatenate(xs).reshape(-1,1)
    ys = np.concatenate(ys).reshape(-1,1)
    ts = np.concatenate(ts).reshape(-1,1)
    ds = np.concatenate(ds).reshape(-1,1)
    xd=torch.tensor(xs,dtype=torch.float32,device=device)
    yd=torch.tensor(ys,dtype=torch.float32,device=device)
    td=torch.tensor(ts,dtype=torch.float32,device=device)
    dt=torch.tensor(ds,dtype=torch.float32,device=device)
    dn = dt + noise_level * torch.randn_like(dt) * dt.abs()
    return xd, yd, td, dn


# ── Single experiment ─────────────────────────────────────────────────────────
ADAM_EPOCHS  = 20000
LBFGS_STEPS  = 300

def run_experiment(noise_level, n_sensors, label):
    xd, yd, td, dn = make_sensor_tensors(n_sensors, noise_level)

    model = PINN().to(device)
    best_loss = float('inf')
    t0 = time.time()

    def loss_fn(fp=None, fi=None, fb=None):
        if fp is None:
            xp,yp,tp = spde(10000); xi,yi,ti = sic(3000); xb,yb,tb = sbc(3000)
        else:
            xp,yp,tp = (v.detach().requires_grad_(True) for v in fp)
            xi,yi,ti = (v.detach().requires_grad_(True) for v in fi)
            xb,yb,tb = (v.detach().requires_grad_(True) for v in fb)

        res   = pde_res(model,xp,yp,tp)
        l_pde = res.pow(2).mean()

        u_ic  = model(xi,yi,ti)
        l_ic1 = mse(u_ic, torch.sin(PI*xi)*torch.sin(PI*yi))
        l_ic2 = _g(u_ic,ti).pow(2).mean()

        l_bc   = model(xb,yb,tb).pow(2).mean()
        l_data = mse(model(xd,yd,td), dn)

        return (W_PDE*l_pde + W_IC1*l_ic1 + W_IC2*l_ic2
                + W_BC*l_bc + W_DATA*l_data)

    # Phase 1: Adam
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1000, min_lr=1e-6)
    best_sd = None
    for ep in range(ADAM_EPOCHS):
        opt.zero_grad()
        L = loss_fn()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(L)
        if L.item() < best_loss:
            best_loss = L.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)

    # Phase 2: L-BFGS
    fp = spde(10000); fi = sic(3000); fb = sbc(3000)
    lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=50,
                        history_size=100, line_search_fn='strong_wolfe')
    def closure():
        lbfgs.zero_grad()
        L = loss_fn(fp, fi, fb)
        L.backward(); return L

    for _ in range(LBFGS_STEPS):
        L = lbfgs.step(closure)
        if L is not None and L.item() < best_loss:
            best_loss = L.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)
    elapsed = time.time() - t0

    # Validation
    kf = model.get_kappa().item()
    err = abs(kf - KAPPA_TRUE) / KAPPA_TRUE * 100

    model.eval()
    with torch.no_grad():
        n_test = 2000
        xt = torch.rand(n_test,1,device=device)
        yt = torch.rand(n_test,1,device=device)
        tt = torch.rand(n_test,1,device=device)*T_MAX
        up = model(xt,yt,tt)
    ue = torch.tensor(
        analytical(xt.cpu(),yt.cpu(),tt.cpu()).reshape(-1,1), dtype=torch.float32)
    l2  = (torch.norm(up.cpu()-ue)/torch.norm(ue)).item()
    linf= torch.max((up.cpu()-ue).abs()).item()

    log(f"  {label:<35s}  kappa={kf:.4e}  err={err:6.2f}%  "
        f"L2={l2:.3e}  Linf={linf:.3e}  t={elapsed/60:.1f}min")
    return kf, err, l2, linf, elapsed


# ── Study 1: noise sensitivity (n_sensors=5 fixed) ───────────────────────────
log("\n" + "="*70)
log("STUDY 1: Noise sensitivity   (n_sensors=5, Adam=20000, LBFGS=300)")
log("="*70)
log(f"  {'Config':<35s}  kappa_found   err%    L2       Linf     time")
log("-"*70)

noise_results = {}
for nl in [1e-5, 1e-4, 1e-3, 1e-2]:
    label = f"noise={nl:.0e}"
    kf, err, l2, linf, t = run_experiment(noise_level=nl, n_sensors=5, label=label)
    noise_results[nl] = (kf, err, l2, linf, t)

log("\nSummary - noise sensitivity:")
log(f"  {'Noise':<12} {'kappa_true':>12} {'kappa_found':>12} {'Err %':>8} {'L2 rel':>10}")
log("-"*60)
for nl, (kf, err, l2, linf, t) in noise_results.items():
    log(f"  {nl:<12.0e} {KAPPA_TRUE:>12.4e} {kf:>12.4e} {err:>8.2f} {l2:>10.3e}")


# ── Study 2: sensor count (noise=1e-5 fixed) ─────────────────────────────────
log("\n" + "="*70)
log("STUDY 2: Sensor count   (noise=1e-5, Adam=20000, LBFGS=300)")
log("="*70)
log(f"  {'Config':<35s}  kappa_found   err%    L2       Linf     time")
log("-"*70)

sensor_results = {}
for ns in [3, 5, 7, 10]:
    label = f"n_sensors={ns}"
    kf, err, l2, linf, t = run_experiment(noise_level=1e-5, n_sensors=ns, label=label)
    sensor_results[ns] = (kf, err, l2, linf, t)

log("\nSummary - sensor count:")
log(f"  {'N sensors':<12} {'kappa_true':>12} {'kappa_found':>12} {'Err %':>8} {'L2 rel':>10}")
log("-"*60)
for ns, (kf, err, l2, linf, t) in sensor_results.items():
    log(f"  {ns:<12d} {KAPPA_TRUE:>12.4e} {kf:>12.4e} {err:>8.2f} {l2:>10.3e}")


log("\n" + "="*70)
log("All done. Results saved to diploma_parametric.txt")
LOG_FILE.close()
