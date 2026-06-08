"""
PINN for inverse problem: identify kappa in GN-III INhomogeneous heat equation.

PDE (dimensionless, q = 2xt):
    u_tt - kappa*(u_xxt + u_yyt) - (u_xx + u_yy) = dq/dt = 2x
    (x,y) in (0,1)^2,  t in [0, T]

IC:  u(x,y,0) = sin(pi*x)*sin(pi*y),   u_t(x,y,0) = 0
BC:  u = 0  on all four edges

True kappa = 1.2e-4  (duralumin D16T, dimensionless)

Analytical solution (many modes contribute):
    u(x,y,t) = sum_{n,m} [C1_nm*exp(r1*t) + C2_nm*exp(r2*t) + f_nm/lam_nm] * sin(n*pi*x)*sin(m*pi*y)

    Forcing Fourier coefficient:
        f_nm = 8*(-1)^{n+1}/(n*pi) * (1-(-1)^m)/(m*pi)
    Non-zero only for ODD m; all n contribute.

    For the (1,1) mode (IC + forcing):  A_11=1, f_11=16/pi^2
    For all other modes:                A_nm=0, f_nm non-zero for odd m

Key improvements over existing code (same as homogeneous v2):
1. t_max=500: decay ~45%, giving clear signal about kappa
2. Two Fourier time features: omega1=pi*sqrt(2) (mode 1,1) and
   omega2=pi*sqrt(5) (mode 2,1 — second largest amplitude)
3. Joint training of network + kappa from scratch
4. kappa initialized at 1e-3
5. High data weight (w_data=2000)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch import autograd, optim
import matplotlib.pyplot as plt

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Constants ─────────────────────────────────────────────────────────────────
PI = math.pi
KAPPA_TRUE = 1.2e-4
T_MAX = 500.0

# Eigenvalue lambda_nm = pi^2*(n^2+m^2), known from domain geometry and BCs.
# Fourier encoding frequencies = undamped natural frequencies sqrt(lambda_nm),
# entirely determined by geometry — no kappa involved.
OMEGA1 = math.sqrt(2) * PI   # sqrt(lambda_11) = pi*sqrt(2) ≈ 4.443, mode (1,1)
OMEGA2 = math.sqrt(5) * PI   # sqrt(lambda_21) = pi*sqrt(5) ≈ 7.025, mode (2,1)


# ── Analytical solution ───────────────────────────────────────────────────────
class AnalyticalSolutionFull:
    """
    Exact multi-mode solution for q = 2xt.

    Modes: all n >= 1 with odd m (even m give f_nm = 0 and A_nm = 0).
    Dominant steady-state contributions:
        f_11/lam_11 = 8/pi^4 ≈ 0.082  (mode 1,1)
        f_21/lam_21 = -8/(5*pi^4) ≈ -0.016  (mode 2,1)
        f_31/lam_31 ≈ +0.005, etc.
    """

    def __init__(self, kappa=KAPPA_TRUE, n_max=30, m_max=30):
        self.kappa = kappa
        self.n_max = n_max
        self.m_max = m_max

    def __call__(self, x, y, t):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy().flatten()
            y = y.detach().cpu().numpy().flatten()
            t = t.detach().cpu().numpy().flatten()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        t = np.asarray(t, dtype=float)
        result = np.zeros_like(x)

        for n in range(1, self.n_max + 1):
            for m in range(1, self.m_max + 1):
                lam = PI ** 2 * (n ** 2 + m ** 2)

                A_nm = 1.0 if (n == 1 and m == 1) else 0.0

                # f_nm = 4 * integral_xy of 2x * sin(n*pi*x)*sin(m*pi*y) dxdy
                int_x = (-1) ** (n + 1) / (n * PI)
                int_y = (1 - (-1) ** m) / (m * PI)
                f_nm = 4 * 2 * int_x * int_y   # = 8*int_x*int_y

                if abs(A_nm) < 1e-12 and abs(f_nm) < 1e-12:
                    continue  # even m: skip entirely

                disc = (self.kappa * lam) ** 2 - 4 * lam

                if disc < 0:
                    real = -self.kappa * lam / 2
                    imag = math.sqrt(-disc) / 2
                    # T(t) = exp(real*t)*[D1*cos(imag*t) + D2*sin(imag*t)] + f/lam
                    # D1 = A - f/lam,  D2 = -real*D1/imag  (from T(0)=A, T'(0)=0)
                    D1 = A_nm - f_nm / lam
                    D2 = -real * D1 / imag
                    T_t = (np.exp(real * t) * (D1 * np.cos(imag * t) + D2 * np.sin(imag * t))
                           + f_nm / lam)
                else:
                    r1 = (-self.kappa * lam + math.sqrt(disc)) / 2
                    r2 = (-self.kappa * lam - math.sqrt(disc)) / 2
                    diff = r1 - r2
                    if abs(diff) > 1e-12:
                        C1 = -r2 * (A_nm - f_nm / lam) / diff
                        C2 = r1 * (A_nm - f_nm / lam) / diff
                    else:
                        C1 = C2 = (A_nm - f_nm / lam) / 2
                    T_t = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t) + f_nm / lam

                result += T_t * np.sin(n * PI * x) * np.sin(m * PI * y)

        return result


analytical = AnalyticalSolutionFull()


# ── PINN architecture ─────────────────────────────────────────────────────────
class PINN(nn.Module):
    """
    Input: [x, y, sin(w1*t), cos(w1*t), sin(w2*t), cos(w2*t), t/T_MAX] — 7 features.

    Two Fourier pairs cover the two most significant oscillation frequencies:
      w1 = pi*sqrt(2) for mode (1,1)
      w2 = pi*sqrt(5) for mode (2,1)
    The network then represents the slowly-varying envelopes, avoiding spectral bias.
    """

    def __init__(self, hidden=128, n_layers=6, t_max=T_MAX):
        super().__init__()
        self.t_max = t_max
        self.omega1 = OMEGA1
        self.omega2 = OMEGA2

        in_dim = 7
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

        self.log_kappa = nn.Parameter(torch.tensor(math.log(1e-3), dtype=torch.float32))

    def forward(self, x, y, t):
        s1 = torch.sin(self.omega1 * t)
        c1 = torch.cos(self.omega1 * t)
        s2 = torch.sin(self.omega2 * t)
        c2 = torch.cos(self.omega2 * t)
        tau = t / self.t_max
        inp = torch.cat([x, y, s1, c1, s2, c2, tau], dim=1)
        return self.net(inp)

    def get_kappa(self):
        return torch.exp(self.log_kappa)


# ── Collocation point samplers ────────────────────────────────────────────────
def sample_pde(n, t_max=T_MAX):
    x = torch.rand(n, 1, requires_grad=True, device=device)
    y = torch.rand(n, 1, requires_grad=True, device=device)
    t = torch.rand(n, 1, requires_grad=True, device=device) * t_max
    return x, y, t


def sample_ic(n):
    x = torch.rand(n, 1, requires_grad=True, device=device)
    y = torch.rand(n, 1, requires_grad=True, device=device)
    t = torch.zeros(n, 1, requires_grad=True, device=device)
    return x, y, t


def sample_bc(n, t_max=T_MAX):
    k = n // 4
    dev = device

    def _rand_t():
        return torch.rand(k, 1, requires_grad=True, device=dev) * t_max

    x0 = torch.zeros(k, 1, requires_grad=True, device=dev)
    x1 = torch.ones(k, 1, requires_grad=True, device=dev)
    y0 = torch.zeros(k, 1, requires_grad=True, device=dev)
    y1 = torch.ones(k, 1, requires_grad=True, device=dev)

    xL = x0; yL = torch.rand(k, 1, requires_grad=True, device=dev); tL = _rand_t()
    xR = x1; yR = torch.rand(k, 1, requires_grad=True, device=dev); tR = _rand_t()
    xB = torch.rand(k, 1, requires_grad=True, device=dev); yB = y0; tB = _rand_t()
    xT = torch.rand(k, 1, requires_grad=True, device=dev); yT = y1; tT = _rand_t()

    return (torch.cat([xL, xR, xB, xT]),
            torch.cat([yL, yR, yB, yT]),
            torch.cat([tL, tR, tB, tT]))


# ── PDE residual ──────────────────────────────────────────────────────────────
def _grad(u, v, **kw):
    return autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                         create_graph=True, retain_graph=True, **kw)[0]


def pde_residual(model, x, y, t):
    """GN-III residual: u_tt - kappa*(u_xxt + u_yyt) - (u_xx + u_yy) - 2x."""
    u = model(x, y, t)
    u_x = _grad(u, x)
    u_y = _grad(u, y)
    u_t = _grad(u, t)
    u_xx = _grad(u_x, x)
    u_yy = _grad(u_y, y)
    u_tt = _grad(u_t, t)
    u_xxt = _grad(u_xx, t)
    u_yyt = _grad(u_yy, t)
    kappa = model.get_kappa()
    return u_tt - kappa * (u_xxt + u_yyt) - (u_xx + u_yy) - 2.0 * x


# ── Sensor data ───────────────────────────────────────────────────────────────
SENSOR_XY = [(0.25, 0.25), (0.25, 0.75), (0.50, 0.50), (0.75, 0.25), (0.75, 0.75)]
N_TIME = 200
NOISE_LEVEL = 1e-5

t_sensor_np = np.linspace(0.0, T_MAX, N_TIME)

xs_list, ys_list, ts_list, data_list = [], [], [], []
for (sx, sy) in SENSOR_XY:
    vals = analytical(np.full(N_TIME, sx), np.full(N_TIME, sy), t_sensor_np)
    xs_list.append(np.full(N_TIME, sx))
    ys_list.append(np.full(N_TIME, sy))
    ts_list.append(t_sensor_np)
    data_list.append(vals)

xs_np = np.concatenate(xs_list).reshape(-1, 1)
ys_np = np.concatenate(ys_list).reshape(-1, 1)
ts_np = np.concatenate(ts_list).reshape(-1, 1)
data_np = np.concatenate(data_list).reshape(-1, 1)

x_data = torch.tensor(xs_np, dtype=torch.float32, device=device)
y_data = torch.tensor(ys_np, dtype=torch.float32, device=device)
t_data = torch.tensor(ts_np, dtype=torch.float32, device=device)
data_true_t = torch.tensor(data_np, dtype=torch.float32, device=device)
data_noisy = data_true_t + NOISE_LEVEL * torch.randn_like(data_true_t) * data_true_t.abs()


# ── Loss helpers ──────────────────────────────────────────────────────────────
mse = nn.MSELoss()

W_PDE  = 1.0
W_IC1  = 200.0
W_IC2  = 200.0
W_BC   = 200.0
W_DATA = 2000.0


def compute_loss(model):
    # PDE residual should be 0 (source 2x already subtracted inside pde_residual)
    xp, yp, tp = sample_pde(10000)
    res = pde_residual(model, xp, yp, tp)
    l_pde = res.pow(2).mean()

    # IC value: u(x,y,0) = sin(pi*x)*sin(pi*y)
    xi, yi, ti = sample_ic(3000)
    u_ic = model(xi, yi, ti)
    ic_ref = torch.sin(PI * xi) * torch.sin(PI * yi)
    l_ic1 = mse(u_ic, ic_ref)

    # IC velocity: u_t(x,y,0) = 0
    l_ic2 = _grad(u_ic, ti).pow(2).mean()

    # BC: u = 0 on all four edges
    xb, yb, tb = sample_bc(3000)
    l_bc = model(xb, yb, tb).pow(2).mean()

    # Data
    l_data = mse(model(x_data, y_data, t_data), data_noisy)

    total = (W_PDE * l_pde + W_IC1 * l_ic1 + W_IC2 * l_ic2
             + W_BC * l_bc + W_DATA * l_data)
    return total, l_pde, l_ic1, l_ic2, l_bc, l_data


# ── Training ──────────────────────────────────────────────────────────────────
model = PINN(hidden=128, n_layers=6, t_max=T_MAX).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"kappa init: {model.get_kappa().item():.4e}  (true: {KAPPA_TRUE:.4e})")

loss_history = []
kappa_history = []
best_loss = float('inf')

print("\n=== Phase 1: Adam (20000 epochs) ===")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1000, min_lr=1e-6
)

ADAM_EPOCHS = 20000
for epoch in range(ADAM_EPOCHS):
    optimizer.zero_grad()
    total, l_pde, l_ic1, l_ic2, l_bc, l_data = compute_loss(model)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(total)

    loss_history.append(total.item())
    kappa_history.append(model.get_kappa().item())

    if total.item() < best_loss:
        best_loss = total.item()
        torch.save(model.state_dict(), 'pinn_inhomogeneous_v2_best.pth')

    if epoch % 500 == 0:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:5d}: loss={total.item():.4e}  "
              f"pde={l_pde.item():.3e}  ic={l_ic1.item():.3e}  "
              f"data={l_data.item():.3e}  "
              f"kappa={model.get_kappa().item():.4e}  lr={lr_now:.2e}")

print("\n=== Phase 2: L-BFGS (300 steps) ===")
model.load_state_dict(torch.load('pinn_inhomogeneous_v2_best.pth'))

fixed_pde = sample_pde(10000)
fixed_ic  = sample_ic(3000)
fixed_bc  = sample_bc(3000)


def _detach_regrad(tensors):
    return tuple(t.detach().requires_grad_(True) for t in tensors)


lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=50,
                    history_size=100, line_search_fn='strong_wolfe')


def closure():
    lbfgs.zero_grad()
    fp = _detach_regrad(fixed_pde)
    fi = _detach_regrad(fixed_ic)
    fb = _detach_regrad(fixed_bc)

    l_pde = pde_residual(model, *fp).pow(2).mean()

    u_ic = model(*fi)
    l_ic1 = mse(u_ic, torch.sin(PI * fi[0]) * torch.sin(PI * fi[1]))
    l_ic2 = _grad(u_ic, fi[2]).pow(2).mean()

    l_bc = model(*fb).pow(2).mean()
    l_data = mse(model(x_data, y_data, t_data), data_noisy)

    loss = (W_PDE * l_pde + W_IC1 * l_ic1 + W_IC2 * l_ic2
            + W_BC * l_bc + W_DATA * l_data)
    loss.backward()
    return loss


for step in range(300):
    loss = lbfgs.step(closure)
    loss_history.append(loss.item())
    kappa_history.append(model.get_kappa().item())
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), 'pinn_inhomogeneous_v2_best.pth')
    if step % 50 == 0:
        print(f"L-BFGS {step:3d}: loss={loss.item():.4e}  "
              f"kappa={model.get_kappa().item():.4e}")

model.load_state_dict(torch.load('pinn_inhomogeneous_v2_best.pth'))

# ── Results ───────────────────────────────────────────────────────────────────
kappa_found = model.get_kappa().item()
rel_err_kappa = abs(kappa_found - KAPPA_TRUE) / KAPPA_TRUE * 100
print(f"\n{'='*55}")
print(f"True  kappa = {KAPPA_TRUE:.4e}")
print(f"Found kappa = {kappa_found:.4e}")
print(f"Relative error = {rel_err_kappa:.2f}%")
print(f"{'='*55}")

# ── Validation metrics ────────────────────────────────────────────────────────
model.eval()
n_test = 2000
with torch.no_grad():
    x_test = torch.rand(n_test, 1, device=device)
    y_test = torch.rand(n_test, 1, device=device)
    t_test = torch.rand(n_test, 1, device=device) * T_MAX
    u_pred = model(x_test, y_test, t_test)

u_exact = torch.tensor(
    analytical(x_test.cpu(), y_test.cpu(), t_test.cpu()).reshape(-1, 1),
    dtype=torch.float32
)
l2_rel = (torch.norm(u_pred.cpu() - u_exact) / torch.norm(u_exact)).item()
linf   = torch.max(torch.abs(u_pred.cpu() - u_exact)).item()
print(f"L2 relative error (field): {l2_rel:.3e}")
print(f"L-inf error (field):       {linf:.3e}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
ax.semilogy(loss_history, 'b-', lw=0.7, label='Total loss')
ax.axvline(ADAM_EPOCHS, color='gray', ls='--', lw=0.8, label='Adam→L-BFGS')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training loss'); ax.legend(); ax.grid(True, which='both', alpha=0.3)

ax2 = axes[0].twinx()
ax2.semilogy(kappa_history, 'r-', lw=0.8, alpha=0.7, label='κ')
ax2.axhline(KAPPA_TRUE, color='darkred', ls=':', lw=1.2, label=f'κ_true={KAPPA_TRUE:.1e}')
ax2.set_ylabel('κ (log scale)', color='r')
ax2.tick_params(axis='y', colors='r')
ax2.legend(loc='lower right')

# Compare u(0.5, 0.5, t) over time
ax = axes[1]
t_plot = np.linspace(0, T_MAX, 1000)
u_exact_center = analytical(np.full(1000, 0.5), np.full(1000, 0.5), t_plot)
x05 = torch.full((1000, 1), 0.5, device=device)
y05 = torch.full((1000, 1), 0.5, device=device)
t05 = torch.tensor(t_plot.reshape(-1, 1), dtype=torch.float32, device=device)
with torch.no_grad():
    u_pinn_center = model(x05, y05, t05).cpu().numpy().flatten()

ax.plot(t_plot, u_exact_center, 'b-', lw=1.5, label='Analytical u(0.5,0.5,t)')
ax.plot(t_plot, u_pinn_center,  'r--', lw=1.2, label='PINN u(0.5,0.5,t)')
ax.set_xlabel('t'); ax.set_ylabel('u(0.5, 0.5, t)')
ax.set_title(f'κ_found={kappa_found:.3e}  (err={rel_err_kappa:.1f}%)')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pinn_inhomogeneous_v2_results.png', dpi=150)
plt.show()

# Spatial snapshots: PINN vs Exact
t_snaps = [1.0, 10.0, 50.0, 100.0, 250.0, 500.0]
n_sp = 50
x_sp = torch.linspace(0, 1, n_sp, device=device)
y_sp = torch.linspace(0, 1, n_sp, device=device)
X, Y = torch.meshgrid(x_sp, y_sp, indexing='ij')
Xf = X.reshape(-1, 1); Yf = Y.reshape(-1, 1)

fig2, axes2 = plt.subplots(2, len(t_snaps), figsize=(18, 6))
for i, t_val in enumerate(t_snaps):
    Tf = torch.full_like(Xf, t_val)
    with torch.no_grad():
        u_p = model(Xf, Yf, Tf).reshape(n_sp, n_sp).cpu().numpy()
    u_a = analytical(Xf.cpu(), Yf.cpu(), Tf.cpu()).reshape(n_sp, n_sp)
    vmin = min(u_p.min(), u_a.min())
    vmax = max(u_p.max(), u_a.max())

    axes2[0, i].contourf(X.cpu(), Y.cpu(), u_p, 30, cmap='jet', vmin=vmin, vmax=vmax)
    axes2[0, i].set_title(f'PINN  t={t_val}')
    axes2[1, i].contourf(X.cpu(), Y.cpu(), u_a, 30, cmap='jet', vmin=vmin, vmax=vmax)
    axes2[1, i].set_title(f'Exact  t={t_val}')
    for ax in axes2[:, i]:
        ax.set_xlabel('x'); ax.set_ylabel('y')

plt.suptitle(f'Inhomogeneous q=2xt  |  κ_found={kappa_found:.3e}  '
             f'(true {KAPPA_TRUE:.1e},  err {rel_err_kappa:.1f}%)', fontsize=11)
plt.tight_layout()
plt.savefig('pinn_inhomogeneous_v2_snapshots.png', dpi=150)
plt.show()

print("\nDone. Saved: pinn_inhomogeneous_v2_best.pth, "
      "pinn_inhomogeneous_v2_results.png, pinn_inhomogeneous_v2_snapshots.png")
