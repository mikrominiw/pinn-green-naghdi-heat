"""
PINN for inverse problem: identify kappa in GN-III homogeneous heat equation.

PDE (dimensionless, q=0):
    u_tt - kappa*(u_xxt + u_yyt) - (u_xx + u_yy) = 0
    (x,y) in (0,1)^2,  t in [0, T]

IC:  u(x,y,0) = sin(pi*x)*sin(pi*y),   u_t(x,y,0) = 0
BC:  u = 0  on all four edges

True kappa = 1.2e-4  (duralumin D16T, dimensionless)

Key improvements over existing code:
1. t_max=500: decay is ~45% at t=500, giving clear signal about kappa
2. Fourier time features [sin(w0*t), cos(w0*t)]: network only needs to
   learn the slowly-varying envelope, not 350+ oscillation cycles
3. Joint training of network + kappa from scratch (no stage separation)
4. kappa initialized at 1e-3 (not 1.0), only 2x in log-space from truth
5. High data weight (w_data=2000) drives kappa gradient from sensor data
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
LAM11 = 2 * PI ** 2   # eigenvalue for (n,m)=(1,1): pi^2*(1^2+1^2), known from geometry

# Fourier encoding frequency = undamped natural frequency sqrt(lambda_11).
# This is fully determined by the domain geometry and BCs — no knowledge of
# kappa required.  The damped frequency beta = sqrt(lam11 - (kappa*lam11/2)^2)
# differs from sqrt(lam11) by a relative correction < 1e-7 for kappa=1.2e-4,
# so using sqrt(lam11) introduces no practical error and no data leakage.
OMEGA0 = math.sqrt(LAM11)   # = pi*sqrt(2) ≈ 4.4429


# ── Analytical solution ───────────────────────────────────────────────────────
class AnalyticalSolution:
    """
    Exact solution for q=0 (homogeneous GN-III), single mode (n=m=1):

        theta(x,y,t) = A(t) * sin(pi*x) * sin(pi*y)

        A(t) = exp(-alpha*t) * [cos(beta*t) + (alpha/beta)*sin(beta*t)]

    This is the only surviving mode given IC u(x,y,0)=sin(pi*x)sin(pi*y).
    """

    def __init__(self, kappa=KAPPA_TRUE):
        lam = LAM11
        disc = (kappa * lam) ** 2 - 4 * lam
        assert disc < 0, "Expected complex roots for small kappa"
        self.alpha = kappa * lam / 2
        self.beta = math.sqrt(-disc) / 2

    def amplitude(self, t):
        """A(t): temporal coefficient (numpy array or scalar)."""
        return np.exp(-self.alpha * t) * (
            np.cos(self.beta * t) + (self.alpha / self.beta) * np.sin(self.beta * t)
        )

    def __call__(self, x, y, t):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy().flatten()
            y = y.detach().cpu().numpy().flatten()
            t = t.detach().cpu().numpy().flatten()
        return self.amplitude(np.asarray(t)) * np.sin(PI * np.asarray(x)) * np.sin(PI * np.asarray(y))


analytical = AnalyticalSolution()


# ── PINN architecture ─────────────────────────────────────────────────────────
class PINN(nn.Module):
    """
    Input: [x, y, sin(omega0*t), cos(omega0*t), t/T_MAX]  —  5 features.

    The two Fourier features handle the fast oscillations (~354 cycles over
    T_MAX=500). The network only needs to learn the slowly-varying envelope
    exp(-alpha*t) and the small sin-term correction. This circumvents the
    spectral bias of standard MLPs against high-frequency signals.

    kappa is stored as log_kappa (unconstrained) and recovered via exp().
    """

    def __init__(self, hidden=128, n_layers=6, t_max=T_MAX):
        super().__init__()
        self.t_max = t_max
        self.omega0 = OMEGA0

        in_dim = 5
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

        # Xavier initialisation for tanh
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # kappa initialised at 1e-3 — only ~2 units in log-space from truth
        self.log_kappa = nn.Parameter(torch.tensor(math.log(1e-3), dtype=torch.float32))

    def forward(self, x, y, t):
        s = torch.sin(self.omega0 * t)
        c = torch.cos(self.omega0 * t)
        tau = t / self.t_max
        inp = torch.cat([x, y, s, c, tau], dim=1)
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

    # x=0: (0, rand_y, rand_t)
    xL = x0; yL = torch.rand(k, 1, requires_grad=True, device=dev); tL = _rand_t()
    # x=1: (1, rand_y, rand_t)
    xR = x1; yR = torch.rand(k, 1, requires_grad=True, device=dev); tR = _rand_t()
    # y=0: (rand_x, 0, rand_t)
    xB = torch.rand(k, 1, requires_grad=True, device=dev); yB = y0; tB = _rand_t()
    # y=1: (rand_x, 1, rand_t)
    xT = torch.rand(k, 1, requires_grad=True, device=dev); yT = y1; tT = _rand_t()

    x = torch.cat([xL, xR, xB, xT])
    y = torch.cat([yL, yR, yB, yT])
    t = torch.cat([tL, tR, tB, tT])
    return x, y, t


# ── PDE residual ──────────────────────────────────────────────────────────────
def _grad(u, v, **kw):
    return autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                         create_graph=True, retain_graph=True, **kw)[0]


def pde_residual(model, x, y, t):
    """Returns GN-III residual: u_tt - kappa*(u_xxt + u_yyt) - (u_xx + u_yy)."""
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
    return u_tt - kappa * (u_xxt + u_yyt) - (u_xx + u_yy)


# ── Sensor data ───────────────────────────────────────────────────────────────
SENSOR_XY = [(0.25, 0.25), (0.25, 0.75), (0.50, 0.50), (0.75, 0.25), (0.75, 0.75)]
N_TIME = 200
NOISE_LEVEL = 1e-3

t_sensor_np = np.linspace(0.0, T_MAX, N_TIME)

xs, ys, ts, data_clean = [], [], [], []
for (sx, sy) in SENSOR_XY:
    vals = analytical(np.full(N_TIME, sx), np.full(N_TIME, sy), t_sensor_np)
    xs.append(np.full(N_TIME, sx))
    ys.append(np.full(N_TIME, sy))
    ts.append(t_sensor_np)
    data_clean.append(vals)

xs = np.concatenate(xs).reshape(-1, 1)
ys = np.concatenate(ys).reshape(-1, 1)
ts = np.concatenate(ts).reshape(-1, 1)
data_clean = np.concatenate(data_clean).reshape(-1, 1)

x_data = torch.tensor(xs, dtype=torch.float32, device=device)
y_data = torch.tensor(ys, dtype=torch.float32, device=device)
t_data = torch.tensor(ts, dtype=torch.float32, device=device)
data_true_t = torch.tensor(data_clean, dtype=torch.float32, device=device)
data_noisy = data_true_t + NOISE_LEVEL * torch.randn_like(data_true_t) * data_true_t.abs()


# ── Loss helpers ──────────────────────────────────────────────────────────────
mse = nn.MSELoss()

W_PDE  = 1.0
W_IC1  = 200.0
W_IC2  = 200.0
W_BC   = 200.0
W_DATA = 2000.0


def compute_loss(model):
    # PDE
    xp, yp, tp = sample_pde(10000)
    res = pde_residual(model, xp, yp, tp)
    l_pde = res.pow(2).mean()

    # IC value: u(x,y,0) = sin(pi*x)*sin(pi*y)
    xi, yi, ti = sample_ic(3000)
    u_ic = model(xi, yi, ti)
    ic_ref = torch.sin(PI * xi) * torch.sin(PI * yi)
    l_ic1 = mse(u_ic, ic_ref)

    # IC velocity: u_t(x,y,0) = 0
    u_t_ic = _grad(u_ic, ti)
    l_ic2 = u_t_ic.pow(2).mean()

    # BC: u = 0 on all four edges
    xb, yb, tb = sample_bc(3000)
    u_bc = model(xb, yb, tb)
    l_bc = u_bc.pow(2).mean()

    # Data
    u_data = model(x_data, y_data, t_data)
    l_data = mse(u_data, data_noisy)

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

# Phase 1: Adam — joint optimisation of network + kappa
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
        torch.save(model.state_dict(), 'pinn_homogeneous_v2_best.pth')

    if epoch % 500 == 0:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:5d}: loss={total.item():.4e}  "
              f"pde={l_pde.item():.3e}  ic={l_ic1.item():.3e}  "
              f"data={l_data.item():.3e}  "
              f"kappa={model.get_kappa().item():.4e}  lr={lr_now:.2e}")

# Phase 2: L-BFGS fine-tuning
print("\n=== Phase 2: L-BFGS (300 steps) ===")
model.load_state_dict(torch.load('pinn_homogeneous_v2_best.pth'))

# Fix collocation points for L-BFGS closure
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

    res = pde_residual(model, *fp)
    l_pde = res.pow(2).mean()

    u_ic = model(*fi)
    l_ic1 = mse(u_ic, torch.sin(PI * fi[0]) * torch.sin(PI * fi[1]))
    l_ic2 = _grad(u_ic, fi[2]).pow(2).mean()

    l_bc = model(*fb).pow(2).mean()

    u_data = model(x_data, y_data, t_data)
    l_data = mse(u_data, data_noisy)

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
        torch.save(model.state_dict(), 'pinn_homogeneous_v2_best.pth')
    if step % 50 == 0:
        print(f"L-BFGS {step:3d}: loss={loss.item():.4e}  "
              f"kappa={model.get_kappa().item():.4e}")

model.load_state_dict(torch.load('pinn_homogeneous_v2_best.pth'))

# ── Results ───────────────────────────────────────────────────────────────────
kappa_found = model.get_kappa().item()
rel_err_kappa = abs(kappa_found - KAPPA_TRUE) / KAPPA_TRUE * 100
print(f"\n{'='*55}")
print(f"True  kappa = {KAPPA_TRUE:.4e}")
print(f"Found kappa = {kappa_found:.4e}")
print(f"Relative error = {rel_err_kappa:.2f}%")
print(f"{'='*55}")

# ── Validation metrics on random test points ──────────────────────────────────
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

# Loss history
ax = axes[0]
epochs_total = len(loss_history)
ax.semilogy(loss_history, 'b-', lw=0.7, label='Total loss')
ax.axvline(ADAM_EPOCHS, color='gray', ls='--', lw=0.8, label='Adam→L-BFGS')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training loss')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# kappa history
ax2 = axes[0].twinx()
ax2.semilogy(kappa_history, 'r-', lw=0.8, alpha=0.7, label='κ')
ax2.axhline(KAPPA_TRUE, color='darkred', ls=':', lw=1.2, label=f'κ_true={KAPPA_TRUE:.1e}')
ax2.set_ylabel('κ (log scale)', color='r')
ax2.tick_params(axis='y', colors='r')
ax2.legend(loc='lower right')

# Temporal amplitude profile A(t) at (x,y)=(0.5,0.5)
ax = axes[1]
t_plot = np.linspace(0, T_MAX, 1000)
A_exact = analytical.amplitude(t_plot)
x05 = torch.full((1000, 1), 0.5, device=device)
y05 = torch.full((1000, 1), 0.5, device=device)
t05 = torch.tensor(t_plot.reshape(-1, 1), dtype=torch.float32, device=device)
with torch.no_grad():
    u_center = model(x05, y05, t05).cpu().numpy().flatten()
sin_sq = math.sin(PI * 0.5) ** 2   # sin(pi*0.5)^2 = 1
A_pinn = u_center / sin_sq

ax.plot(t_plot, A_exact, 'b-', lw=1.5, label='Analytical A(t)')
ax.plot(t_plot, A_pinn,  'r--', lw=1.2, label='PINN A(t)')
ax.set_xlabel('t')
ax.set_ylabel('A(t)')
ax.set_title(f'Temporal amplitude   κ_found={kappa_found:.3e}  (err={rel_err_kappa:.1f}%)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pinn_homogeneous_v2_results.png', dpi=150)
plt.show()

# Spatial comparison at several time snapshots
t_snaps = [1.0, 10.0, 50.0, 100.0, 250.0, 500.0]
n_sp = 50
x_sp = torch.linspace(0, 1, n_sp, device=device)
y_sp = torch.linspace(0, 1, n_sp, device=device)
X, Y = torch.meshgrid(x_sp, y_sp, indexing='ij')
Xf = X.reshape(-1, 1)
Yf = Y.reshape(-1, 1)

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

plt.suptitle(f'κ_found={kappa_found:.3e}  (true {KAPPA_TRUE:.1e},  err {rel_err_kappa:.1f}%)',
             fontsize=12)
plt.tight_layout()
plt.savefig('pinn_homogeneous_v2_snapshots.png', dpi=150)
plt.show()

print("\nDone. Saved: pinn_homogeneous_v2_best.pth, pinn_homogeneous_v2_results.png, "
      "pinn_homogeneous_v2_snapshots.png")
