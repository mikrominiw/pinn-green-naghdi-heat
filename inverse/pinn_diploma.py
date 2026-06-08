"""
PINN for GN-III inverse problem -- full diploma run.

Trains PINN to identify kappa in
    u_tt - kappa(u_xxt + u_yyt) - (u_xx + u_yy) = dq/dt
on (0,1)^2 with two source variants:
    homo   q = 0
    inhomo q = 2 x t
over 5 random seeds each (main study), plus ablations of kappa_init
and noise level (single seed each).

Hard IC/BC via ansatz; only PDE + data losses are penalised.

Outputs in results_diploma/<RUN_ID>/{main,ablation_kappa,ablation_noise}/
together with metrics.json, stats_report.txt and a large set of PNG figures
intended to be embedded individually in the thesis.
"""
from __future__ import annotations

import json
import math
import time
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import autograd, optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  -- needed for projection='3d'
from scipy import stats

# -- Output dir + sub-folders ------------------------------------------------
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT_DIR = Path(__file__).parent.parent / "results_diploma" / RUN_ID
DIR_MAIN = ROOT_DIR / "main"
DIR_ABL_KAPPA = ROOT_DIR / "ablation_kappa"
DIR_ABL_NOISE = ROOT_DIR / "ablation_noise"
for d in (DIR_MAIN, DIR_ABL_KAPPA, DIR_ABL_NOISE):
    d.mkdir(parents=True, exist_ok=True)

# -- Logging ------------------------------------------------------------------
logger = logging.getLogger("pinn_diploma")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
_fh = logging.FileHandler(ROOT_DIR / "log.txt", mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_sh)


def log(msg: str = "") -> None:
    logger.info(msg)


# -- Device -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

log(f"Run ID: {RUN_ID}")
log(f"Root dir: {ROOT_DIR}")
log(f"Device: {device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")


# -- Constants ----------------------------------------------------------------
PI = math.pi
KAPPA_TRUE = 1.2e-4
T_MAX = 500.0
LAM11 = 2 * PI ** 2
OMEGA0 = math.sqrt(LAM11)

SENSOR_XY = [(0.25, 0.25), (0.25, 0.75), (0.50, 0.50),
             (0.75, 0.25), (0.75, 0.75)]
N_TIME = 200

# Main study parameters
SEEDS = [1, 7, 13, 42, 123]
NOISE_DEFAULT = 1e-2
KAPPA_INIT_DEFAULT = 1e-3

# Optimisation
FLAT_EPOCHS = 15000
LBFGS_STEPS = 300
N_PDE = 10000
LOG_EVERY = 500

W_PDE = 1.0
W_DATA = 2000.0

# Ablations: 1 seed per point
ABL_KAPPA_VALUES = [1e-5, 1e-4, 1e-3, 1e-2]
ABL_NOISE_VALUES = [1e-2, 5e-2, 1e-1]
ABL_SEED = 1
ABL_KINDS = ("homo", "inhomo")

# Convergence detection
KAPPA_STABILITY_WINDOW = 500
KAPPA_STABILITY_THRESHOLD = 1e-7  # std threshold on log_kappa

# -- Analytical reference -----------------------------------------------------
class AnalyticalSolution:
    """
    Fourier series solution on (0,1)^2 with zero Dirichlet BC,
    u(0)=sin(pi x)sin(pi y), u_t(0)=0.

    Source: q=0 (homo) or q=2xt (inhomo, dq/dt=2x).
    """

    def __init__(self, kappa: float = KAPPA_TRUE, kind: str = "homo",
                 n_max: int = 30, m_max: int = 30) -> None:
        self.kappa = kappa
        self.kind = kind
        self.n_max = n_max
        self.m_max = m_max

    @staticmethod
    def _to_np(*arrs):
        out = []
        for a in arrs:
            if torch.is_tensor(a):
                a = a.detach().cpu().numpy()
            out.append(np.asarray(a, dtype=float))
        return tuple(out)

    def f_nm(self, n: int, m: int) -> float:
        if self.kind == "homo":
            return 0.0
        int_x = (-1) ** (n + 1) / (n * PI)
        int_y = (1 - (-1) ** m) / (m * PI)
        return 8.0 * int_x * int_y

    def __call__(self, x, y, t):
        x, y, t = self._to_np(x, y, t)
        shape = x.shape
        x = x.flatten()
        y = y.flatten()
        t = t.flatten()
        out = np.zeros_like(x)
        for n in range(1, self.n_max + 1):
            for m in range(1, self.m_max + 1):
                lam = PI ** 2 * (n ** 2 + m ** 2)
                A = 1.0 if (n == 1 and m == 1) else 0.0
                f = self.f_nm(n, m)
                if abs(A) < 1e-12 and abs(f) < 1e-12:
                    continue
                disc = (self.kappa * lam) ** 2 - 4 * lam
                if disc < 0:
                    real = -self.kappa * lam / 2
                    imag = math.sqrt(-disc) / 2
                    D1 = A - f / lam
                    D2 = -real * D1 / imag
                    T_t = (np.exp(real * t) *
                           (D1 * np.cos(imag * t) + D2 * np.sin(imag * t))
                           + f / lam)
                else:
                    r1 = (-self.kappa * lam + math.sqrt(disc)) / 2
                    r2 = (-self.kappa * lam - math.sqrt(disc)) / 2
                    diff = r1 - r2
                    if abs(diff) > 1e-12:
                        C1 = -r2 * (A - f / lam) / diff
                        C2 = r1 * (A - f / lam) / diff
                    else:
                        C1 = C2 = (A - f / lam) / 2
                    T_t = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t) + f / lam
                out = out + T_t * np.sin(n * PI * x) * np.sin(m * PI * y)
        return out.reshape(shape)


# -- Particular solution for inhomogeneous case -----------------------------
class ParticularSolution:
    """
    Torch tensor u_p(x,y,t) satisfying u_p(t=0)=0, u_p_t(t=0)=0,
    u_p|dOmega=0 and asymptoting to the steady-state Fourier series for q=2xt.

    For homo: returns 0.
    """

    def __init__(self, kind: str, n_max: int = 12, m_max: int = 12,
                 eta: float = 0.01) -> None:
        self.kind = kind
        self.eta = eta
        coeffs = []
        if kind == "inhomo":
            for n in range(1, n_max + 1):
                for m in range(1, m_max + 1):
                    int_x = (-1) ** (n + 1) / (n * PI)
                    int_y = (1 - (-1) ** m) / (m * PI)
                    f = 8.0 * int_x * int_y
                    if abs(f) < 1e-12:
                        continue
                    lam = PI ** 2 * (n ** 2 + m ** 2)
                    coeffs.append((n, m, f / lam))
        self.coeffs = coeffs

    def __call__(self, x: torch.Tensor, y: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        if not self.coeffs:
            return torch.zeros_like(x)
        eta = self.eta
        transient = 1.0 - torch.exp(-eta * t) * (1.0 + eta * t)
        out = torch.zeros_like(x)
        for n, m, amp in self.coeffs:
            out = out + amp * torch.sin(n * PI * x) * torch.sin(m * PI * y)
        return transient * out


# -- PINN with hard IC/BC ansatz ----------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        for m in (self.lin1, self.lin2):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.lin1(x))
        h = self.lin2(h)
        return torch.tanh(x + h)


class PINN(nn.Module):
    def __init__(self, kappa_init: float, particular: ParticularSolution,
                 hidden: int = 64, n_blocks: int = 3) -> None:
        super().__init__()
        self.particular = particular
        self.input_layer = nn.Linear(5, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(hidden, 1)
        for m in (self.input_layer, self.output_layer):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        self.log_kappa = nn.Parameter(
            torch.tensor(math.log(kappa_init), dtype=torch.float32))

    def forward(self, x, y, t):
        inp = torch.cat([x, y,
                         torch.sin(OMEGA0 * t), torch.cos(OMEGA0 * t),
                         t / T_MAX], dim=1)
        h = torch.tanh(self.input_layer(inp))
        for b in self.blocks:
            h = b(h)
        N = self.output_layer(h)
        env = torch.sin(PI * x) * torch.sin(PI * y)
        tn = t / T_MAX
        u_h = env * (torch.cos(OMEGA0 * t) + tn * tn * N)
        return u_h + self.particular(x, y, t)

    def get_kappa(self) -> torch.Tensor:
        return torch.exp(self.log_kappa)


# -- Autograd helpers ---------------------------------------------------------
def _g(u, v):
    return autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                         create_graph=True, retain_graph=True)[0]


def pde_residual(model: PINN, x, y, t, kind: str):
    u = model(x, y, t)
    ux, uy, ut = autograd.grad(u, [x, y, t],
                               grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)
    uxx = _g(ux, x)
    uyy = _g(uy, y)
    utt = _g(ut, t)
    uxxt = _g(uxx, t)
    uyyt = _g(uyy, t)
    res = utt - model.get_kappa() * (uxxt + uyyt) - (uxx + uyy)
    if kind == "inhomo":
        res = res - 2.0 * x
    return res


def sample_pde(n: int, t_max: float):
    x = torch.rand(n, 1, requires_grad=True, device=device)
    y = torch.rand(n, 1, requires_grad=True, device=device)
    t = torch.rand(n, 1, requires_grad=True, device=device) * t_max
    return x, y, t


mse = nn.MSELoss()


def make_sensor_data(kind: str, seed: int, noise_level: float):
    rng = np.random.default_rng(seed)
    analytical = AnalyticalSolution(kappa=KAPPA_TRUE, kind=kind)
    t_sensor = np.linspace(0.0, T_MAX, N_TIME)
    xs, ys, ts, ds = [], [], [], []
    for sx, sy in SENSOR_XY:
        vals = analytical(np.full(N_TIME, sx), np.full(N_TIME, sy), t_sensor)
        xs.append(np.full(N_TIME, sx))
        ys.append(np.full(N_TIME, sy))
        ts.append(t_sensor)
        ds.append(vals)
    xs = np.concatenate(xs).reshape(-1, 1)
    ys = np.concatenate(ys).reshape(-1, 1)
    ts = np.concatenate(ts).reshape(-1, 1)
    ds = np.concatenate(ds).reshape(-1, 1)
    ds_n = ds + rng.standard_normal(ds.shape) * noise_level * np.abs(ds)
    return (
        torch.tensor(xs, dtype=torch.float32, device=device),
        torch.tensor(ys, dtype=torch.float32, device=device),
        torch.tensor(ts, dtype=torch.float32, device=device),
        torch.tensor(ds_n, dtype=torch.float32, device=device),
    )


# -- Result containers --------------------------------------------------------
@dataclass
class RunResult:
    tag: str             # e.g. "main_homo_seed1" or "abl_kappa_1e-3"
    kind: str
    seed: int
    kappa_init: float
    noise_level: float
    kappa_final: float
    rel_err_pct: float
    l2_field_err: float
    loss_final: float
    elapsed_s: float
    # histories
    loss_adam: list[float] = field(default_factory=list)
    loss_lbfgs: list[float] = field(default_factory=list)
    kappa_adam: list[float] = field(default_factory=list)
    kappa_lbfgs: list[float] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)
    pde_resid_after: list[float] = field(default_factory=list)
    # convergence stats
    convergence_epoch: int = -1
    log_loss_slope: float = float("nan")
    log_loss_intercept: float = float("nan")
    # phase-final kappa values (Adam end, LBFGS end)
    kappa_after_adam: float = float("nan")
    kappa_after_lbfgs: float = float("nan")


def field_l2_error(model: PINN, kind: str, n: int = 4000) -> float:
    analytical = AnalyticalSolution(kappa=KAPPA_TRUE, kind=kind)
    x = torch.rand(n, 1, device=device)
    y = torch.rand(n, 1, device=device)
    t = torch.rand(n, 1, device=device) * T_MAX
    with torch.no_grad():
        u_pred = model(x, y, t).cpu().numpy().flatten()
    u_true = analytical(x, y, t).flatten()
    num = np.linalg.norm(u_pred - u_true)
    den = np.linalg.norm(u_true) + 1e-12
    return float(num / den)


def collect_pde_residuals(model: PINN, kind: str, n: int = 4000) -> np.ndarray:
    """|PDE residual| on a random sample (with grad enabled)."""
    x = torch.rand(n, 1, requires_grad=True, device=device)
    y = torch.rand(n, 1, requires_grad=True, device=device)
    t = torch.rand(n, 1, requires_grad=True, device=device) * T_MAX
    res = pde_residual(model, x, y, t, kind).detach().abs().cpu().numpy().flatten()
    return res


def detect_convergence(kappa_history: list[float]) -> int:
    """First epoch at which std of log(kappa) over the trailing window
    drops below threshold; -1 if never."""
    if len(kappa_history) < KAPPA_STABILITY_WINDOW:
        return -1
    arr = np.log(np.asarray(kappa_history) + 1e-20)
    for i in range(KAPPA_STABILITY_WINDOW, len(arr)):
        if arr[i - KAPPA_STABILITY_WINDOW:i].std() < KAPPA_STABILITY_THRESHOLD:
            return i
    return -1


def fit_log_loss_slope(loss_history: list[float],
                       skip_front: int = 200) -> tuple[float, float]:
    """Fit log(loss) ~ a * log(epoch) + b, return (a, b).  Skips the warmup."""
    if len(loss_history) <= skip_front + 100:
        return float("nan"), float("nan")
    eps = np.arange(skip_front, len(loss_history), dtype=float)
    losses = np.asarray(loss_history[skip_front:], dtype=float)
    losses = np.maximum(losses, 1e-20)
    a, b = np.polyfit(np.log(eps), np.log(losses), 1)
    return float(a), float(b)


# -- Training -----------------------------------------------------------------
def train_one(tag: str, kind: str, seed: int, kappa_init: float,
              noise_level: float) -> tuple[RunResult, PINN]:
    log("")
    log("=" * 72)
    log(f"Run: {tag}  kind={kind}  seed={seed}  "
        f"kappa_init={kappa_init:.1e}  noise={noise_level:.1e}")
    log("=" * 72)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x_data, y_data, t_data, data_noisy = make_sensor_data(kind, seed, noise_level)
    particular = ParticularSolution(kind=kind)
    model = PINN(kappa_init=kappa_init, particular=particular).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"params={n_params:,}  sensors={len(SENSOR_XY)}  data_points={x_data.numel()}")

    LR_NET_BASE = 1e-3
    LR_KAPPA_BASE = 1e-2
    net_params = [p for n, p in model.named_parameters() if n != "log_kappa"]
    kappa_param = [model.log_kappa]
    opt = optim.Adam([
        {"params": net_params, "lr": LR_NET_BASE, "name": "net"},
        {"params": kappa_param, "lr": LR_KAPPA_BASE, "name": "kappa"},
    ])
    sch = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=500, threshold=0.01, min_lr=1e-6)

    loss_adam: list[float] = []
    kappa_adam: list[float] = []
    lr_hist: list[float] = []
    t_start = time.time()

    def compute_loss(t_max: float):
        xp, yp, tp = sample_pde(N_PDE, t_max)
        l_pde = pde_residual(model, xp, yp, tp, kind).pow(2).mean()
        l_data = mse(model(x_data, y_data, t_data), data_noisy)
        return W_PDE * l_pde + W_DATA * l_data, l_pde, l_data

    log(f"=== Adam ({FLAT_EPOCHS} epochs, T={T_MAX}) ===")
    for ep in range(FLAT_EPOCHS):
        opt.zero_grad()
        L, l_pde, l_data = compute_loss(T_MAX)
        L.backward()
        opt.step()
        sch.step(L.item())
        loss_adam.append(L.item())
        kappa_adam.append(model.get_kappa().item())
        lrn = next(g["lr"] for g in opt.param_groups if g["name"] == "net")
        lr_hist.append(lrn)
        if ep % LOG_EVERY == 0:
            lrk = next(g["lr"] for g in opt.param_groups if g["name"] == "kappa")
            k = model.get_kappa().item()
            err = abs(k - KAPPA_TRUE) / KAPPA_TRUE * 100
            log(f"   ep={ep:5d}  loss={L.item():.3e}  "
                f"pde={l_pde.item():.2e}  data={l_data.item():.2e}  "
                f"kappa={k:.4e}  err={err:5.1f}%  "
                f"lr_net={lrn:.2e}  lr_kap={lrk:.2e}")
    kappa_after_adam = model.get_kappa().item()

    log(f"=== L-BFGS ({LBFGS_STEPS} steps) ===")
    fp = sample_pde(N_PDE, T_MAX)
    lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=50,
                        history_size=100, line_search_fn="strong_wolfe")
    loss_lbfgs: list[float] = []
    kappa_lbfgs: list[float] = []

    def closure():
        lbfgs.zero_grad()
        xp = fp[0].detach().requires_grad_(True)
        yp = fp[1].detach().requires_grad_(True)
        tp = fp[2].detach().requires_grad_(True)
        l_pde = pde_residual(model, xp, yp, tp, kind).pow(2).mean()
        l_data = mse(model(x_data, y_data, t_data), data_noisy)
        L = W_PDE * l_pde + W_DATA * l_data
        L.backward()
        return L

    for step in range(LBFGS_STEPS):
        try:
            L = lbfgs.step(closure)
        except Exception as e:
            log(f"   L-BFGS step {step}: {type(e).__name__} -- stop")
            break
        if L is None or not math.isfinite(L.item()):
            log(f"   L-BFGS step {step}: non-finite -- stop")
            break
        loss_lbfgs.append(L.item())
        kappa_lbfgs.append(model.get_kappa().item())
        if step % 50 == 0:
            k = model.get_kappa().item()
            err = abs(k - KAPPA_TRUE) / KAPPA_TRUE * 100
            log(f"   step={step:4d}  loss={L.item():.3e}  "
                f"kappa={k:.4e}  err={err:5.1f}%")

    kappa_after_lbfgs = model.get_kappa().item()
    rel_err = abs(kappa_after_lbfgs - KAPPA_TRUE) / KAPPA_TRUE * 100
    l2_err = field_l2_error(model, kind)
    pde_res = collect_pde_residuals(model, kind)
    elapsed = time.time() - t_start

    a_slope, a_int = fit_log_loss_slope(loss_adam)
    conv_ep = detect_convergence(kappa_adam)

    log("")
    log(f"FINAL  {tag}  kappa={kappa_after_lbfgs:.6e}  rel_err={rel_err:.2f}%  "
        f"L2_field={l2_err:.4e}  loss={(loss_lbfgs[-1] if loss_lbfgs else loss_adam[-1]):.3e}  "
        f"elapsed={elapsed:.1f}s  "
        f"log_loss_slope={a_slope:.3f}  conv_epoch={conv_ep}  "
        f"|PDE|_mean={pde_res.mean():.3e}  |PDE|_max={pde_res.max():.3e}")

    result = RunResult(
        tag=tag, kind=kind, seed=seed,
        kappa_init=kappa_init, noise_level=noise_level,
        kappa_final=kappa_after_lbfgs, rel_err_pct=rel_err,
        l2_field_err=l2_err,
        loss_final=loss_lbfgs[-1] if loss_lbfgs else loss_adam[-1],
        elapsed_s=elapsed,
        loss_adam=loss_adam, loss_lbfgs=loss_lbfgs,
        kappa_adam=kappa_adam, kappa_lbfgs=kappa_lbfgs,
        lr_history=lr_hist,
        pde_resid_after=pde_res.tolist(),
        convergence_epoch=conv_ep,
        log_loss_slope=a_slope, log_loss_intercept=a_int,
        kappa_after_adam=kappa_after_adam,
        kappa_after_lbfgs=kappa_after_lbfgs,
    )
    return result, model


# -- Statistics ---------------------------------------------------------------
def bootstrap_ci(x: np.ndarray, n_boot: int = 10000,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=n, replace=True).mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def descriptive(name: str, x: np.ndarray) -> dict:
    lo, hi = bootstrap_ci(x)
    return {
        "name": name, "n": int(len(x)),
        "mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)), "min": float(np.min(x)), "max": float(np.max(x)),
        "ci95_low": lo, "ci95_high": hi,
    }


def run_statistics(results: list[RunResult]) -> dict:
    homo = [r for r in results if r.kind == "homo"]
    inho = [r for r in results if r.kind == "inhomo"]
    if not homo or not inho:
        return {}
    k_h = np.array([r.kappa_final for r in homo])
    k_i = np.array([r.kappa_final for r in inho])
    e_h = np.array([r.rel_err_pct for r in homo])
    e_i = np.array([r.rel_err_pct for r in inho])
    l2_h = np.array([r.l2_field_err for r in homo])
    l2_i = np.array([r.l2_field_err for r in inho])
    slope_h = np.array([r.log_loss_slope for r in homo])
    slope_i = np.array([r.log_loss_slope for r in inho])

    rep: dict = {
        "descriptive": {
            "kappa_homo": descriptive("kappa_homo", k_h),
            "kappa_inho": descriptive("kappa_inho", k_i),
            "rel_err_homo": descriptive("rel_err_homo", e_h),
            "rel_err_inho": descriptive("rel_err_inho", e_i),
            "l2_field_homo": descriptive("l2_field_homo", l2_h),
            "l2_field_inho": descriptive("l2_field_inho", l2_i),
            "log_loss_slope_homo": descriptive("log_loss_slope_homo", slope_h),
            "log_loss_slope_inho": descriptive("log_loss_slope_inho", slope_i),
        },
    }

    def _sw(x):
        r = stats.shapiro(x)
        return {"W": float(r.statistic), "p": float(r.pvalue)}

    def _t1(x, mu):
        r = stats.ttest_1samp(x, mu)
        return {"t": float(r.statistic), "p": float(r.pvalue)}

    def _welch(a, b):
        r = stats.ttest_ind(a, b, equal_var=False)
        return {"t": float(r.statistic), "p": float(r.pvalue)}

    def _mwu(a, b):
        r = stats.mannwhitneyu(a, b, alternative="two-sided")
        return {"U": float(r.statistic), "p": float(r.pvalue)}

    rep.update({
        "shapiro_wilk": {
            "kappa_homo": _sw(k_h), "kappa_inho": _sw(k_i),
            "rel_err_homo": _sw(e_h), "rel_err_inho": _sw(e_i),
        },
        "t_test_kappa_eq_true": {
            "kappa_homo": _t1(k_h, KAPPA_TRUE),
            "kappa_inho": _t1(k_i, KAPPA_TRUE),
        },
        "homo_vs_inho_kappa": {
            "welch_t": _welch(k_h, k_i),
            "mann_whitney": _mwu(k_h, k_i),
        },
        "homo_vs_inho_rel_err": {
            "welch_t": _welch(e_h, e_i),
            "mann_whitney": _mwu(e_h, e_i),
        },
    })
    return rep


def format_stats(rep: dict) -> str:
    if not rep:
        return "(no two-case statistics: need both homo and inho results)\n"
    L: list[str] = []
    L.append("=" * 72)
    L.append("STATISTICAL REPORT")
    L.append("=" * 72)
    L.append("")
    L.append("Descriptive (mean +- std, 95% bootstrap CI):")
    for key, d in rep["descriptive"].items():
        L.append(f"  {key:>22s}: n={d['n']} "
                 f"mean={d['mean']:.4e} std={d['std']:.4e} "
                 f"median={d['median']:.4e}  CI95=[{d['ci95_low']:.4e}, {d['ci95_high']:.4e}]")
    L.append("")
    L.append("Shapiro-Wilk normality test (H0: normal):")
    for k, v in rep["shapiro_wilk"].items():
        verdict = "fail to reject" if v["p"] > 0.05 else "reject"
        L.append(f"  {k:>22s}: W={v['W']:.4f}  p={v['p']:.4f}  -> {verdict} H0")
    L.append("")
    L.append(f"One-sample t-test (H0: mean(kappa) == kappa_true={KAPPA_TRUE:.4e}):")
    for k, v in rep["t_test_kappa_eq_true"].items():
        verdict = "fail to reject" if v["p"] > 0.05 else "reject"
        L.append(f"  {k:>22s}: t={v['t']:.3f}  p={v['p']:.4f}  -> {verdict} H0")
    L.append("")
    L.append("Two-sample tests (homo vs inho):")
    for label in ("homo_vs_inho_kappa", "homo_vs_inho_rel_err"):
        L.append(f"  {label}:")
        w = rep[label]["welch_t"]
        u = rep[label]["mann_whitney"]
        L.append(f"    Welch t: t={w['t']:.3f}  p={w['p']:.4f}")
        L.append(f"    MW-U   : U={u['U']:.3f}  p={u['p']:.4f}")
    L.append("")
    return "\n".join(L)


# -- Plotting helpers ---------------------------------------------------------
def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# -- Per-seed and aggregated plots --------------------------------------------
def plot_loss_single(r: RunResult, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_a = len(r.loss_adam)
    n_l = len(r.loss_lbfgs)
    ax.semilogy(range(n_a), r.loss_adam, lw=0.8, label="Adam")
    if n_l:
        ax.semilogy(range(n_a, n_a + n_l), r.loss_lbfgs, lw=0.8, color="r",
                    label="L-BFGS")
    ax.axvline(n_a, color="k", ls=":", lw=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss")
    ax.set_title(f"Loss -- {r.tag}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, outdir / f"loss_{r.tag}.png")


def plot_kappa_single(r: RunResult, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_a = len(r.kappa_adam)
    n_l = len(r.kappa_lbfgs)
    ax.semilogy(range(n_a), r.kappa_adam, lw=0.8, label="Adam")
    if n_l:
        ax.semilogy(range(n_a, n_a + n_l), r.kappa_lbfgs, lw=0.8, color="r",
                    label="L-BFGS")
    ax.axhline(KAPPA_TRUE, color="k", ls="--", lw=1, label=r"$\kappa_{true}$")
    ax.axvline(n_a, color="k", ls=":", lw=0.6)
    if r.convergence_epoch > 0:
        ax.axvline(r.convergence_epoch, color="g", ls="--", lw=0.8,
                   label=f"converged @ ep={r.convergence_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\kappa$")
    ax.set_title(f"kappa trajectory -- {r.tag}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, outdir / f"kappa_{r.tag}.png")


def plot_lr_single(r: RunResult, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(r.lr_history, lw=0.8)
    ax.set_xlabel("epoch (Adam)")
    ax.set_ylabel("learning rate (net)")
    ax.set_title(f"LR schedule -- {r.tag}")
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, outdir / f"lr_{r.tag}.png")


def plot_pde_residual_hist(r: RunResult, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    arr = np.asarray(r.pde_resid_after)
    ax.hist(np.log10(arr + 1e-20), bins=60, color="steelblue", alpha=0.85)
    ax.set_xlabel(r"$\log_{10}|$PDE residual$|$")
    ax.set_ylabel("count")
    ax.set_title(f"PDE residual histogram -- {r.tag}  (mean={arr.mean():.2e})")
    ax.grid(True, alpha=0.3)
    _save(fig, outdir / f"pde_residual_{r.tag}.png")


def plot_log_loss_slope(r: RunResult, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    loss = np.maximum(np.asarray(r.loss_adam), 1e-20)
    eps = np.arange(1, len(loss) + 1)
    ax.loglog(eps, loss, lw=0.7, label="log loss")
    if math.isfinite(r.log_loss_slope):
        fit_y = np.exp(r.log_loss_intercept) * eps ** r.log_loss_slope
        ax.loglog(eps, fit_y, "r--", lw=1.2,
                  label=f"fit: slope={r.log_loss_slope:.3f}")
    ax.set_xlabel("epoch (log)")
    ax.set_ylabel("loss (log)")
    ax.set_title(f"Log-log loss convergence -- {r.tag}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, outdir / f"loss_loglog_{r.tag}.png")


def plot_loss_grouped(results: list[RunResult], kind: str, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        if r.kind != kind:
            continue
        ax.semilogy(r.loss_adam + r.loss_lbfgs, lw=0.7, alpha=0.85,
                    label=f"seed={r.seed}")
    ax.set_xlabel("epoch (Adam + L-BFGS)")
    ax.set_ylabel("loss")
    ax.set_title(f"Loss curves -- {kind}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, outdir / f"loss_grouped_{kind}.png")


def plot_kappa_grouped(results: list[RunResult], kind: str, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        if r.kind != kind:
            continue
        ax.semilogy(r.kappa_adam + r.kappa_lbfgs, lw=0.7, alpha=0.85,
                    label=f"seed={r.seed}")
    ax.axhline(KAPPA_TRUE, color="k", ls="--", lw=1, label=r"$\kappa_{true}$")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\kappa$")
    ax.set_title(f"kappa trajectories -- {kind}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, outdir / f"kappa_grouped_{kind}.png")


def plot_boxplot_kappa(results: list[RunResult], outdir: Path) -> None:
    homo = [r.kappa_final for r in results if r.kind == "homo"]
    inho = [r.kappa_final for r in results if r.kind == "inhomo"]
    if not homo or not inho:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot([homo, inho], labels=["homogeneous", "inhomogeneous"],
                    patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#9ecae1", "#fdae6b"]):
        patch.set_facecolor(c)
    for i, data in enumerate([homo, inho], start=1):
        xs = np.random.default_rng(0).normal(i, 0.04, size=len(data))
        ax.scatter(xs, data, color="k", s=20, zorder=3)
    ax.axhline(KAPPA_TRUE, color="r", ls="--", lw=1, label=r"$\kappa_{true}$")
    ax.set_ylabel(r"$\kappa$ identified")
    ax.set_title("Final kappa across seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save(fig, outdir / "kappa_boxplot.png")


# -- Field comparison plots ---------------------------------------------------
def _grid_eval(model: PINN, kind: str, gx: np.ndarray, gy: np.ndarray,
               tval: float) -> tuple[np.ndarray, np.ndarray]:
    analytical = AnalyticalSolution(kappa=KAPPA_TRUE, kind=kind)
    xf = gx.flatten()
    yf = gy.flatten()
    tf = np.full_like(xf, tval)
    u_true = analytical(xf, yf, tf).reshape(gx.shape)
    xt = torch.tensor(xf, dtype=torch.float32, device=device).view(-1, 1)
    yt = torch.tensor(yf, dtype=torch.float32, device=device).view(-1, 1)
    tt = torch.tensor(tf, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        u_pred = model(xt, yt, tt).cpu().numpy().reshape(gx.shape)
    return u_pred, u_true


def plot_field_snapshots(model: PINN, kind: str, seed: int,
                         outdir: Path) -> None:
    times = [1.0, 10.0, 100.0, 250.0, 500.0]
    gx, gy = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60))
    fig, axes = plt.subplots(3, len(times), figsize=(4 * len(times), 10))
    for j, tv in enumerate(times):
        u_pred, u_true = _grid_eval(model, kind, gx, gy, tv)
        err = u_pred - u_true
        for ax, data, ttl, cmap in zip(
            axes[:, j], [u_true, u_pred, err],
            [f"analytical t={tv}", f"PINN t={tv}", f"error t={tv}"],
            ["RdBu_r", "RdBu_r", "RdBu_r"]):
            im = ax.pcolormesh(gx, gy, data, cmap=cmap, shading="auto")
            ax.set_title(ttl, fontsize=10)
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Field snapshots u(x,y,t)  --  {kind}  seed={seed}", fontsize=12)
    _save(fig, outdir / f"field_snapshots_{kind}_seed{seed}.png")


def plot_field_xt_slice(model: PINN, kind: str, seed: int,
                        outdir: Path) -> None:
    """u(x, y=0.5, t) on a 2D grid."""
    nx, nt = 80, 200
    xs = np.linspace(0, 1, nx)
    ts = np.linspace(0, T_MAX, nt)
    gx, gt = np.meshgrid(xs, ts, indexing="xy")
    gy = np.full_like(gx, 0.5)
    analytical = AnalyticalSolution(kappa=KAPPA_TRUE, kind=kind)
    u_true = analytical(gx.flatten(), gy.flatten(), gt.flatten()).reshape(gx.shape)
    xt = torch.tensor(gx.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    yt = torch.tensor(gy.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    tt = torch.tensor(gt.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        u_pred = model(xt, yt, tt).cpu().numpy().reshape(gx.shape)
    err = u_pred - u_true
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, dat, ttl in zip(axes, [u_true, u_pred, err],
                            ["analytical", "PINN", "error"]):
        im = ax.pcolormesh(gt, gx, dat, cmap="RdBu_r", shading="auto")
        ax.set_xlabel("t"); ax.set_ylabel("x")
        ax.set_title(f"{ttl}, y=0.5")
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(f"u(x, 0.5, t) -- {kind}  seed={seed}")
    _save(fig, outdir / f"field_xt_slice_{kind}_seed{seed}.png")


def plot_field_yt_slice(model: PINN, kind: str, seed: int,
                        outdir: Path) -> None:
    """u(x=0.5, y, t) on a 2D grid."""
    ny, nt = 80, 200
    ys = np.linspace(0, 1, ny)
    ts = np.linspace(0, T_MAX, nt)
    gy, gt = np.meshgrid(ys, ts, indexing="xy")
    gx = np.full_like(gy, 0.5)
    analytical = AnalyticalSolution(kappa=KAPPA_TRUE, kind=kind)
    u_true = analytical(gx.flatten(), gy.flatten(), gt.flatten()).reshape(gy.shape)
    xt = torch.tensor(gx.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    yt = torch.tensor(gy.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    tt = torch.tensor(gt.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        u_pred = model(xt, yt, tt).cpu().numpy().reshape(gy.shape)
    err = u_pred - u_true
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, dat, ttl in zip(axes, [u_true, u_pred, err],
                            ["analytical", "PINN", "error"]):
        im = ax.pcolormesh(gt, gy, dat, cmap="RdBu_r", shading="auto")
        ax.set_xlabel("t"); ax.set_ylabel("y")
        ax.set_title(f"{ttl}, x=0.5")
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(f"u(0.5, y, t) -- {kind}  seed={seed}")
    _save(fig, outdir / f"field_yt_slice_{kind}_seed{seed}.png")


def plot_temporal_lines(model: PINN, kind: str, seed: int,
                        outdir: Path) -> None:
    """1D lines u(x*, y*, t) vs t at several probe points."""
    probes = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.25), (0.5, 0.25)]
    t = np.linspace(0, T_MAX, 1000)
    analytical = AnalyticalSolution(kappa=KAPPA_TRUE, kind=kind)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (sx, sy) in zip(axes.flat, probes):
        u_true = analytical(np.full_like(t, sx), np.full_like(t, sy), t)
        xt = torch.full((len(t), 1), sx, device=device)
        yt = torch.full((len(t), 1), sy, device=device)
        tt = torch.tensor(t, dtype=torch.float32, device=device).view(-1, 1)
        with torch.no_grad():
            u_pred = model(xt, yt, tt).cpu().numpy().flatten()
        ax.plot(t, u_true, "k-", lw=1.2, label="analytical")
        ax.plot(t, u_pred, "r--", lw=1.0, label="PINN")
        ax.set_xlabel("t"); ax.set_ylabel("u")
        ax.set_title(f"x={sx}, y={sy}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Temporal profiles -- {kind}  seed={seed}")
    _save(fig, outdir / f"temporal_lines_{kind}_seed{seed}.png")


def plot_field_3d(model: PINN, kind: str, seed: int, outdir: Path) -> None:
    """3D surfaces u(x,y) at one fixed t."""
    tv = 50.0
    gx, gy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    u_pred, u_true = _grid_eval(model, kind, gx, gy, tv)
    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.plot_surface(gx, gy, u_true, cmap="viridis", linewidth=0)
    ax1.set_title(f"Analytical  t={tv}")
    ax2.plot_surface(gx, gy, u_pred, cmap="viridis", linewidth=0)
    ax2.set_title(f"PINN  t={tv}")
    for ax in (ax1, ax2):
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("u")
    fig.suptitle(f"3D surface u(x,y, t={tv}) -- {kind}  seed={seed}")
    _save(fig, outdir / f"field_3d_{kind}_seed{seed}.png")


def plot_adam_vs_lbfgs_temporal(model: PINN, model_after_adam_state: dict,
                                kind: str, seed: int, outdir: Path) -> None:
    """Probe u(0.5,0.5,t) for Adam-only model vs Adam+LBFGS vs analytical."""
    t = np.linspace(0, T_MAX, 1000)
    sx = sy = 0.5
    analytical = AnalyticalSolution(kappa=KAPPA_TRUE, kind=kind)
    u_true = analytical(np.full_like(t, sx), np.full_like(t, sy), t)
    xt = torch.full((len(t), 1), sx, device=device)
    yt = torch.full((len(t), 1), sy, device=device)
    tt = torch.tensor(t, dtype=torch.float32, device=device).view(-1, 1)

    with torch.no_grad():
        u_full = model(xt, yt, tt).cpu().numpy().flatten()

    # restore Adam state, evaluate, then restore current state
    current_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(model_after_adam_state)
    with torch.no_grad():
        u_adam = model(xt, yt, tt).cpu().numpy().flatten()
    model.load_state_dict(current_state)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, u_true, "k-", lw=1.2, label="analytical")
    ax.plot(t, u_adam, "b--", lw=1.0, label="Adam only")
    ax.plot(t, u_full, "r:", lw=1.2, label="Adam + L-BFGS")
    ax.set_xlabel("t"); ax.set_ylabel("u(0.5,0.5,t)")
    ax.set_title(f"Adam vs Adam+LBFGS vs Analytical -- {kind}  seed={seed}")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, outdir / f"adam_vs_lbfgs_{kind}_seed{seed}.png")


# -- Ablation plots -----------------------------------------------------------
def _group_by_kind(results: list[RunResult]) -> dict[str, list[RunResult]]:
    out: dict[str, list[RunResult]] = {}
    for r in results:
        out.setdefault(r.kind, []).append(r)
    return out


def plot_ablation_kappa(results: list[RunResult], outdir: Path) -> None:
    groups = _group_by_kind(results)
    colors = {"homo": "darkred", "inhomo": "darkblue"}

    # Combined: error vs kappa_init for both kinds
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, rs in groups.items():
        rs = sorted(rs, key=lambda r: r.kappa_init)
        inits = np.array([r.kappa_init for r in rs])
        errs = np.array([r.rel_err_pct for r in rs])
        ax.semilogx(inits, errs, "o-", color=colors.get(kind, "k"), label=kind)
    ax.set_xlabel(r"$\kappa_{init}$"); ax.set_ylabel("rel. error  (%)")
    ax.set_title("Ablation: sensitivity to kappa_init")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    _save(fig, outdir / "ablation_kappa_init_err.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, rs in groups.items():
        rs = sorted(rs, key=lambda r: r.kappa_init)
        inits = np.array([r.kappa_init for r in rs])
        finals = np.array([r.kappa_final for r in rs])
        ax.loglog(inits, finals, "o-", color=colors.get(kind, "k"), label=kind)
    ax.axhline(KAPPA_TRUE, color="k", ls="--", lw=1, label=r"$\kappa_{true}$")
    ax.set_xlabel(r"$\kappa_{init}$"); ax.set_ylabel(r"$\kappa_{final}$")
    ax.set_title("Identified kappa vs initialization")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    _save(fig, outdir / "ablation_kappa_init_final.png")


def plot_ablation_noise(results: list[RunResult], outdir: Path) -> None:
    groups = _group_by_kind(results)
    colors = {"homo": "darkgreen", "inhomo": "indigo"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, rs in groups.items():
        rs = sorted(rs, key=lambda r: r.noise_level)
        noises = np.array([r.noise_level for r in rs])
        errs = np.array([r.rel_err_pct for r in rs])
        ax.semilogx(noises, errs, "o-", color=colors.get(kind, "k"), label=kind)
    ax.set_xlabel("noise level (relative)"); ax.set_ylabel("rel. error  (%)")
    ax.set_title("Ablation: sensitivity to measurement noise")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    _save(fig, outdir / "ablation_noise_err.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, rs in groups.items():
        rs = sorted(rs, key=lambda r: r.noise_level)
        noises = np.array([r.noise_level for r in rs])
        finals = np.array([r.kappa_final for r in rs])
        ax.semilogx(noises, finals, "o-", color=colors.get(kind, "k"), label=kind)
    ax.axhline(KAPPA_TRUE, color="k", ls="--", lw=1, label=r"$\kappa_{true}$")
    ax.set_xlabel("noise level (relative)"); ax.set_ylabel(r"$\kappa_{final}$")
    ax.set_title("Identified kappa vs noise")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, outdir / "ablation_noise_final.png")


# -- Main ---------------------------------------------------------------------
def main() -> None:
    log("")
    log("Configuration:")
    log(f"  SEEDS={SEEDS}")
    log(f"  FLAT_EPOCHS={FLAT_EPOCHS}  LBFGS_STEPS={LBFGS_STEPS}")
    log(f"  N_PDE={N_PDE}  T_MAX={T_MAX}  N_TIME={N_TIME}")
    log(f"  noise_default={NOISE_DEFAULT}  kappa_init_default={KAPPA_INIT_DEFAULT}")
    log(f"  W_PDE={W_PDE}  W_DATA={W_DATA}")
    log(f"  sensors={SENSOR_XY}")
    log(f"  ablation kappa values={ABL_KAPPA_VALUES}")
    log(f"  ablation noise values={ABL_NOISE_VALUES}")

    # ---------------- MAIN STUDY ---------------------------------------------
    main_results: list[RunResult] = []
    # We keep last-trained model per (kind, seed) for plots
    main_models: dict[str, PINN] = {}
    adam_only_states: dict[str, dict] = {}

    for kind in ("homo", "inhomo"):
        for seed in SEEDS:
            tag = f"main_{kind}_seed{seed}"
            r, model = train_one(tag, kind, seed,
                                 kappa_init=KAPPA_INIT_DEFAULT,
                                 noise_level=NOISE_DEFAULT)
            main_results.append(r)
            main_models[tag] = model
            # Best seed by rel_err -> we'll render field plots from it.
            # Also reconstruct Adam-only network for comparison: store
            # a snapshot of the kappa post-Adam; the network state at that
            # moment isn't saved (would need extra memory) so the
            # Adam-vs-LBFGS plot uses the same network but with
            # log_kappa swapped to kappa_after_adam.
            adam_state = {k: v.clone() for k, v in model.state_dict().items()}
            adam_state["log_kappa"] = torch.tensor(
                math.log(r.kappa_after_adam), device=device)
            adam_only_states[tag] = adam_state

    # Per-seed plots
    log("")
    log("Plotting per-seed figures ...")
    for r in main_results:
        plot_loss_single(r, DIR_MAIN)
        plot_kappa_single(r, DIR_MAIN)
        plot_lr_single(r, DIR_MAIN)
        plot_pde_residual_hist(r, DIR_MAIN)
        plot_log_loss_slope(r, DIR_MAIN)

    # Grouped plots per kind
    for kind in ("homo", "inhomo"):
        plot_loss_grouped(main_results, kind, DIR_MAIN)
        plot_kappa_grouped(main_results, kind, DIR_MAIN)
    plot_boxplot_kappa(main_results, DIR_MAIN)

    # Field comparison plots: render for the best seed of each kind.
    for kind in ("homo", "inhomo"):
        best = min((r for r in main_results if r.kind == kind),
                   key=lambda r: r.rel_err_pct)
        tag = f"main_{kind}_seed{best.seed}"
        m = main_models[tag]
        log(f"Rendering field plots for {tag} (rel_err={best.rel_err_pct:.2f}%)")
        plot_field_snapshots(m, kind, best.seed, DIR_MAIN)
        plot_field_xt_slice(m, kind, best.seed, DIR_MAIN)
        plot_field_yt_slice(m, kind, best.seed, DIR_MAIN)
        plot_temporal_lines(m, kind, best.seed, DIR_MAIN)
        plot_field_3d(m, kind, best.seed, DIR_MAIN)
        plot_adam_vs_lbfgs_temporal(m, adam_only_states[tag], kind,
                                    best.seed, DIR_MAIN)

    # Statistics
    log("")
    rep = run_statistics(main_results)
    txt = format_stats(rep)
    for line in txt.splitlines():
        log(line)
    (DIR_MAIN / "stats_report.txt").write_text(txt, encoding="utf-8")

    # Helper: find a main-run result that matches given ablation point
    # (seed=ABL_SEED, given kind, kappa_init, noise_level). If found,
    # reuse it instead of training again.
    def find_main(kind: str, kinit: float, noise: float) -> RunResult | None:
        for r in main_results:
            if (r.kind == kind and r.seed == ABL_SEED
                    and math.isclose(r.kappa_init, kinit, rel_tol=1e-9)
                    and math.isclose(r.noise_level, noise, rel_tol=1e-9)):
                return r
        return None

    def render_single_run_plots(r: RunResult, outdir: Path) -> None:
        """Full per-run plot set: loss, kappa, lr, pde-residual, log-log loss."""
        plot_loss_single(r, outdir)
        plot_kappa_single(r, outdir)
        plot_lr_single(r, outdir)
        plot_pde_residual_hist(r, outdir)
        plot_log_loss_slope(r, outdir)

    # ---------------- ABLATION: kappa_init -----------------------------------
    log("")
    log("=" * 72)
    log("ABLATION: kappa_init  (both homo and inhomo, seed=" f"{ABL_SEED})")
    log("=" * 72)
    abl_kappa_results: list[RunResult] = []
    for kind in ABL_KINDS:
        for kinit in ABL_KAPPA_VALUES:
            existing = find_main(kind, kinit, NOISE_DEFAULT)
            if existing is not None:
                log(f"  reuse main result for {kind} kappa_init={kinit:.1e}: "
                    f"{existing.tag} (kappa={existing.kappa_final:.4e} "
                    f"err={existing.rel_err_pct:.2f}%)")
                abl_kappa_results.append(existing)
                # render the same per-run plot set into DIR_ABL_KAPPA so this
                # ablation directory has a complete, self-contained set of
                # figures (including this reused point).
                render_single_run_plots(existing, DIR_ABL_KAPPA)
                continue
            tag = (f"abl_kappa_init_{kind}_{kinit:.0e}"
                   .replace("e-0", "e-").replace("e+0", "e+"))
            r, _ = train_one(tag, kind, ABL_SEED,
                             kappa_init=kinit, noise_level=NOISE_DEFAULT)
            abl_kappa_results.append(r)
            render_single_run_plots(r, DIR_ABL_KAPPA)
    plot_ablation_kappa(abl_kappa_results, DIR_ABL_KAPPA)

    # ---------------- ABLATION: noise ----------------------------------------
    log("")
    log("=" * 72)
    log(f"ABLATION: noise  (both homo and inhomo, seed={ABL_SEED})")
    log("=" * 72)
    abl_noise_results: list[RunResult] = []
    for kind in ABL_KINDS:
        for noise in ABL_NOISE_VALUES:
            existing = find_main(kind, KAPPA_INIT_DEFAULT, noise)
            if existing is not None:
                log(f"  reuse main result for {kind} noise={noise:.1e}: "
                    f"{existing.tag} (kappa={existing.kappa_final:.4e} "
                    f"err={existing.rel_err_pct:.2f}%)")
                abl_noise_results.append(existing)
                render_single_run_plots(existing, DIR_ABL_NOISE)
                continue
            tag = (f"abl_noise_{kind}_{noise:.0e}".replace("e-0", "e-"))
            r, _ = train_one(tag, kind, ABL_SEED,
                             kappa_init=KAPPA_INIT_DEFAULT, noise_level=noise)
            abl_noise_results.append(r)
            render_single_run_plots(r, DIR_ABL_NOISE)
    plot_ablation_noise(abl_noise_results, DIR_ABL_NOISE)

    # ---------------- Save metrics ------------------------------------------
    def serialise(r: RunResult) -> dict:
        d = asdict(r)
        # Truncate histories to keep JSON manageable -- we already saved figures.
        for k in ("loss_adam", "loss_lbfgs", "kappa_adam", "kappa_lbfgs",
                  "lr_history", "pde_resid_after"):
            arr = d[k]
            if isinstance(arr, list) and len(arr) > 500:
                idx = np.linspace(0, len(arr) - 1, 500).astype(int)
                d[k] = [arr[i] for i in idx]
        return d

    metrics = {
        "run_id": RUN_ID,
        "kappa_true": KAPPA_TRUE,
        "config": {
            "T_MAX": T_MAX, "N_TIME": N_TIME, "sensors": SENSOR_XY,
            "seeds": SEEDS, "flat_epochs": FLAT_EPOCHS,
            "lbfgs_steps": LBFGS_STEPS, "n_pde": N_PDE,
            "w_pde": W_PDE, "w_data": W_DATA,
            "noise_default": NOISE_DEFAULT,
            "kappa_init_default": KAPPA_INIT_DEFAULT,
            "abl_kappa_values": ABL_KAPPA_VALUES,
            "abl_noise_values": ABL_NOISE_VALUES,
        },
        "main_runs": [serialise(r) for r in main_results],
        "ablation_kappa_runs": [serialise(r) for r in abl_kappa_results],
        "ablation_noise_runs": [serialise(r) for r in abl_noise_results],
        "stats_main": rep,
    }
    (ROOT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    log("")
    log(f"All results saved to: {ROOT_DIR}")
    log("Subdirs: main/ ablation_kappa/ ablation_noise/")


if __name__ == "__main__":
    main()
