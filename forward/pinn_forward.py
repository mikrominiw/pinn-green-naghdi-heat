"""
PINN for GN-III forward problem.

PDE:  u_tt - kappa*(u_xxt + u_yyt) - (u_xx + u_yy) = dq/dt
Domain: (0,1)^2 x [0, T_MAX],  kappa = 1.2e-4  (known)

Two source variants:
    homo   : q = 0
    inhomo : q = 2*x*t  =>  dq/dt = 2*x

Hard IC/BC via Hermite-distance ansatz with built-in oscillation carrier:
    u = sin(pi*x)*sin(pi*y)*cos(omega0*t) + d(x,y)*tau^2*N(x,y,t)
    d(x,y) = x*(1-x)*y*(1-y),  tau = t/T_MAX,  omega0 = pi*sqrt(2)
    => u(0)=theta0, u_t(0)=0, u|dOmega=0 satisfied identically.
    The carrier cos(omega0*t) removes spectral bias: N learns only the
    slow envelope and the forced response.

Soft IC/BC: PDE + IC1 + IC2 + BC losses with learnable log-weights.

Architecture (fixed, ~17k params):
    [x, y, sin(w1*t), cos(w1*t), ..., sin(w4*t), cos(w4*t), t/T] -> 2 ResBlocks -> u
    Fourier frequencies are FIXED at multiples of omega0 (not learnable).

Optimisers: NAdam (flat_epochs) then L-BFGS (lbfgs_steps). No scheduler.

Study: 2 sources x 2 ansatze x 3 seeds = 12 runs.
Optuna: tune LR on best config. Final run with best LR.
"""
from __future__ import annotations

import json
import math
import time
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import autograd, optim
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------
RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT_DIR = Path(__file__).parent / "results" / RUN_ID
ROOT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("pinn_fwd")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
_fh  = logging.FileHandler(ROOT_DIR / "log.txt", mode="w", encoding="utf-8")
_sh  = logging.StreamHandler()
for h in (_fh, _sh):
    h.setFormatter(_fmt)
    logger.addHandler(h)

def log(msg: str = "") -> None:
    logger.info(msg)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32  = True
torch.backends.cudnn.allow_tf32        = True
torch.backends.cudnn.benchmark         = True

log(f"Run ID : {RUN_ID}")
log(f"Device : {device}" + (f"  ({torch.cuda.get_device_name(0)})"
                             if torch.cuda.is_available() else ""))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PI        = math.pi
KAPPA     = 1.2e-4
T_MAX     = 30.0                  # текущий интервал (меняется train_one по t_max)
OMEGA0    = PI * math.sqrt(2.0)   # dominant eigenfrequency, mode (1,1)
SOURCES   = ["homo", "inhomo"]
ANSATZE   = ["hard", "soft"]
SEEDS     = [16, 5, 22]           # 3 инициализации для статистики

# Исследование A: зависимость точности от длины интервала T (hard vs soft).
T_LIST    = [3.0, 30.0, 100.0]

# Исследование B (размер сети) проводится на одном представительном T:
SIZE_T    = 30.0
SIZE_SRC  = "homo"

# Training
FLAT_EPOCHS  = 8_000
LBFGS_STEPS  = 200
LR_DEFAULT   = 1e-3
N_PDE        = 15_000    # крупный батч: лучше загружает GPU и точнее градиент
N_IC         = 4_000     # (НУ/ГУ используются только мягким представлением)
N_BC         = 4_000
LOG_EVERY    = 1_000

# Ранняя остановка NAdam: прерываем, если потеря не улучшилась за
# EARLY_PATIENCE эпох (после MIN_EPOCHS). Снижает время на сошедшихся прогонах.
MIN_EPOCHS     = 3_000
EARLY_PATIENCE = 1_500
EARLY_REL_TOL  = 1e-4    # «улучшение» = снижение лучшей потери на >0.01 %

# Исследование размера архитектуры (только жёсткое представление решения) через Optuna.
# Цель — построить кривую L2(число параметров) и выделить две схемы:
# лучшую по качеству и лучшую по соотношению качество/скорость.
ARCH_HIDDEN   = [16, 32, 48, 64]     # варианты ширины
ARCH_BLOCKS   = [1, 2, 3, 4]         # варианты глубины (число ResBlock)
# Полный перебор сетки (GridSampler): каждая конфигурация ровно один раз
# (без отсечения по числу параметров — диапазон 753 ... 34 113).
OPTUNA_EPOCHS = 3_000                # эпох NAdam на одно испытание скрининга

# Этап 2: настоящая байесовская оптимизация (TPE) скорости обучения LR
# на лучшей по качеству схеме.
LR_TRIALS = 10                       # испытаний TPE
LR_EPOCHS = 2_000                    # эпох NAdam на одно испытание LR
LR_LOW, LR_HIGH = 1e-4, 3e-3         # диапазон поиска LR (лог-шкала)

# Architecture (fixed, ~17 k params)
HIDDEN  = 64
NBLOCKS = 2
N_FREQ  = 4   # learnable Fourier pairs -> 2*N_FREQ+1 time features

# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
_METRICS = ROOT_DIR / "metrics.json"
_done: set[str] = set()

def _load_done() -> None:
    if _METRICS.exists():
        data = json.loads(_METRICS.read_text(encoding="utf-8"))
        for lst in data.values():
            for r in lst:
                _done.add(r["tag"])
        log(f"Resume: {len(_done)} completed tags.")

_load_done()

def _save(category: str, d: dict) -> None:
    data = json.loads(_METRICS.read_text(encoding="utf-8")) \
        if _METRICS.exists() else {}
    data.setdefault(category, []).append(d)
    _METRICS.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    _done.add(d["tag"])

# ---------------------------------------------------------------------------
# Analytical reference
# ---------------------------------------------------------------------------

class AnalyticalSolution:
    """
    Modal ODE reference, integrated with DOP853 (error ~1e-10).
    A_nm'' + kappa*lam*A_nm' + lam*A_nm = F_nm'(t),  A(0)=A0, A'(0)=0.
    u = sum A_nm(t)*sin(n*pi*x)*sin(m*pi*y).
    """

    def __init__(self, source: str, t_max: float, n_modes: int = 20):
        self.source = source
        self.t_max = t_max
        self._odes: dict = {}

        for n in range(1, n_modes + 1):
            for m in range(1, n_modes + 1):
                A0   = 1.0 if (n == 1 and m == 1) else 0.0
                fprime = self._fprime(n, m, source)
                lam  = PI**2 * (n**2 + m**2)

                def rhs(t, y, lam=lam, fp=fprime):
                    f = fp(t) if fp is not None else 0.0
                    return [y[1], f - KAPPA*lam*y[1] - lam*y[0]]

                t_eval = np.linspace(0, t_max, max(2001, int(t_max*10+1)))
                sol = solve_ivp(rhs, [0, t_max], [A0, 0.0],
                                method="DOP853", t_eval=t_eval,
                                rtol=1e-10, atol=1e-12, dense_output=True)
                self._odes[(n, m)] = sol

    @staticmethod
    def _fprime(n: int, m: int, source: str):
        """dq_nm/dt — time derivative of modal coefficient of q."""
        if source == "homo":
            return None
        if source == "inhomo":
            # q=2xt -> q_nm = 2t * 4*int x*sin(n*pi*x)dx * int sin(m*pi*y)dy
            # int_0^1 x sin(npix)dx = (-1)^(n+1)/(n*pi)
            # int_0^1 sin(mpiy)dy   = (1-cos(m*pi))/(m*pi)  = 0 if m even else 2/(m*pi)
            if m % 2 == 0:
                return None
            Ix = (-1)**(n+1) / (n * PI)
            Iy = 2.0 / (m * PI)
            c  = 4 * 2 * Ix * Iy   # dq_nm/dt = c (constant)
            if abs(c) < 1e-15:
                return None
            return lambda t, _c=c: _c
        return None

    def __call__(self, x: np.ndarray, y: np.ndarray,
                 t: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.float64)
        for (n, m), sol in self._odes.items():
            A = sol.sol(t)[0]
            out += A * np.sin(n*PI*x) * np.sin(m*PI*y)
        return out


_anal_cache: dict[tuple[str, float], AnalyticalSolution] = {}

def get_anal(source: str, t_max: float = None) -> AnalyticalSolution:
    if t_max is None:
        t_max = T_MAX
    key = (source, float(t_max))
    if key not in _anal_cache:
        log(f"  Building analytical reference for '{source}', T={t_max:g} ...")
        # homo: ненулевая только мода (1,1) -> n_modes=1 даёт точное решение.
        # inhomo: вынужденный отклик по нечётным m -> 15 мод достаточно.
        n_modes = 1 if source == "homo" else 15
        _anal_cache[key] = AnalyticalSolution(source, t_max, n_modes=n_modes)
    return _anal_cache[key]

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.l1.weight)
        nn.init.xavier_normal_(self.l2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.l2(torch.tanh(self.l1(torch.tanh(x))))


class PINN(nn.Module):
    """
    Fixed architecture (~17k params).
    Time encoding: learnable Fourier [sin(w_k*t), cos(w_k*t)] + t/T_MAX.
    """

    def __init__(self, source: str, ansatz: str = "hard",
                 hidden: int = HIDDEN, n_blocks: int = NBLOCKS,
                 n_freq: int = N_FREQ):
        super().__init__()
        self.source = source
        self.ansatz = ansatz

        # Фиксированные частоты Фурье-кодирования времени:
        # кратные собственной частоте моды (1,1)  ω₀ = π√2.
        # В отличие от обучаемых частот, фиксированные не «уплывают»
        # при больших T и гарантируют представимость колебаний.
        self.register_buffer(
            "freqs",
            torch.tensor([OMEGA0 * (k + 1) for k in range(n_freq)],
                         dtype=torch.float32))

        t_dim  = 2 * n_freq + 1          # sin/cos pairs + t/T_MAX
        in_dim = 2 + t_dim               # x, y, time_features

        self.proj  = nn.Linear(in_dim, hidden)
        nn.init.xavier_normal_(self.proj.weight)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.out    = nn.Linear(hidden, 1)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        # Learnable loss weights for soft ansatz
        if ansatz == "soft":
            self.log_w_pde = nn.Parameter(torch.tensor(0.0))
            self.log_w_ic1 = nn.Parameter(torch.tensor(math.log(50.0)))
            self.log_w_ic2 = nn.Parameter(torch.tensor(math.log(50.0)))
            self.log_w_bc  = nn.Parameter(torch.tensor(math.log(50.0)))

    def _encode_t(self, t: torch.Tensor) -> torch.Tensor:
        phase = t * self.freqs.unsqueeze(0)       # (N, K)
        return torch.cat([torch.sin(phase),
                          torch.cos(phase),
                          t / T_MAX], dim=1)      # (N, 2K+1)

    def _net(self, x: torch.Tensor, y: torch.Tensor,
             t: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.proj(torch.cat([x, y, self._encode_t(t)], dim=1)))
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        N = self._net(x, y, t)
        if self.ansatz == "soft":
            return N

        # Жёсткое представление решения на основе функции-расстояния Эрмита со
        # встроенным носителем колебаний cos(ω₀t):
        #   u = sin(πx)sin(πy)·cos(ω₀t) + d(x,y)·τ²·N
        # где d(x,y)=x(1-x)y(1-y) зануляется на всех 4 границах.
        # Проверка условий:
        #   u(t=0)   = sin(πx)sin(πy)·1 + 0 = θ₀          (НУ по значению)
        #   ∂u/∂t|₀  = -ω₀·θ₀·sin(0) + 0   = 0            (НУ по скорости)
        #   u|∂Ω     = 0 (т.к. sin(πx)sin(πy)=0 и d=0)     (ГУ Дирихле)
        # Носитель cos(ω₀t) снимает спектральную предвзятость:
        # сеть N учит лишь медленную огибающую и вынужденную составляющую.
        d   = x * (1 - x) * y * (1 - y)
        tau = t / T_MAX
        theta0  = torch.sin(PI * x) * torch.sin(PI * y)
        carrier = torch.cos(OMEGA0 * t)
        return theta0 * carrier + d * tau**2 * N


# ---------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------

def pde_residual(model: PINN, x: torch.Tensor, y: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    def g(u, v):
        return autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                             create_graph=True, retain_graph=True)[0]

    u     = model(x, y, t)
    u_x   = g(u, x);    u_y  = g(u, y);   u_t  = g(u, t)
    u_xx  = g(u_x, x);  u_yy = g(u_y, y); u_tt = g(u_t, t)
    u_xxt = g(u_xx, t); u_yyt = g(u_yy, t)

    rhs = 2.0*x if model.source == "inhomo" else torch.zeros_like(x)
    return u_tt - KAPPA*(u_xxt + u_yyt) - (u_xx + u_yy) - rhs

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _r(n: int) -> torch.Tensor:
    return torch.rand(n, 1, device=device)

def s_pde(n: int = N_PDE):
    return _r(n), _r(n), _r(n) * T_MAX

def s_ic(n: int = N_IC):
    return _r(n), _r(n), torch.zeros(n, 1, device=device)

def s_bc(n: int = N_BC):
    q = n // 4
    xs = torch.cat([torch.zeros(q,1,device=device), torch.ones(q,1,device=device),
                    _r(q),                           _r(q)])
    ys = torch.cat([_r(q),                           _r(q),
                    torch.zeros(q,1,device=device),  torch.ones(q,1,device=device)])
    ts = _r(n) * T_MAX
    return xs, ys, ts

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(model: PINN) -> tuple[torch.Tensor, dict]:
    x, y, t = s_pde()
    res  = pde_residual(model, x, y, t)
    l_pde = (res**2).mean()

    if model.ansatz == "hard":
        info = {"pde": l_pde.item(), "total": l_pde.item()}
        return l_pde, info

    # Soft: IC1, IC2, BC with learnable weights
    xi, yi, ti = s_ic()
    u_ic = model(xi, yi, ti)
    l_ic1 = ((u_ic - torch.sin(PI*xi)*torch.sin(PI*yi))**2).mean()

    ti_g = ti.requires_grad_(True)
    u_ic2 = model(xi, yi, ti_g)
    u_t0  = autograd.grad(u_ic2, ti_g, grad_outputs=torch.ones_like(u_ic2),
                          create_graph=True)[0]
    l_ic2 = (u_t0**2).mean()

    xb, yb, tb = s_bc()
    l_bc = (model(xb, yb, tb)**2).mean()

    wp = torch.exp(model.log_w_pde)
    wi1= torch.exp(model.log_w_ic1)
    wi2= torch.exp(model.log_w_ic2)
    wb = torch.exp(model.log_w_bc)
    total = wp*l_pde + wi1*l_ic1 + wi2*l_ic2 + wb*l_bc
    info  = {"pde": l_pde.item(), "ic1": l_ic1.item(),
             "ic2": l_ic2.item(), "bc":  l_bc.item(),
             "w_pde": wp.item(), "w_ic1": wi1.item(),
             "w_ic2": wi2.item(), "w_bc": wb.item(),
             "total": total.item()}
    return total, info

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def field_metrics(model: PINN, n_space: int = 40,
                  n_time: int = 30) -> tuple[float, float]:
    anal = get_anal(model.source)
    xs   = np.linspace(0, 1, n_space)
    ts   = np.linspace(0, T_MAX, n_time)
    xx, _, tt = np.meshgrid(xs, xs, ts)
    xf, yf, tf = xx.ravel(), \
                 np.meshgrid(xs, xs, ts)[1].ravel(), \
                 tt.ravel()
    u_ref = anal(xf, yf, tf)
    chunk = 20_000
    pred  = []
    for i in range(0, len(xf), chunk):
        xb = torch.tensor(xf[i:i+chunk], dtype=torch.float32, device=device).unsqueeze(1)
        yb = torch.tensor(yf[i:i+chunk], dtype=torch.float32, device=device).unsqueeze(1)
        tb = torch.tensor(tf[i:i+chunk], dtype=torch.float32, device=device).unsqueeze(1)
        pred.append(model(xb, yb, tb).squeeze(1).cpu().numpy())
    u_pred = np.concatenate(pred)
    err  = u_pred - u_ref
    denom_l2 = np.sqrt(np.mean(u_ref**2)) + 1e-12
    denom_li = np.max(np.abs(u_ref))       + 1e-12
    return (float(np.sqrt(np.mean(err**2)) / denom_l2),
            float(np.max(np.abs(err))      / denom_li))

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    tag:      str
    source:   str
    ansatz:   str
    seed:     int
    lr:       float
    t_max:    float  = 0.0
    hidden:   int    = 0
    n_blocks: int    = 0
    l2:       float  = 0.0
    linf:     float  = 0.0
    time_s:   float  = 0.0
    loss_hist: list[float] = field(default_factory=list)
    info_hist: list[dict]  = field(default_factory=list)

    def slim(self) -> dict:
        d = asdict(self)
        d["loss_hist"] = d["loss_hist"][::20]
        d["info_hist"] = d["info_hist"][::20]
        return d

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def count_params(hidden: int, n_blocks: int, n_freq: int = N_FREQ) -> int:
    """Число обучаемых параметров PINN для заданной (ширина, глубина)."""
    in_dim = 2 + 2 * n_freq + 1
    proj  = in_dim * hidden + hidden
    block = 2 * (hidden * hidden + hidden)
    out   = hidden + 1
    return proj + n_blocks * block + out


def train_one(tag: str, source: str, ansatz: str, seed: int,
              lr: float = LR_DEFAULT,
              flat_epochs: int = FLAT_EPOCHS,
              lbfgs_steps: int = LBFGS_STEPS,
              hidden: int = HIDDEN,
              n_blocks: int = NBLOCKS,
              t_max: float = None) -> tuple[RunResult, PINN]:

    global T_MAX
    if t_max is not None:
        T_MAX = float(t_max)      # текущий интервал — глобально на этот прогон

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PINN(source, ansatz=ansatz, hidden=hidden,
                 n_blocks=n_blocks).to(device)
    npar  = sum(p.numel() for p in model.parameters())
    log(f"  [{tag}] source={source} ansatz={ansatz} seed={seed} T={T_MAX:g} "
        f"lr={lr:.2e} hidden={hidden} blocks={n_blocks} params={npar:,}")

    res = RunResult(tag=tag, source=source, ansatz=ansatz, seed=seed, lr=lr,
                    t_max=T_MAX, hidden=hidden, n_blocks=n_blocks)
    t0  = time.time()

    # ---- NAdam ----
    opt = optim.NAdam(model.parameters(), lr=lr)
    best_loss = float("inf")
    stall = 0
    for ep in range(flat_epochs):
        opt.zero_grad()
        loss, info = compute_loss(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        res.loss_hist.append(info["total"])
        res.info_hist.append(info)
        if (ep + 1) % LOG_EVERY == 0:
            parts = [f"pde={info['pde']:.2e}"]
            if "ic1" in info: parts.append(f"ic1={info['ic1']:.2e}")
            if "ic2" in info: parts.append(f"ic2={info['ic2']:.2e}")
            if "bc"  in info: parts.append(f"bc={info['bc']:.2e}")
            log(f"  [{tag}] ep={ep+1:5d}  loss={info['total']:.3e}  "
                + "  ".join(parts))
        # ранняя остановка по плато
        cur = info["total"]
        if cur < best_loss * (1.0 - EARLY_REL_TOL):
            best_loss = cur
            stall = 0
        else:
            stall += 1
        if ep + 1 >= MIN_EPOCHS and stall >= EARLY_PATIENCE:
            log(f"  [{tag}] ранняя остановка на эпохе {ep+1} "
                f"(плато {EARLY_PATIENCE} эпох, loss={cur:.3e})")
            break

    # ---- L-BFGS ----
    x_p, y_p, t_p = s_pde(N_PDE)
    x_i, y_i, _   = s_ic(N_IC)
    x_b, y_b, t_b = s_bc(N_BC)

    lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=50,
                         history_size=100, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        xp = x_p.detach().requires_grad_(True)
        yp = y_p.detach().requires_grad_(True)
        tp = t_p.detach().requires_grad_(True)
        res_pde = pde_residual(model, xp, yp, tp)
        l = (res_pde**2).mean()

        if model.ansatz == "soft":
            ti_g = torch.zeros(N_IC, 1, device=device).requires_grad_(True)
            u0   = model(x_i.detach(), y_i.detach(), ti_g)
            l_ic1 = ((u0 - torch.sin(PI*x_i.detach())*torch.sin(PI*y_i.detach()))**2).mean()
            u_t0  = autograd.grad(u0, ti_g, grad_outputs=torch.ones_like(u0),
                                  create_graph=True)[0]
            l_ic2 = (u_t0**2).mean()
            l_bc  = (model(x_b.detach(), y_b.detach(), t_b.detach())**2).mean()
            wp = torch.exp(model.log_w_pde); wi1 = torch.exp(model.log_w_ic1)
            wi2 = torch.exp(model.log_w_ic2); wb = torch.exp(model.log_w_bc)
            l = wp*l + wi1*l_ic1 + wi2*l_ic2 + wb*l_bc

        l.backward()
        return l

    best_lb  = float("inf")
    stall_lb = 0
    LB_PATIENCE = 20
    LB_REL_TOL  = 1e-5
    for step in range(lbfgs_steps):
        val = lbfgs.step(closure)
        if val is None:
            continue
        v = float(val)
        res.loss_hist.append(v)
        res.info_hist.append({"total": v, "pde": v})
        if (step + 1) % 50 == 0:
            log(f"  [{tag}] L-BFGS {step+1:3d}  loss={v:.3e}")
        if v < best_lb * (1.0 - LB_REL_TOL):
            best_lb = v; stall_lb = 0
        else:
            stall_lb += 1
        if stall_lb >= LB_PATIENCE:
            log(f"  [{tag}] L-BFGS ранняя остановка на шаге {step+1} "
                f"(плато {LB_PATIENCE} шагов, loss={v:.3e})")
            break

    res.time_s = time.time() - t0
    res.l2, res.linf = field_metrics(model)
    log(f"  [{tag}] DONE  L2={res.l2:.3e}  Linf={res.linf:.3e}  "
        f"t={res.time_s:.0f}s")
    return res, model

# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

# Цвет + стиль линий: цвет для экранного чтения, стиль — дублирующий
# признак для чёрно-белой печати.
_LINE_STYLES = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2))]
_MARKERS     = ["o", "s", "^", "D", "v", "P"]
_COLORS      = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                "#ff7f0e", "#8c564b"]
_REF_COLOR   = "#1f77b4"   # аналитическое решение (синий)
_PINN_COLOR  = "#d62728"   # решение PINN (красный)
_SRC_RU      = {"homo": "однородный (q=0)", "inhomo": "неоднородный (q=2xt)"}
_ANS_RU      = {"hard": "жёсткое наложение НУ/ГУ",
                "soft": "мягкое наложение НУ/ГУ"}

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "legend.framealpha": 0.85,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.4,
    "figure.constrained_layout.use": False,
})


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --- 1. Функция потерь ---

def plot_loss(r: RunResult, outdir: Path) -> None:
    hist = np.array(r.loss_hist)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(hist, lw=1.2, color=_REF_COLOR, ls="-",
                label="Суммарная функция потерь")
    ax.axvline(FLAT_EPOCHS, color="gray", ls="--", lw=1.2,
               label=f"NAdam → L-BFGS (итерация {FLAT_EPOCHS})")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Значение функции потерь")
    src_ru = _SRC_RU.get(r.source, r.source)
    ans_ru = _ANS_RU.get(r.ansatz, r.ansatz)
    ax.set_title(f"Сходимость обучения\n"
                 f"Источник: {src_ru}, ограничения: {ans_ru}, seed={r.seed}")
    ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, outdir / f"{r.tag}_loss.png")


# --- 2. Компоненты функции потерь (для soft) ---

def plot_loss_components(r: RunResult, outdir: Path) -> None:
    if r.ansatz != "soft" or not r.info_hist:
        return
    comp_keys = [("pde",  "Невязка УЧП"),
                 ("ic1",  "НУ по значению"),
                 ("ic2",  "НУ по скорости"),
                 ("bc",   "Граничные условия")]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (k, lbl) in enumerate(comp_keys):
        vals = [d.get(k, float("nan")) for d in r.info_hist]
        ax.semilogy(vals, lw=1.2, ls=_LINE_STYLES[i],
                    color=_COLORS[i % len(_COLORS)], label=lbl)
    ax.axvline(FLAT_EPOCHS, color="gray", ls=":", lw=1.0,
               label=f"NAdam → L-BFGS")
    ax.set_xlabel("Итерация (прореженные данные)")
    ax.set_ylabel("Значение компоненты потерь")
    src_ru = _SRC_RU.get(r.source, r.source)
    ax.set_title(f"Компоненты функции потерь (мягкие ограничения)\n"
                 f"Источник: {src_ru}, seed={r.seed}")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    _savefig(fig, outdir / f"{r.tag}_loss_components.png")


# --- 3. Адаптивные веса (для soft) ---

def plot_soft_weights(r: RunResult, outdir: Path) -> None:
    if r.ansatz != "soft" or not r.info_hist:
        return
    weight_keys = [("w_pde", "Вес УЧП"),
                   ("w_ic1", "Вес НУ (значение)"),
                   ("w_ic2", "Вес НУ (скорость)"),
                   ("w_bc",  "Вес ГУ")]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (k, lbl) in enumerate(weight_keys):
        vals = [d.get(k, float("nan")) for d in r.info_hist]
        ax.semilogy(vals, lw=1.2, ls=_LINE_STYLES[i],
                    color=_COLORS[i % len(_COLORS)], label=lbl)
    ax.set_xlabel("Итерация (прореженные данные)")
    ax.set_ylabel("Значение адаптивного веса")
    src_ru = _SRC_RU.get(r.source, r.source)
    ax.set_title(f"Динамика адаптивных весов\n"
                 f"Источник: {src_ru}, seed={r.seed}")
    ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, outdir / f"{r.tag}_weights.png")


# --- 4. Временны́е профили в контрольных точках ---

@torch.no_grad()
def plot_temporal(model: PINN, r: RunResult, outdir: Path) -> None:
    anal    = get_anal(r.source)
    t_vals  = np.linspace(0, T_MAX, 500)
    sensors = [(0.25, 0.25), (0.50, 0.50), (0.75, 0.75),
               (0.25, 0.75), (0.75, 0.25)]
    n_s = len(sensors)
    fig, axes = plt.subplots(1, n_s, figsize=(4.5 * n_s, 4), sharey=False)

    for ax, (sx, sy) in zip(axes, sensors):
        xv = np.full_like(t_vals, sx); yv = np.full_like(t_vals, sy)
        u_ref  = anal(xv, yv, t_vals)
        xb = torch.tensor(xv, dtype=torch.float32, device=device).unsqueeze(1)
        yb = torch.tensor(yv, dtype=torch.float32, device=device).unsqueeze(1)
        tb = torch.tensor(t_vals, dtype=torch.float32, device=device).unsqueeze(1)
        u_pred = model(xb, yb, tb).squeeze(1).cpu().numpy()
        ax.plot(t_vals, u_ref,  color=_REF_COLOR, ls="-",  lw=1.4,
                label="Аналитическое решение")
        ax.plot(t_vals, u_pred, color=_PINN_COLOR, ls="--", lw=1.1,
                label="PINN")
        ax.set_title(f"Точка ({sx}, {sy})", fontsize=9)
        ax.set_xlabel("Время $t$")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Температура $u$")
    axes[0].legend(loc="upper right")
    src_ru = _SRC_RU.get(r.source, r.source)
    ans_ru = _ANS_RU.get(r.ansatz, r.ansatz)
    plt.suptitle(f"Временны́е профили в контрольных точках\n"
                 f"Источник: {src_ru}, ограничения: {ans_ru}, seed={r.seed}",
                 fontsize=10)
    plt.tight_layout()
    _savefig(fig, outdir / f"{r.tag}_temporal.png")


# --- 5. Снимки поля ---

@torch.no_grad()
def plot_snapshots(model: PINN, r: RunResult, outdir: Path) -> None:
    anal    = get_anal(r.source)
    # моменты времени привязаны к фактической длине интервала T_MAX
    snap_ts = [T_MAX * f for f in (0.02, 0.25, 0.5, 1.0)]
    n = 60
    xs = np.linspace(0, 1, n); ys = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(xs, ys)
    xf = xx.ravel(); yf = yy.ravel()

    refs, preds, errs = [], [], []
    for ti in snap_ts:
        tf     = np.full_like(xf, ti)
        u_ref  = anal(xf, yf, tf).reshape(n, n)
        xb = torch.tensor(xf, dtype=torch.float32, device=device).unsqueeze(1)
        yb = torch.tensor(yf, dtype=torch.float32, device=device).unsqueeze(1)
        tb = torch.tensor(tf, dtype=torch.float32, device=device).unsqueeze(1)
        u_pred = model(xb, yb, tb).squeeze(1).cpu().numpy().reshape(n, n)
        refs.append(u_ref); preds.append(u_pred)
        errs.append(np.abs(u_pred - u_ref))

    # единые шкалы: для решения — симметричная, для ошибки — общий максимум
    vmax = max(max(np.abs(a).max() for a in refs),
               max(np.abs(a).max() for a in preds), 1e-10)
    emax = max(a.max() for a in errs) or 1e-10

    fig, axes = plt.subplots(3, len(snap_ts),
                             figsize=(3.4 * len(snap_ts) + 1.2, 9.5),
                             sharex=True, sharey=True)
    row_spec = [("Аналитическое решение", refs,  "RdBu_r", -vmax, vmax),
                ("PINN",                  preds, "RdBu_r", -vmax, vmax),
                ("Модуль ошибки",         errs,  "inferno",  0.0, emax)]
    for row, (rlabel, datas, cmap, lo, hi) in enumerate(row_spec):
        im = None
        for col, (ti, data) in enumerate(zip(snap_ts, datas)):
            ax = axes[row, col]
            im = ax.contourf(xx, yy, data, levels=20, cmap=cmap,
                             vmin=lo, vmax=hi)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"$t = {ti:.0f}$")
            if col == 0:
                ax.set_ylabel(f"{rlabel}\n$y$")
            if row == 2:
                ax.set_xlabel("$x$")
        # один общий цветовой бар на строку
        cb = fig.colorbar(im, ax=axes[row, :].tolist(),
                          fraction=0.025, pad=0.02)
        cb.formatter.set_powerlimits((-2, 3))

    src_ru = _SRC_RU.get(r.source, r.source)
    ans_ru = _ANS_RU.get(r.ansatz, r.ansatz)
    fig.suptitle(f"Поля температуры и ошибки. Источник: {src_ru}; "
                 f"{ans_ru}; seed={r.seed}", fontsize=12)
    _savefig(fig, outdir / f"{r.tag}_snapshots.png")


# --- 6. Точечная ошибка во времени ---

@torch.no_grad()
def plot_error_temporal(model: PINN, r: RunResult, outdir: Path) -> None:
    anal   = get_anal(r.source)
    t_vals = np.linspace(0, T_MAX, 500)
    sensors = [(0.25, 0.25), (0.50, 0.50), (0.75, 0.75)]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (sx, sy) in enumerate(sensors):
        xv = np.full_like(t_vals, sx); yv = np.full_like(t_vals, sy)
        u_ref = anal(xv, yv, t_vals)
        xb = torch.tensor(xv, dtype=torch.float32, device=device).unsqueeze(1)
        yb = torch.tensor(yv, dtype=torch.float32, device=device).unsqueeze(1)
        tb = torch.tensor(t_vals, dtype=torch.float32, device=device).unsqueeze(1)
        u_pred = model(xb, yb, tb).squeeze(1).cpu().numpy()
        ax.semilogy(t_vals, np.abs(u_pred - u_ref) + 1e-15,
                    color=_COLORS[i % len(_COLORS)], ls=_LINE_STYLES[i], lw=1.1,
                    label=f"Точка ({sx}, {sy})")

    ax.set_xlabel("Время $t$")
    ax.set_ylabel(r"$|u_\mathrm{PINN} - u_\mathrm{точн.}|$")
    src_ru = _SRC_RU.get(r.source, r.source)
    ans_ru = _ANS_RU.get(r.ansatz, r.ansatz)
    ax.set_title(f"Точечная погрешность во времени\n"
                 f"Источник: {src_ru}, ограничения: {ans_ru}, seed={r.seed}")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    _savefig(fig, outdir / f"{r.tag}_error_temporal.png")


# --- 7. Амплитуда затухания A(t) = u(0.5, 0.5, t) ---

@torch.no_grad()
def plot_amplitude(model: PINN, r: RunResult, outdir: Path) -> None:
    anal   = get_anal(r.source)
    t_vals = np.linspace(0, T_MAX, 600)
    xv = np.full_like(t_vals, 0.5); yv = np.full_like(t_vals, 0.5)
    u_ref  = anal(xv, yv, t_vals)
    xb = torch.tensor(xv, dtype=torch.float32, device=device).unsqueeze(1)
    yb = torch.tensor(yv, dtype=torch.float32, device=device).unsqueeze(1)
    tb = torch.tensor(t_vals, dtype=torch.float32, device=device).unsqueeze(1)
    u_pred = model(xb, yb, tb).squeeze(1).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_vals, u_ref,  color=_REF_COLOR, ls="-",  lw=1.4,
                 label="Аналитическое решение")
    axes[0].plot(t_vals, u_pred, color=_PINN_COLOR, ls="--", lw=1.1,
                 label="PINN")
    axes[0].set_xlabel("Время $t$"); axes[0].set_ylabel("$u(0.5, 0.5, t)$")
    axes[0].set_title("Амплитуда в центре области")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Относительная ошибка
    rel_err = np.abs(u_pred - u_ref) / (np.abs(u_ref) + 1e-12)
    axes[1].semilogy(t_vals, rel_err + 1e-15, color=_PINN_COLOR, ls="-", lw=1.1)
    axes[1].set_xlabel("Время $t$")
    axes[1].set_ylabel("Относительная погрешность")
    axes[1].set_title("Относительная погрешность в центре области")
    axes[1].grid(True, alpha=0.3, which="both")

    src_ru = _SRC_RU.get(r.source, r.source)
    ans_ru = _ANS_RU.get(r.ansatz, r.ansatz)
    plt.suptitle(f"Источник: {src_ru},  ограничения: {ans_ru},  seed={r.seed}",
                 fontsize=10)
    plt.tight_layout()
    _savefig(fig, outdir / f"{r.tag}_amplitude.png")


# --- 8. Сравнение hard vs soft (столбчатые диаграммы) ---

def plot_hard_vs_soft(results: list[RunResult], outdir: Path) -> None:
    src_labels = [_SRC_RU.get(s, s) for s in SOURCES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(SOURCES))
    w = 0.35
    hatches = {"hard": "///", "soft": "..."}
    bar_colors = {"hard": _COLORS[0], "soft": _COLORS[1]}

    for ax, metric, ylabel in zip(axes,
                                   ["l2",   "linf"],
                                   ["Относительная погрешность L₂",
                                    "Относительная погрешность L∞"]):
        for i, ans in enumerate(ANSATZE):
            means, stds = [], []
            for src in SOURCES:
                sub = [getattr(r, metric)
                       for r in results if r.source == src and r.ansatz == ans]
                means.append(np.mean(sub) if sub else 0)
                stds.append(np.std(sub, ddof=1) if len(sub) > 1 else 0)
            ax.bar(x + i*w, means, w, yerr=stds, capsize=5,
                   color=bar_colors[ans], edgecolor="black",
                   hatch=hatches[ans], alpha=0.85,
                   label=_ANS_RU[ans], linewidth=0.8)
        ax.set_xticks(x + w/2)
        ax.set_xticklabels(src_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(f"Сравнение подходов к ограничениям\n({ylabel})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, which="both", axis="y")

    plt.tight_layout()
    _savefig(fig, outdir / "hard_vs_soft.png")


# --- 9. Ящик с усами ---

def plot_boxplot(results: list[RunResult], outdir: Path) -> None:
    from collections import defaultdict
    groups   = defaultdict(list)
    tick_lbl = []
    for src in SOURCES:
        for ans in ANSATZE:
            key = f"{_SRC_RU.get(src,src)}\n{_ANS_RU.get(ans,ans)}"
            for r in results:
                if r.source == src and r.ansatz == ans:
                    groups[key].append(r.l2)
            if groups[key]:
                tick_lbl.append(key)

    data = [groups[k] for k in tick_lbl]
    fig, ax = plt.subplots(figsize=(max(7, 2.5*len(tick_lbl)), 5))
    bp = ax.boxplot(data, labels=tick_lbl, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for j, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_COLORS[j % len(_COLORS)])
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
    ax.set_ylabel("Относительная погрешность L₂")
    ax.set_yscale("log")
    ax.set_title("Распределение погрешности по конфигурациям\n"
                 "(ящик с усами, 3 случайных инициализации)")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    plt.tight_layout()
    _savefig(fig, outdir / "boxplot_l2.png")


# --- 10. Кривые обучения по seeds ---

def plot_loss_grouped(results: list[RunResult], source: str,
                      outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, ans in zip(axes, ANSATZE):
        sub = [r for r in results if r.source == source and r.ansatz == ans]
        for i, r in enumerate(sub):
            ax.semilogy(r.loss_hist, lw=1.1,
                        color=_COLORS[i % len(_COLORS)],
                        ls=_LINE_STYLES[i % len(_LINE_STYLES)],
                        label=f"seed = {r.seed}")
        ax.axvline(FLAT_EPOCHS, color="gray", ls=":", lw=1.2,
                   label=f"NAdam → L-BFGS")
        ax.set_title(f"{_SRC_RU.get(source,source)}\n{_ANS_RU.get(ans,ans)}")
        ax.set_xlabel("Итерация")
        ax.set_ylabel("Функция потерь")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Кривые обучения для разных инициализаций\n"
                 f"Источник: {_SRC_RU.get(source,source)}", fontsize=10)
    plt.tight_layout()
    _savefig(fig, outdir / f"loss_grouped_{source}.png")


# --- 11. Сравнение двух источников у лучшего метода ---

@torch.no_grad()
def plot_sources_comparison(models: dict, results: list[RunResult],
                             ansatz: str, outdir: Path) -> None:
    t_vals = np.linspace(0, T_MAX, 500)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, src in zip(axes, SOURCES):
        best = min((r for r in results if r.source==src and r.ansatz==ansatz),
                   key=lambda r: r.l2, default=None)
        if best is None:
            continue
        model = models.get(best.tag)
        if model is None:
            continue
        anal  = get_anal(src)
        xv = np.full_like(t_vals, 0.5); yv = np.full_like(t_vals, 0.5)
        u_ref  = anal(xv, yv, t_vals)
        xb = torch.tensor(xv, dtype=torch.float32, device=device).unsqueeze(1)
        yb = torch.tensor(yv, dtype=torch.float32, device=device).unsqueeze(1)
        tb = torch.tensor(t_vals, dtype=torch.float32, device=device).unsqueeze(1)
        u_pred = model(xb, yb, tb).squeeze(1).cpu().numpy()

        ax.plot(t_vals, u_ref,  color=_REF_COLOR, ls="-",  lw=1.4,
                label="Аналитическое решение")
        ax.plot(t_vals, u_pred, color=_PINN_COLOR, ls="--", lw=1.1,
                label="PINN (лучший seed)")
        ax.set_title(_SRC_RU.get(src, src))
        ax.set_xlabel("Время $t$"); ax.set_ylabel("$u(0.5, 0.5, t)$")
        ax.legend(); ax.grid(True, alpha=0.3)
        l2_val = best.l2
        ax.text(0.97, 0.95, f"L₂ = {l2_val:.2e}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    plt.suptitle(f"Сравнение источников тепловыделения\n"
                 f"Ограничения: {_ANS_RU.get(ansatz, ansatz)}", fontsize=10)
    plt.tight_layout()
    _savefig(fig, outdir / f"sources_comparison_{ansatz}.png")


# --- 12. Спектр сигнала (FFT) в центре области ---

@torch.no_grad()
def plot_spectrum(model: PINN, r: RunResult, outdir: Path) -> None:
    """Амплитудный спектр u(0.5,0.5,t): проверка, поймала ли сеть
    собственную частоту ω₀ (ключевой вопрос спектральной предвзятости)."""
    anal   = get_anal(r.source)
    n = 4000
    t_vals = np.linspace(0, T_MAX, n)
    xv = np.full_like(t_vals, 0.5); yv = np.full_like(t_vals, 0.5)
    u_ref = anal(xv, yv, t_vals)
    xb = torch.tensor(xv, dtype=torch.float32, device=device).unsqueeze(1)
    yb = torch.tensor(yv, dtype=torch.float32, device=device).unsqueeze(1)
    tb = torch.tensor(t_vals, dtype=torch.float32, device=device).unsqueeze(1)
    u_pred = model(xb, yb, tb).squeeze(1).cpu().numpy()

    dt = t_vals[1] - t_vals[0]
    freq = np.fft.rfftfreq(n, d=dt) * 2 * PI          # угловая частота
    A_ref  = np.abs(np.fft.rfft(u_ref))  * 2 / n
    A_pred = np.abs(np.fft.rfft(u_pred)) * 2 / n

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freq, A_ref,  color=_REF_COLOR, ls="-",  lw=1.4,
            label="Аналитическое решение")
    ax.plot(freq, A_pred, color=_PINN_COLOR, ls="--", lw=1.1, label="PINN")
    ax.axvline(OMEGA0, color="gray", ls=":", lw=1.2,
               label=f"ω₀ = π√2 ≈ {OMEGA0:.2f}")
    ax.set_xlim(0, 6 * OMEGA0)
    ax.set_xlabel("Угловая частота ω")
    ax.set_ylabel("Амплитуда спектра")
    ax.set_title(f"Спектр сигнала в центре области\n"
                 f"Источник: {_SRC_RU.get(r.source, r.source)}, "
                 f"{_ANS_RU.get(r.ansatz, r.ansatz)}, seed={r.seed}")
    ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, outdir / f"{r.tag}_spectrum.png")


# --- 13. Пространственная L2-ошибка как функция времени ---

@torch.no_grad()
def plot_l2_in_time(model: PINN, r: RunResult, outdir: Path) -> None:
    """Относительная L2-ошибка поля на пространственном срезе в зависимости
    от t — показывает накопление ошибки на длинном интервале."""
    anal = get_anal(r.source)
    n_sp = 30
    xs = np.linspace(0, 1, n_sp); ys = np.linspace(0, 1, n_sp)
    xx, yy = np.meshgrid(xs, ys)
    xf, yf = xx.ravel(), yy.ravel()
    t_slices = np.linspace(0, T_MAX, 60)
    l2t = []
    xb = torch.tensor(xf, dtype=torch.float32, device=device).unsqueeze(1)
    yb = torch.tensor(yf, dtype=torch.float32, device=device).unsqueeze(1)
    for ti in t_slices:
        tf = np.full_like(xf, ti)
        u_ref = anal(xf, yf, tf)
        tb = torch.tensor(tf, dtype=torch.float32, device=device).unsqueeze(1)
        u_pred = model(xb, yb, tb).squeeze(1).cpu().numpy()
        denom = np.sqrt(np.mean(u_ref**2)) + 1e-12
        l2t.append(np.sqrt(np.mean((u_pred - u_ref)**2)) / denom)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_slices, l2t, color=_PINN_COLOR, ls="-", lw=1.3, marker="o", ms=3)
    ax.set_xlabel("Время t")
    ax.set_ylabel("Относительная L₂-ошибка на срезе")
    ax.set_title(f"Накопление ошибки во времени\n"
                 f"Источник: {_SRC_RU.get(r.source, r.source)}, "
                 f"{_ANS_RU.get(r.ansatz, r.ansatz)}, seed={r.seed}")
    ax.grid(True, alpha=0.3)
    _savefig(fig, outdir / f"{r.tag}_l2_in_time.png")


# --- 14. Прямое наложение hard vs soft (один источник) ---

@torch.no_grad()
def plot_hard_vs_soft_overlay(models: dict, results: list[RunResult],
                              source: str, outdir: Path) -> None:
    """Профиль u(0.5,0.5,t): аналитика vs лучший hard vs лучший soft."""
    t_vals = np.linspace(0, T_MAX, 800)
    anal = get_anal(source)
    xv = np.full_like(t_vals, 0.5); yv = np.full_like(t_vals, 0.5)
    u_ref = anal(xv, yv, t_vals)
    xb = torch.tensor(xv, dtype=torch.float32, device=device).unsqueeze(1)
    yb = torch.tensor(yv, dtype=torch.float32, device=device).unsqueeze(1)
    tb = torch.tensor(t_vals, dtype=torch.float32, device=device).unsqueeze(1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_vals, u_ref, color="black", ls="-", lw=1.5,
            label="Аналитическое решение")
    style = {"hard": (_REF_COLOR, "--"), "soft": (_PINN_COLOR, "-.")}
    for ans in ANSATZE:
        best = min((r for r in results
                    if r.source == source and r.ansatz == ans),
                   key=lambda r: r.l2, default=None)
        if best is None or best.tag not in models:
            continue
        up = models[best.tag](xb, yb, tb).squeeze(1).cpu().numpy()
        c, ls = style[ans]
        ax.plot(t_vals, up, color=c, ls=ls, lw=1.1,
                label=f"{_ANS_RU[ans]} (L₂={best.l2:.2e})")
    ax.set_xlabel("Время t"); ax.set_ylabel("u(0.5, 0.5, t)")
    ax.set_title(f"Сравнение жёсткого и мягкого представления решения\n"
                 f"Источник: {_SRC_RU.get(source, source)}")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    _savefig(fig, outdir / f"hard_vs_soft_overlay_{source}.png")


# --- 15. Время обучения по конфигурациям ---

def plot_training_time(results: list[RunResult], outdir: Path) -> None:
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[(r.source, r.ansatz)].append(r.time_s)
    keys = [(s, a) for s in SOURCES for a in ANSATZE if (s, a) in agg]
    labels = [f"{_SRC_RU.get(s,s)}\n{_ANS_RU.get(a,a)}" for s, a in keys]
    means = [np.mean(agg[k]) for k in keys]
    stds  = [np.std(agg[k], ddof=1) if len(agg[k]) > 1 else 0 for k in keys]
    fig, ax = plt.subplots(figsize=(max(7, 2.2*len(keys)), 4.5))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=5,
                  color=[_COLORS[i % len(_COLORS)] for i in range(len(keys))],
                  edgecolor="black", alpha=0.85)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Время обучения, с")
    ax.set_title("Время обучения по конфигурациям\n(среднее по seeds)")
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, outdir / "training_time.png")


# --- 16. Ящик с усами для L∞ ---

def plot_boxplot_linf(results: list[RunResult], outdir: Path) -> None:
    from collections import defaultdict
    groups = defaultdict(list); tick_lbl = []
    for src in SOURCES:
        for ans in ANSATZE:
            key = f"{_SRC_RU.get(src,src)}\n{_ANS_RU.get(ans,ans)}"
            for r in results:
                if r.source == src and r.ansatz == ans:
                    groups[key].append(r.linf)
            if groups[key]:
                tick_lbl.append(key)
    data = [groups[k] for k in tick_lbl]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(7, 2.5*len(tick_lbl)), 5))
    bp = ax.boxplot(data, labels=tick_lbl, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for j, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_COLORS[j % len(_COLORS)])
        patch.set_alpha(0.6); patch.set_edgecolor("black")
    ax.set_ylabel("Относительная погрешность L∞")
    ax.set_yscale("log")
    ax.set_title("Распределение погрешности L∞ по конфигурациям\n"
                 "(ящик с усами, 3 случайных инициализации)")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    plt.tight_layout()
    _savefig(fig, outdir / "boxplot_linf.png")


# --- 17. Зависимость качества/скорости от размера сети (Optuna grid) ---

def plot_arch_study(trials: list[dict], source: str, outdir: Path,
                    best_quality: dict | None = None,
                    best_tradeoff: dict | None = None) -> None:
    """trials: список словарей {params, hidden, blocks, l2, time}.
    Строит L2(число параметров) и время(число параметров); отмечает
    лучшую по качеству и лучшую по соотношению качество/скорость схемы."""
    if not trials:
        return
    ts = sorted(trials, key=lambda d: d["params"])
    pars = np.array([d["params"] for d in ts])
    l2   = np.array([d["l2"]     for d in ts])
    tm   = np.array([d["time"]   for d in ts])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # L2 vs параметры; цвет по глубине
    for nb in sorted(set(d["blocks"] for d in ts)):
        sel = [d for d in ts if d["blocks"] == nb]
        ax1.scatter([d["params"] for d in sel], [d["l2"] for d in sel],
                    s=45, color=_COLORS[(nb - 1) % len(_COLORS)],
                    edgecolor="black", linewidth=0.5,
                    label=f"{nb} блок(ов)", zorder=3)
    order = np.argsort(pars)
    ax1.plot(pars[order], np.minimum.accumulate(l2[order]),
             color="gray", ls="--", lw=1.2, label="огибающая (лучшее)")

    def _same(a, b):
        return (a is not None and b is not None
                and a.get("hidden") == b.get("hidden")
                and a.get("blocks") == b.get("blocks"))

    def _mark(ax, d, marker, lbl, size=240):
        if d is not None:
            ax.scatter([d["params"]], [d["l2"]], s=size, marker=marker,
                       facecolor="none", edgecolor="red", linewidth=1.8,
                       zorder=5, label=lbl)
    if _same(best_quality, best_tradeoff):
        _mark(ax1, best_quality, "*",
              "лучшая по качеству и скорости", size=320)
    else:
        _mark(ax1, best_quality,  "o", "лучшая по качеству")
        _mark(ax1, best_tradeoff, "s", "лучшая качество/скорость")

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Число параметров сети")
    ax1.set_ylabel("Относительная погрешность L₂")
    ax1.set_title("Точность от размера сети")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, which="both")

    # Время vs параметры
    ax2.scatter(pars, tm, s=45, color=_PINN_COLOR,
                edgecolor="black", linewidth=0.5, zorder=3)
    if best_quality is not None:
        ax2.scatter([best_quality["params"]], [best_quality["time"]], s=240,
                    marker="o", facecolor="none", edgecolor="red", linewidth=1.8)
    if best_tradeoff is not None and not _same(best_quality, best_tradeoff):
        ax2.scatter([best_tradeoff["params"]], [best_tradeoff["time"]], s=240,
                    marker="s", facecolor="none", edgecolor="red", linewidth=1.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Число параметров сети")
    ax2.set_ylabel("Время обучения, с")
    ax2.set_title("Скорость от размера сети")
    ax2.grid(True, alpha=0.3, which="both")

    plt.suptitle(f"Исследование размера архитектуры (жёсткое представление решения)\n"
                 f"Источник: {_SRC_RU.get(source, source)}", fontsize=10)
    plt.tight_layout()
    _savefig(fig, outdir / f"arch_study_{source}.png")


# --- 18. История байесовского подбора LR (Optuna TPE) ---

def plot_optuna_history(values: list[float], best_lr: float,
                        source: str, hidden: int, n_blocks: int,
                        outdir: Path) -> None:
    if not values:
        return
    best_so_far = np.minimum.accumulate(values)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, len(values) + 1), values, "o",
                color=_COLORS[0], alpha=0.6, ms=5, label="Значение испытания")
    ax.semilogy(range(1, len(values) + 1), best_so_far,
                color=_PINN_COLOR, ls="-", lw=2, label="Лучший результат")
    ax.set_xlabel("Номер испытания (trial)")
    ax.set_ylabel("Относительная погрешность L₂ (валидация)")
    ax.set_title(f"Байесовский подбор скорости обучения (Optuna TPE)\n"
                 f"Источник: {_SRC_RU.get(source, source)}, "
                 f"лучшая схема h={hidden}, blocks={n_blocks}, "
                 f"LR*={best_lr:.2e}")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    _savefig(fig, outdir / f"optuna_lr_history_{source}.png")


# --- 19. Зависимость точности от длины интервала T (hard vs soft) ---

def plot_l2_vs_T(results: list[RunResult], source: str, outdir: Path) -> None:
    """Средняя L2 (± разброс по seeds) в зависимости от T для каждого представления решения."""
    from collections import defaultdict
    style = {"hard": (_REF_COLOR, "-", "o"), "soft": (_PINN_COLOR, "--", "s")}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_data = False
    for ans in ANSATZE:
        agg = defaultdict(list)
        for r in results:
            if r.source == source and r.ansatz == ans:
                agg[r.t_max].append(r.l2)
        if not agg:
            continue
        any_data = True
        Ts = sorted(agg)
        means = [float(np.mean(agg[t])) for t in Ts]
        stds  = [float(np.std(agg[t], ddof=1)) if len(agg[t]) > 1 else 0.0
                 for t in Ts]
        c, ls, mk = style.get(ans, (_COLORS[0], "-", "o"))
        ax.errorbar(Ts, means, yerr=stds, color=c, ls=ls, marker=mk,
                    lw=1.4, capsize=4, label=_ANS_RU.get(ans, ans))
    if not any_data:
        plt.close(fig); return
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Длина интервала T")
    ax.set_ylabel("Относительная погрешность L₂")
    ax.set_title(f"Зависимость точности от длины интервала\n"
                 f"Источник: {_SRC_RU.get(source, source)} "
                 f"(среднее ± σ по seeds)")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    _savefig(fig, outdir / f"l2_vs_T_{source}.png")


# --- 20. 3D-поверхность u(x,y,t*) ---

@torch.no_grad()
def plot_surface_3d(model: PINN, r: RunResult, outdir: Path) -> None:
    """3D-поверхности u(x,y,t*) для четырёх моментов времени:
    верхний ряд — аналитика, нижний — PINN. Общий масштаб z и цвета."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    anal = get_anal(r.source)
    times = [T_MAX * f for f in (0.02, 0.25, 0.5, 1.0)]
    n = 40
    xs = np.linspace(0, 1, n); ys = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(xs, ys)
    xf = xx.ravel(); yf = yy.ravel()

    refs, preds = [], []
    for ti in times:
        tf = np.full_like(xf, ti)
        u_ref = anal(xf, yf, tf).reshape(n, n)
        xb = torch.tensor(xf, dtype=torch.float32, device=device).unsqueeze(1)
        yb = torch.tensor(yf, dtype=torch.float32, device=device).unsqueeze(1)
        tb = torch.tensor(tf, dtype=torch.float32, device=device).unsqueeze(1)
        u_pred = model(xb, yb, tb).squeeze(1).cpu().numpy().reshape(n, n)
        refs.append(u_ref); preds.append(u_pred)
    vmax = max(max(np.abs(a).max() for a in refs),
               max(np.abs(a).max() for a in preds), 1e-10)

    fig = plt.figure(figsize=(4.0 * len(times), 7.5))
    for col, ti in enumerate(times):
        for row, (data, label) in enumerate([(refs[col],  "Аналитическое"),
                                             (preds[col], "PINN")]):
            ax = fig.add_subplot(2, len(times), row * len(times) + col + 1,
                                 projection="3d")
            ax.plot_surface(xx, yy, data, cmap="viridis",
                            vmin=-vmax, vmax=vmax,
                            rstride=1, cstride=1,
                            linewidth=0, antialiased=True, alpha=0.95)
            ax.set_zlim(-vmax, vmax)
            ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
            ax.set_zlabel("$u$")
            if row == 0:
                ax.set_title(f"$t = {ti:.1f}$")
            if col == 0:
                ax.text2D(-0.05, 0.5, label, transform=ax.transAxes,
                          rotation=90, va="center", fontsize=11)

    src_ru = _SRC_RU.get(r.source, r.source)
    ans_ru = _ANS_RU.get(r.ansatz, r.ansatz)
    fig.suptitle(f"3D-поверхности решения. Источник: {src_ru}; "
                 f"{ans_ru}; seed={r.seed}", fontsize=12)
    fig.tight_layout()
    _savefig(fig, outdir / f"{r.tag}_surface3d.png")


# --- 21. Пространственно-временной срез u(x, y=0.5, t) ---

@torch.no_grad()
def plot_xt_slice(model: PINN, r: RunResult, outdir: Path) -> None:
    """Heatmap u(x, y=0.5, t) на плоскости (x, t): аналитика / PINN / |ошибка|."""
    anal = get_anal(r.source)
    nx, nt = 80, 200
    xs = np.linspace(0, 1, nx)
    ts = np.linspace(0, T_MAX, nt)
    xx, tt = np.meshgrid(xs, ts, indexing="xy")     # (nt, nx)
    yy = np.full_like(xx, 0.5)
    u_ref = anal(xx.ravel(), yy.ravel(), tt.ravel()).reshape(nt, nx)
    xb = torch.tensor(xx.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
    yb = torch.tensor(yy.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
    tb = torch.tensor(tt.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
    u_pred = model(xb, yb, tb).squeeze(1).cpu().numpy().reshape(nt, nx)
    err = np.abs(u_pred - u_ref)

    vmax = max(np.abs(u_ref).max(), np.abs(u_pred).max(), 1e-10)
    emax = err.max() or 1e-10

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    extents = [0, 1, 0, T_MAX]
    panels = [(u_ref,  "Аналитическое решение", "RdBu_r", -vmax, vmax),
              (u_pred, "PINN",                  "RdBu_r", -vmax, vmax),
              (err,    "Модуль ошибки",         "inferno",  0,   emax)]
    for ax, (data, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(data, extent=extents, origin="lower",
                       aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_xlabel("$x$")
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.formatter.set_powerlimits((-2, 3))
    axes[0].set_ylabel("Время $t$")

    src_ru = _SRC_RU.get(r.source, r.source)
    ans_ru = _ANS_RU.get(r.ansatz, r.ansatz)
    fig.suptitle(f"Пространственно-временной срез $u(x, y{{=}}0.5, t)$. "
                 f"Источник: {src_ru}; {ans_ru}; seed={r.seed}", fontsize=12)
    fig.tight_layout()
    _savefig(fig, outdir / f"{r.tag}_xt_slice.png")


# --- 22. 3D-поверхность поля ошибки при t = T_MAX ---

@torch.no_grad()
def plot_error_surface_3d(model: PINN, r: RunResult, outdir: Path) -> None:
    """3D-поверхность |u_PINN - u_точн| на конечный момент t=T_MAX."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    anal = get_anal(r.source)
    n = 60
    xs = np.linspace(0, 1, n); ys = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(xs, ys)
    xf = xx.ravel(); yf = yy.ravel(); tf = np.full_like(xf, T_MAX)
    u_ref = anal(xf, yf, tf).reshape(n, n)
    xb = torch.tensor(xf, dtype=torch.float32, device=device).unsqueeze(1)
    yb = torch.tensor(yf, dtype=torch.float32, device=device).unsqueeze(1)
    tb = torch.tensor(tf, dtype=torch.float32, device=device).unsqueeze(1)
    u_pred = model(xb, yb, tb).squeeze(1).cpu().numpy().reshape(n, n)
    err = np.abs(u_pred - u_ref)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, err, cmap="inferno",
                            rstride=1, cstride=1,
                            linewidth=0, antialiased=True)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_zlabel(r"$|u_\mathrm{PINN} - u_\mathrm{точн}|$")
    src_ru = _SRC_RU.get(r.source, r.source)
    ans_ru = _ANS_RU.get(r.ansatz, r.ansatz)
    ax.set_title(f"Поле модуля ошибки при $t = T = {T_MAX:.0f}$\n"
                 f"Источник: {src_ru}; {ans_ru}; seed={r.seed}")
    cb = fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1)
    cb.formatter.set_powerlimits((-2, 3))
    _savefig(fig, outdir / f"{r.tag}_err_surface3d.png")


# --- 23. Линии уровня решения для четырёх моментов времени ---

@torch.no_grad()
def plot_contour_levels(model: PINN, r: RunResult, outdir: Path) -> None:
    """Изолинии u(x,y)=const на 4 момента времени: верхний ряд — аналитика,
    нижний — PINN. Уровни общие для всех панелей."""
    anal = get_anal(r.source)
    times = [T_MAX * f for f in (0.02, 0.25, 0.5, 1.0)]
    n = 80
    xs = np.linspace(0, 1, n); ys = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(xs, ys)
    xf = xx.ravel(); yf = yy.ravel()

    refs, preds = [], []
    for ti in times:
        tf = np.full_like(xf, ti)
        u_ref = anal(xf, yf, tf).reshape(n, n)
        xb = torch.tensor(xf, dtype=torch.float32, device=device).unsqueeze(1)
        yb = torch.tensor(yf, dtype=torch.float32, device=device).unsqueeze(1)
        tb = torch.tensor(tf, dtype=torch.float32, device=device).unsqueeze(1)
        u_pred = model(xb, yb, tb).squeeze(1).cpu().numpy().reshape(n, n)
        refs.append(u_ref); preds.append(u_pred)
    vmax = max(max(np.abs(a).max() for a in refs),
               max(np.abs(a).max() for a in preds), 1e-10)
    levels = np.linspace(-vmax, vmax, 11)

    fig, axes = plt.subplots(2, len(times),
                             figsize=(3.4 * len(times) + 0.6, 7),
                             sharex=True, sharey=True)
    for col, ti in enumerate(times):
        for row, (data, label) in enumerate([(refs[col],  "Аналитическое"),
                                             (preds[col], "PINN")]):
            ax = axes[row, col]
            cs = ax.contour(xx, yy, data, levels=levels,
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            linewidths=1.1)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"$t = {ti:.1f}$")
            if col == 0:
                ax.set_ylabel(f"{label}\n$y$")
            if row == 1:
                ax.set_xlabel("$x$")

    src_ru = _SRC_RU.get(r.source, r.source)
    ans_ru = _ANS_RU.get(r.ansatz, r.ansatz)
    fig.suptitle(f"Линии уровня решения. Источник: {src_ru}; "
                 f"{ans_ru}; seed={r.seed}", fontsize=12)
    fig.tight_layout()
    _savefig(fig, outdir / f"{r.tag}_contour_levels.png")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def run_statistics(results: list[RunResult], outdir: Path) -> None:
    lines = ["="*60, "STATISTICAL REPORT — FORWARD PROBLEM", "="*60, ""]

    for src in SOURCES:
        for ans in ANSATZE:
            vals = [r.l2 for r in results if r.source==src and r.ansatz==ans]
            if not vals:
                continue
            arr = np.array(vals)
            lines.append(f"[{src} / {ans}]  n={len(arr)}")
            lines.append(f"  mean L2 = {arr.mean():.4e}  std = {arr.std(ddof=1):.4e}")
            lines.append(f"  min  L2 = {arr.min():.4e}  max = {arr.max():.4e}")
            if len(arr) >= 3:
                sw = stats.shapiro(arr)
                lines.append(f"  Shapiro-Wilk: stat={sw.statistic:.4f} p={sw.pvalue:.4f}")
            lines.append("")

    for src in SOURCES:
        hard = np.array([r.l2 for r in results if r.source==src and r.ansatz=="hard"])
        soft = np.array([r.l2 for r in results if r.source==src and r.ansatz=="soft"])
        if len(hard) >= 2 and len(soft) >= 2:
            w = stats.ttest_ind(hard, soft, equal_var=False)
            m = stats.mannwhitneyu(hard, soft, alternative="two-sided")
            lines.append(f"[{src}] Hard vs Soft")
            lines.append(f"  Welch t-test: stat={w.statistic:.4f} p={w.pvalue:.4f}")
            lines.append(f"  Mann-Whitney: stat={m.statistic:.1f} p={m.pvalue:.4f}")
            lines.append("")

    txt = "\n".join(lines)
    for line in lines:
        log(line)
    (outdir / "stats_report.txt").write_text(txt, encoding="utf-8")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _studyA_stats(results: list[RunResult], outdir: Path) -> None:
    """Статистика hard vs soft по каждому (источник, T)."""
    from collections import defaultdict
    lines = ["="*60, "СТАТИСТИКА: hard vs soft по интервалам T", "="*60, ""]
    grp: dict = defaultdict(lambda: defaultdict(list))
    for r in results:
        grp[(r.source, r.t_max)][r.ansatz].append(r.l2)
    for (src, T) in sorted(grp, key=lambda k: (k[0], k[1])):
        hard = np.array(grp[(src, T)].get("hard", []))
        soft = np.array(grp[(src, T)].get("soft", []))
        lines.append(f"[{src}, T={T:g}]")
        if len(hard):
            lines.append(f"  hard: mean={hard.mean():.3e} std="
                         f"{hard.std(ddof=1) if len(hard)>1 else 0:.2e} n={len(hard)}")
        if len(soft):
            lines.append(f"  soft: mean={soft.mean():.3e} std="
                         f"{soft.std(ddof=1) if len(soft)>1 else 0:.2e} n={len(soft)}")
        if len(hard) >= 2 and len(soft) >= 2:
            w = stats.ttest_ind(hard, soft, equal_var=False)
            lines.append(f"  Welch t-test (hard vs soft): "
                         f"stat={w.statistic:.3f} p={w.pvalue:.4f}")
        lines.append("")
    txt = "\n".join(lines)
    for ln in lines:
        log(ln)
    (outdir / "stats_studyA.txt").write_text(txt, encoding="utf-8")


def main() -> None:
    global T_MAX
    log(""); log("="*60)
    log("FORWARD PROBLEM — PINN DIPLOMA")
    log("Исследования: A) hard vs soft и зависимость от интервала T;")
    log("              B) размер сети (Optuna Grid) + TPE-подбор LR.")
    log(f"T_LIST={T_LIST}  Источники={SOURCES}  Представления={ANSATZE}  Seeds={SEEDS}")
    log(f"FLAT_EPOCHS={FLAT_EPOCHS}  LBFGS={LBFGS_STEPS}  N_PDE={N_PDE}  "
        f"KAPPA={KAPPA}  n_freq={N_FREQ}")
    log("="*60)

    if not HAS_OPTUNA:
        log("  ВНИМАНИЕ: optuna не установлена — исследование B пропустится.")

    outdir = ROOT_DIR

    # =====================================================================
    # ИССЛЕДОВАНИЕ A: hard vs soft × длина интервала T × seeds
    # =====================================================================
    log(""); log("#"*60)
    log("ИССЛЕДОВАНИЕ A — hard vs soft и зависимость от интервала T")
    log("#"*60)
    studyA: list[RunResult] = []
    models_A: dict = {}                       # tag → модель (только seed[0])
    for src in SOURCES:
        for t_max in T_LIST:
            get_anal(src, t_max)              # эталон под этот T
            for ansatz in ANSATZE:
                for seed in SEEDS:
                    tag = f"A_{src}_{ansatz}_T{int(t_max)}_s{seed}"
                    if tag in _done:
                        log(f"  SKIP {tag}"); continue
                    r, m = train_one(tag, src, ansatz, seed, t_max=t_max)
                    studyA.append(r)
                    _save("study_T", r.slim())
                    # Полный набор графиков — только для первого seed (экономия)
                    if seed == SEEDS[0]:
                        models_A[r.tag] = m
                        plot_loss(r, outdir)
                        plot_temporal(m, r, outdir)
                        plot_amplitude(m, r, outdir)
                        plot_spectrum(m, r, outdir)
                        plot_error_temporal(m, r, outdir)
                        plot_snapshots(m, r, outdir)
                        plot_surface_3d(m, r, outdir)
                        plot_xt_slice(m, r, outdir)
                        plot_error_surface_3d(m, r, outdir)
                        plot_contour_levels(m, r, outdir)
                        if ansatz == "soft":
                            plot_loss_components(r, outdir)
                            plot_soft_weights(r, outdir)

    # Сводные графики и статистика исследования A
    for src in SOURCES:
        plot_l2_vs_T(studyA, src, outdir)
        plot_loss_grouped(studyA, src, outdir)
    if studyA:
        plot_hard_vs_soft(studyA, outdir)
        plot_boxplot(studyA, outdir)
        plot_boxplot_linf(studyA, outdir)
        plot_training_time(studyA, outdir)
        # Качественные сравнения профилей — на представительном T (наибольшем)
        T_rep = max(T_LIST)
        studyA_rep = [r for r in studyA if abs(r.t_max - T_rep) < 1e-9]
        if studyA_rep:
            T_MAX = float(T_rep)
            for src in SOURCES:
                plot_hard_vs_soft_overlay(models_A, studyA_rep, src, outdir)
            for ans in ANSATZE:
                plot_sources_comparison(models_A, studyA_rep, ans, outdir)
        _studyA_stats(studyA, outdir)

    # =====================================================================
    # ИССЛЕДОВАНИЕ B: размер сети (Optuna Grid) на одном T, hard
    # =====================================================================
    if HAS_OPTUNA:
        log(""); log("#"*60)
        log(f"ИССЛЕДОВАНИЕ B — размер сети, источник={SIZE_SRC}, T={SIZE_T:g}, "
            f"представление=hard")
        log("#"*60)
        T_MAX = float(SIZE_T)                 # глобально на исследование B
        get_anal(SIZE_SRC, SIZE_T)
        SEED = SEEDS[0]
        trial_records: list[dict] = []

        def objective(trial, _rec=trial_records):
            hidden   = trial.suggest_categorical("hidden", ARCH_HIDDEN)
            n_blocks = trial.suggest_int("n_blocks",
                                         ARCH_BLOCKS[0], ARCH_BLOCKS[-1])
            npar = count_params(hidden, n_blocks)
            torch.manual_seed(SEED); np.random.seed(SEED)
            m = PINN(SIZE_SRC, ansatz="hard", hidden=hidden,
                     n_blocks=n_blocks).to(device)
            opt = optim.NAdam(m.parameters(), lr=LR_DEFAULT)
            t0 = time.time()
            for _ in range(OPTUNA_EPOCHS):
                opt.zero_grad()
                loss, _ = compute_loss(m)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            dt = time.time() - t0
            l2, _ = field_metrics(m, n_space=30, n_time=20)
            _rec.append({"params": npar, "hidden": hidden,
                         "blocks": n_blocks, "l2": float(l2), "time": dt})
            log(f"  trial {trial.number+1:2d}/{len(ARCH_HIDDEN)*len(ARCH_BLOCKS):2d}  "
                f"h={hidden:>3d} b={n_blocks}  p={npar:>6,d}  "
                f"L2={float(l2):.3e}  t={dt:.0f}s")
            return l2

        ss = {"hidden": ARCH_HIDDEN, "n_blocks": ARCH_BLOCKS}
        n_grid = len(ARCH_HIDDEN) * len(ARCH_BLOCKS)
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.GridSampler(ss))
        log(f"  Перебор {n_grid} конфигураций × {OPTUNA_EPOCHS} эпох ...")
        study.optimize(objective, n_trials=n_grid)

        if trial_records:
            best_q = min(trial_records, key=lambda d: d["l2"])
            TRADEOFF_TOL = 1.5
            near = [d for d in trial_records
                    if d["l2"] <= best_q["l2"] * TRADEOFF_TOL]
            best_qs = min(near, key=lambda d: d["time"])
            _save("arch_trials", {"tag": f"arch_{SIZE_SRC}", "source": SIZE_SRC,
                                  "t_max": SIZE_T, "l2": float(best_q["l2"]),
                                  "best_quality": best_q,
                                  "best_tradeoff": best_qs,
                                  "trials": trial_records})
            plot_arch_study(trial_records, SIZE_SRC, outdir,
                            best_quality=best_q, best_tradeoff=best_qs)
            log(f"  [по качеству]       h={best_q['hidden']} b={best_q['blocks']} "
                f"p={best_q['params']:,} L2={best_q['l2']:.3e} t={best_q['time']:.0f}s")
            log(f"  [качество/скорость] h={best_qs['hidden']} b={best_qs['blocks']} "
                f"p={best_qs['params']:,} L2={best_qs['l2']:.3e} t={best_qs['time']:.0f}s")

            # TPE-подбор LR на лучшей по качеству схеме
            bqh, bqn = best_q["hidden"], best_q["blocks"]
            lr_vals: list[float] = []

            def objective_lr(trial, _h=bqh, _b=bqn, _v=lr_vals):
                lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
                torch.manual_seed(SEED); np.random.seed(SEED)
                m = PINN(SIZE_SRC, ansatz="hard", hidden=_h,
                         n_blocks=_b).to(device)
                opt = optim.NAdam(m.parameters(), lr=lr)
                t0 = time.time()
                for _ in range(LR_EPOCHS):
                    opt.zero_grad()
                    loss, _ = compute_loss(m)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                dt = time.time() - t0
                l2, _ = field_metrics(m, n_space=30, n_time=20)
                _v.append(float(l2))
                log(f"  trial {trial.number+1:2d}/{LR_TRIALS:2d}  "
                    f"lr={lr:.3e}  L2={float(l2):.3e}  t={dt:.0f}s")
                return l2

            log(f"  TPE-подбор LR на h={bqh}, b={bqn} "
                f"({LR_TRIALS}×{LR_EPOCHS}) ...")
            study_lr = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=4))
            study_lr.optimize(objective_lr, n_trials=LR_TRIALS)
            best_lr = study_lr.best_params["lr"]
            log(f"  Лучший LR={best_lr:.4e} (L2={study_lr.best_value:.3e})")
            plot_optuna_history(lr_vals, best_lr, SIZE_SRC, bqh, bqn, outdir)
            _save("lr_search", {"tag": f"lr_{SIZE_SRC}", "source": SIZE_SRC,
                                "hidden": bqh, "blocks": bqn,
                                "best_lr": float(best_lr),
                                "l2": float(study_lr.best_value),
                                "values": lr_vals})

            # Финальные полные прогоны: лучшая по качеству (с LR*) и компромисс
            picks = [("quality", best_q, best_lr)]
            if (best_qs["hidden"], best_qs["blocks"]) != (bqh, bqn):
                picks.append(("tradeoff", best_qs, LR_DEFAULT))
            for kind, cfg, lr_use in picks:
                ftag = f"final_{kind}_{SIZE_SRC}_h{cfg['hidden']}_b{cfg['blocks']}"
                if ftag in _done:
                    continue
                r_f, m_f = train_one(ftag, SIZE_SRC, "hard", SEED, lr=lr_use,
                                     hidden=cfg["hidden"], n_blocks=cfg["blocks"],
                                     t_max=SIZE_T)
                r_f.tag = ftag
                _save("final", r_f.slim())
                plot_loss(r_f, outdir); plot_temporal(m_f, r_f, outdir)
                plot_error_temporal(m_f, r_f, outdir); plot_amplitude(m_f, r_f, outdir)
                plot_snapshots(m_f, r_f, outdir); plot_spectrum(m_f, r_f, outdir)
                plot_l2_in_time(m_f, r_f, outdir)
                plot_surface_3d(m_f, r_f, outdir)
                plot_xt_slice(m_f, r_f, outdir)
                plot_error_surface_3d(m_f, r_f, outdir)
                plot_contour_levels(m_f, r_f, outdir)
                log(f"  Финал [{kind}]: L2={r_f.l2:.3e} Linf={r_f.linf:.3e}")

    # =========================================================
    # Summary
    # =========================================================
    log(""); log("="*60); log("SUMMARY")
    if _METRICS.exists():
        data = json.loads(_METRICS.read_text())
        for cat, runs in data.items():
            l2s = [r["l2"] for r in runs if isinstance(r, dict) and "l2" in r]
            if l2s:
                log(f"  {cat:14s} n={len(l2s)}  "
                    f"L2 mean={np.mean(l2s):.3e}  best={np.min(l2s):.3e}")
    log(f"Results: {ROOT_DIR}"); log("Done.")


if __name__ == "__main__":
    main()
