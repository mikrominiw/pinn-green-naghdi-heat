# -*- coding: utf-8 -*-
"""
Классический численный решатель прямой задачи Грина–Нагди III типа
(метод прямых: конечные разности по пространству + интегрирование по времени).

Уравнение (безразмерное):
    u_tt - κ (u_xxt + u_yyt) - (u_xx + u_yy) = f,
    f = 0 (однородный, q=0) или f = 2x (неоднородный, q=2xt),
    Ω = (0,1)², u|∂Ω = 0, u(x,y,0)=sin(πx)sin(πy), u_t(x,y,0)=0.

Сведение к системе 1-го порядка. Пусть v = u_t. Так как
    u_xxt + u_yyt = ∂/∂t (Δu) = Δv,  u_xx + u_yy = Δu,
получаем
    u_t = v,
    v_t = κ Δv + Δu + f.
После дискретизации лапласиана L (5-точечная схема, Дирихле) система линейна:
    d/dt [u; v] = A [u; v] + b,   A = [[0, I], [L, κL]],  b = [0; f].
Интегрируется solve_ivp (BDF, постоянный якобиан A — быстро и устойчиво).

Цель файла — измерить ВРЕМЯ численного решения для сравнения с PINN.
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse import identity, kron, bmat
from scipy.integrate import solve_ivp

PI = math.pi
KAPPA = 1.2e-4


def laplacian_2d(N: int, h: float) -> sp.csr_matrix:
    """5-точечный лапласиан на сетке N×N внутренних узлов, Дирихле."""
    main = -2.0 * np.ones(N)
    off = np.ones(N - 1)
    D2 = sp.diags([off, main, off], [-1, 0, 1]) / h**2
    I = identity(N)
    return (kron(D2, I) + kron(I, D2)).tocsr()


def build_system(N: int, source: str):
    """Возвращает (A, b, X, Y, u0) для системы d/dt[u;v] = A[u;v] + b."""
    h = 1.0 / (N + 1)
    xs = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    Xf, Yf = X.ravel(), Y.ravel()
    n = N * N

    L = laplacian_2d(N, h)
    I = identity(n)
    Z = sp.csr_matrix((n, n))
    A = bmat([[Z, I], [L, KAPPA * L]]).tocsr()

    f = (2.0 * Xf) if source == "inhomo" else np.zeros(n)
    b = np.concatenate([np.zeros(n), f])

    u0 = (np.sin(PI * Xf) * np.sin(PI * Yf))
    return A, b, X, Y, u0


def solve_numerical(N: int, T: float, source: str,
                    rtol: float = 1e-6, atol: float = 1e-9,
                    n_eval: int = 200):
    """Численное решение. Возвращает (t_eval, U[t, n], X, Y, время_с)."""
    A, b, X, Y, u0 = build_system(N, source)
    n = N * N
    y0 = np.concatenate([u0, np.zeros(n)])

    def rhs(t, y):
        return A.dot(y) + b

    t_eval = np.linspace(0.0, T, n_eval)
    t0 = time.time()
    sol = solve_ivp(rhs, [0.0, T], y0, method="BDF", jac=lambda t, y: A,
                    t_eval=t_eval, rtol=rtol, atol=atol)
    elapsed = time.time() - t0
    U = sol.y[:n, :].T            # (n_eval, n) — только u (без v)
    return t_eval, U, X, Y, elapsed


# --- Аналитическое решение для проверки точности ---

def analytical(X: np.ndarray, Y: np.ndarray, t: float,
               source: str, n_max: int = 30) -> np.ndarray:
    """Замкнутое модальное решение (для контроля точности)."""
    out = np.zeros_like(X, dtype=float)
    for nx in range(1, n_max + 1):
        for my in range(1, n_max + 1):
            lam = PI**2 * (nx**2 + my**2)
            A0 = 1.0 if (nx == 1 and my == 1) else 0.0
            if source == "inhomo":
                ix = (-1)**(nx + 1) / (nx * PI)
                iy = (1 - (-1)**my) / (my * PI)
                f_nm = 8.0 * ix * iy        # коэф. разложения f=2x
            else:
                f_nm = 0.0
            if abs(A0) < 1e-15 and abs(f_nm) < 1e-15:
                continue
            disc = (KAPPA * lam)**2 - 4 * lam
            part = f_nm / lam
            if disc < 0:
                sig = -KAPPA * lam / 2
                om = math.sqrt(-disc) / 2
                D1 = A0 - part
                D2 = -sig * D1 / om
                Tt = math.exp(sig * t) * (D1 * math.cos(om * t)
                                          + D2 * math.sin(om * t)) + part
            else:
                r1 = (-KAPPA * lam + math.sqrt(disc)) / 2
                r2 = (-KAPPA * lam - math.sqrt(disc)) / 2
                C1 = -r2 * (A0 - part) / (r1 - r2)
                C2 = r1 * (A0 - part) / (r1 - r2)
                Tt = C1 * math.exp(r1 * t) + C2 * math.exp(r2 * t) + part
            out += Tt * np.sin(nx * PI * X) * np.sin(my * PI * Y)
    return out


def rel_l2(U_num: np.ndarray, X, Y, t_eval, source: str) -> float:
    """Относительная L2-ошибка численного решения относительно аналитики."""
    errs, refs = [], []
    N = X.shape[0]
    for k, t in enumerate(t_eval):
        u_ref = analytical(X, Y, t, source).ravel()
        errs.append(np.sum((U_num[k] - u_ref)**2))
        refs.append(np.sum(u_ref**2))
    return float(np.sqrt(sum(errs) / (sum(refs) + 1e-30)))


def main() -> None:
    print("Численное решение прямой задачи Грина–Нагди III типа")
    print("Метод прямых (конечные разности + BDF)\n")
    print(f"{'источник':10s} {'сетка':>8s} {'T':>6s} "
          f"{'ОДУ':>7s} {'время,с':>9s} {'L2-ошибка':>11s}")
    print("-" * 60)
    for source in ("homo", "inhomo"):
        for N in (40, 60):
            for T in (3.0, 30.0):
                t_eval, U, X, Y, el = solve_numerical(N, T, source)
                err = rel_l2(U, X, Y, t_eval, source)
                print(f"{source:10s} {N}x{N:<4d} {T:6.0f} "
                      f"{2*N*N:7d} {el:9.2f} {err:11.2e}")
    print("\n(время — только интегрирование по времени, без построения сетки)")


if __name__ == "__main__":
    main()
