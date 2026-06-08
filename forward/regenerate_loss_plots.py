# -*- coding: utf-8 -*-
"""Перерисовка проблемных графиков:
1. loss_grouped_{homo,inhomo}.png — сплошные контрастные линии, без маркеров.
2. sources_comparison_{hard,soft}.png — два аналитических профиля u(0,5; 0,5; t)
   для q = 0 и q = 2xt (PINN-модели в памяти не сохранены, поэтому
   показано только эталонное решение).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES  = HERE / "results" / "20260528_100734"
m    = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))

# Для аналитики используем класс из pinn_forward.py
sys.path.insert(0, str(HERE))

_SRC_RU = {"homo": "q = 0 (отсутствие источника)",
           "inhomo": "q = 2xt (линейный источник)"}
_ANS_RU = {"hard": "жёсткий анзац",
           "soft": "мягкий анзац"}

# яркие контрастные цвета без маркеров
_SEED_COLOR = {
    16: "#1f77b4",   # синий
    5:  "#d62728",   # красный
    22: "#2ca02c",   # зелёный
}

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi": 200,
})


def _runs(src: str, ans: str, seed: int):
    out = {}
    for r in m["study_T"]:
        if r["source"] == src and r["ansatz"] == ans and r["seed"] == seed:
            out[float(r["t_max"])] = np.array(r["loss_hist"], dtype=float)
    return out


def plot_loss_grouped(source: str, out_path: Path) -> None:
    """Сетка 2 (анзацы) × 3 (T); сплошные линии разных цветов, без маркеров."""
    Ts = [3.0, 30.0, 100.0]
    ansatze = ["hard", "soft"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharey=False)

    for row, ans in enumerate(ansatze):
        for col, T in enumerate(Ts):
            ax = axes[row, col]
            for seed in (16, 5, 22):
                rs = _runs(source, ans, seed)
                if T not in rs:
                    continue
                hist = rs[T]
                ax.semilogy(np.arange(1, len(hist) + 1), hist,
                            color=_SEED_COLOR[seed], ls="-", lw=1.8,
                            label=f"seed = {seed}")
            ax.set_xlabel("Итерация (с логированием)", fontsize=10)
            ax.set_ylabel("Функция потерь", fontsize=10)
            ax.set_title(f"{_ANS_RU[ans]}, T = {int(T)}", fontsize=11)
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle(f"Сходимость функции потерь по случайным инициализациям. "
                 f"Источник: {_SRC_RU[source]}", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("written:", out_path)


# --------------------------------------------------------------------------
# sources_comparison — аналитика для двух источников
# --------------------------------------------------------------------------
def plot_sources_comparison(ansatz: str, out_path: Path) -> None:
    """Две панели — q=0 и q=2xt — профиль аналитического решения
    u(0,5; 0,5; t) на интервале T = 100 (репрезентативный для серии)."""
    from pinn_forward import AnalyticalSolution

    T_MAX = 100.0
    t_vals = np.linspace(0.0, T_MAX, 800)
    xv = np.full_like(t_vals, 0.5)
    yv = np.full_like(t_vals, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, src in zip(axes, ("homo", "inhomo")):
        anal = AnalyticalSolution(src, T_MAX,
                                  n_modes=1 if src == "homo" else 15)
        u_ref = anal(xv, yv, t_vals)
        ax.plot(t_vals, u_ref, color="#1f77b4", ls="-", lw=1.6,
                label="Аналитическое решение")
        ax.set_xlabel("Время $t$")
        ax.set_ylabel("$u(0{,}5;\\;0{,}5;\\;t)$")
        ax.set_title(_SRC_RU[src])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(f"Сравнение источников тепловыделения "
                 f"(аналитическое решение, T = {int(T_MAX)})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("written:", out_path)


if __name__ == "__main__":
    for src in ("homo", "inhomo"):
        plot_loss_grouped(src, RES / f"loss_grouped_{src}.png")
    for ans in ("hard", "soft"):
        plot_sources_comparison(ans, RES / f"sources_comparison_{ans}.png")
