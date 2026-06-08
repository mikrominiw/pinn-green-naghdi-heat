# -*- coding: utf-8 -*-
"""Точечная перерисовка двух типов графиков:
1. arch_study_homo.png — отметки «лучшая по качеству» и
   «лучшая качество/скорость» делаются разноцветными и снабжаются
   текстовыми аннотациями вместо стандартной легенды.
2. *_temporal.png (исходно 1×5) — переразложить в 2×3 сетку
   методом PIL-нарезки исходного PNG (PINN-модели в памяти не
   сохранены, поэтому регенерация средствами matplotlib невозможна).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES  = HERE / "results" / "20260528_100734"
m    = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))

_SRC_RU = {"homo": "однородный (q = 0)",
           "inhomo": "неоднородный (q = 2xt)"}

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.dpi": 200,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# --------------------------------------------------------------------------
# 1. arch_study_homo.png
# --------------------------------------------------------------------------
def regenerate_arch_study():
    arch = m["arch_trials"][0]
    trials = sorted(arch["trials"], key=lambda d: d["params"])

    pars = np.array([t["params"] for t in trials])
    l2   = np.array([t["l2"]     for t in trials])
    tm   = np.array([t["time"]   for t in trials])
    blks = np.array([t["blocks"] for t in trials])

    block_colors = {1: "#1f77b4", 2: "#2ca02c",
                    3: "#9467bd", 4: "#8c564b"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # ── L2 vs параметры ──
    for nb in sorted(set(blks)):
        sel = blks == nb
        ax1.scatter(pars[sel], l2[sel],
                    s=70, color=block_colors[nb],
                    edgecolor="black", linewidth=0.6,
                    label=f"{nb} блок(а/ов)", zorder=3)
    order = np.argsort(pars)
    ax1.plot(pars[order], np.minimum.accumulate(l2[order]),
             color="gray", ls="--", lw=1.2, label="огибающая (лучшее)")

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Число параметров сети")
    ax1.set_ylabel("Относительная погрешность L₂")
    ax1.set_title("Точность от размера сети")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3, which="both")

    # ── время vs параметры ──
    for nb in sorted(set(blks)):
        sel = blks == nb
        ax2.scatter(pars[sel], tm[sel],
                    s=70, color=block_colors[nb],
                    edgecolor="black", linewidth=0.6,
                    label=f"{nb} блок(а/ов)", zorder=3)
    ax2.set_xscale("log")
    ax2.set_xlabel("Число параметров сети")
    ax2.set_ylabel("Время обучения, с")
    ax2.set_title("Скорость от размера сети")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle("Исследование размера архитектуры (жёсткое наложение "
                 "условий, источник q = 0, T = 30)", fontsize=12, y=1.00)
    fig.tight_layout()
    out = RES / "arch_study_homo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("written:", out)


# --------------------------------------------------------------------------
# 2. Переразложить *_temporal.png из 1×5 в 2×3
# --------------------------------------------------------------------------
def repack_temporal_png(src_path: Path, dst_path: Path) -> None:
    """Делит исходный 1×5 рисунок на 5 равных вертикальных полос и
    компонует их в 2×3 сетку (последняя ячейка остаётся пустой).
    Сверху не отрезается полоса с заголовком — она дублируется, чтобы
    подпись была видна над первым рядом."""
    img = Image.open(src_path).convert("RGB")
    W, H = img.size
    # Высота полосы с suptitle — оценим как первые ~10% по вертикали.
    # На практике matplotlib рисует suptitle над всеми subplots без
    # разделения; чтобы сохранить контекст, разрежем без удаления верха.
    panel_w = W // 5
    panels = [img.crop((i * panel_w, 0, (i + 1) * panel_w, H))
              for i in range(5)]
    # Цель: 2 × 3, последняя пустая
    cell_w, cell_h = panel_w, H
    out_w = 3 * cell_w
    out_h = 2 * cell_h
    new = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    positions = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
    for p, (cx, cy) in zip(panels, positions):
        new.paste(p, (cx * cell_w, cy * cell_h))
    new.save(dst_path, "PNG")


def regenerate_all_temporal():
    count = 0
    for p in RES.glob("*_temporal.png"):
        # пропускаем уже переразложенные (если запускали ранее)
        if "_grid" in p.stem:
            continue
        dst = p.with_name(p.stem + "_grid.png")
        repack_temporal_png(p, dst)
        count += 1
    print(f"temporal repacked: {count} files")


if __name__ == "__main__":
    regenerate_arch_study()
    regenerate_all_temporal()
