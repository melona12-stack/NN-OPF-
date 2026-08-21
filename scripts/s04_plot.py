r"""실험 결과를 그림으로 — 표로는 안 보이는 것을 본다.

    python scripts/s04_plot.py --case case118 --tag _fix_un1
    python scripts/s04_plot.py --case case30  --dark

``results/<case><tag>_mlp.json`` 과 같은 이름의 ``.pt`` 체크포인트를 읽어
``figures/`` 에 PNG 를 만든다. 체크포인트가 없으면 학습곡선만 그린다.

그리는 것 다섯 가지:

1. **학습곡선** — 언제 멈췄어야 했나, 과적합인가
2. **오차 분포** — 정상 토폴로지 vs 미지 N-1 (06 문서 §4.1 의 핵심)
3. **모델 비교** — 선형 기준선 대비 어디서 이기나
4. **모선별 오차** — 오차가 어느 모선에 몰려 있나
5. **예측 vs 실제** — 어디서 무너지나

한글 폰트가 없는 환경이면 라벨이 자동으로 영문으로 바뀐다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnopf.baselines import fit_linear  # noqa: E402
from nnopf.case import load_case  # noqa: E402
from nnopf.dataset import PowerFlowDataset  # noqa: E402
from nnopf.models import IOLayout, SurrogateSpec  # noqa: E402
from nnopf.train import prepare  # noqa: E402
from nnopf.viz import annotate_bars, bar_gap, label, pct_axis, save, setup  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

FIG = ROOT / "figures"


# --------------------------------------------------------------------------
# 1. 학습곡선
# --------------------------------------------------------------------------
def plot_learning_curve(payload: dict, t, out: Path) -> Path | None:
    runs = [r for r in payload.get("results", []) if r.get("history")]
    if not runs:
        return None
    h = runs[-1]["history"]
    ep = np.array([d["epoch"] for d in h])
    tr = np.array([d["train"] for d in h])
    va = np.array([d["val"] for d in h])
    best = int(runs[-1].get("best_epoch", int(ep[np.argmin(va)])))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ep, tr, color=t.series[0], label=label("학습", "train"))
    ax.plot(ep, va, color=t.series[1], label=label("검증", "val"))
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(label("손실 (표준화 공간)", "loss (standardized)"))

    j = int(np.searchsorted(ep, best))
    if j < len(ep):
        ax.axvline(best, color=t.ink2, lw=1, ls="--", alpha=0.6)
        ax.annotate(
            label(f"최고 검증\nepoch {best}", f"best val\nepoch {best}"),
            (best, va[j]), xytext=(10, 14), textcoords="offset points",
            color=t.ink, fontsize=9,
        )

    gap = tr[-1] / max(va[-1], 1e-30)
    ax.set_title(
        label(
            f"학습곡선 — 마지막 학습/검증 비 {1/gap:,.0f}배",
            f"Learning curve — final val/train ratio {1/gap:,.0f}x",
        ),
        color=t.ink, fontsize=12, loc="left", pad=12,
    )
    ax.legend(loc="upper right")
    return save(fig, out)


# --------------------------------------------------------------------------
# 2~5. 모델이 필요한 그림들
# --------------------------------------------------------------------------
def per_sample_pct(model, b, idx) -> np.ndarray:
    """표본별 |ΔP| 합을 그 표본의 부하로 나눈 값 [%]."""
    ii = torch.as_tensor(idx, dtype=torch.long)
    with torch.no_grad():
        Vm, Va = model(b.X[ii])
    dP, _ = b.physics64.residual(
        Vm.double(), Va.double(),
        b.p_spec[ii].double(), b.q_spec[ii].double(), b.outage[ii],
    )
    load = b.p_spec[ii].double().abs().sum(-1).clamp(min=1e-9)
    return (dP.abs().sum(-1) / load * 100).numpy()


def plot_error_split(pct_mlp, pct_lin, n1, t, out: Path, kind: str) -> Path:
    """오차 분포를 토폴로지로 쪼개서 — 이번 단계의 핵심 그림.

    가로축은 로그다. 두 모델의 오차 규모가 100배 차이라 선형 눈금으로는
    한쪽이 반드시 뭉개진다. 눈금을 공유하되 로그로 두면 둘 다 보인다.
    """
    both = np.concatenate([pct_lin, pct_mlp])
    lo = max(np.percentile(both, 0.5), 1e-3)
    hi = np.percentile(both, 99.9) * 1.3
    bins = np.geomspace(lo, hi, 70)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, sharex=True)
    for ax, pct, name in (
        (axes[0], pct_lin, label("선형 최소제곱", "Linear least squares")),
        (axes[1], pct_mlp, label("MLP + 선형지름길", "MLP + linear skip")),
    ):
        ax.hist(np.clip(pct[~n1], lo, hi), bins=bins, color=t.series[0],
                alpha=0.85, label=label("정상 토폴로지", "intact topology"))
        ax.hist(np.clip(pct[n1], lo, hi), bins=bins, color=t.series[1],
                alpha=0.85, label=kind)
        ax.set_xscale("log")
        ax.set_xlabel(label("표본별 유효전력 불일치 / 부하 (로그)",
                            "P mismatch / load (log)"))
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}%")
        ax.set_title(
            f"{name}\n"
            + label(
                f"정상 {pct[~n1].mean():.2f}%   {kind} {pct[n1].mean():.2f}%",
                f"intact {pct[~n1].mean():.2f}%   {kind} {pct[n1].mean():.2f}%",
            ),
            color=t.ink, fontsize=11, loc="left", pad=10,
        )
    axes[0].set_ylabel(label("표본 수", "samples"))
    axes[0].legend(loc="upper left", fontsize=10)
    fig.suptitle(
        label(
            f"오차가 {kind} 표본에 뭉쳐 있는가",
            f"Is the error concentrated in {kind} samples?",
        ),
        color=t.ink, fontsize=13, x=0.005, ha="left", y=1.06,
    )
    return save(fig, out)


def plot_model_compare(pct_mlp, pct_lin, n1, t, out: Path, kind: str) -> Path:
    """어디서 이기는지 — 정상/미지/전체 세 구간."""
    groups = [
        (label("정상 토폴로지", "intact"), ~n1),
        (kind, n1),
        (label("전체", "all"), np.ones_like(n1, bool)),
    ]
    names = [g[0] for g in groups]
    lin = [pct_lin[m].mean() for _, m in groups]
    mlp = [pct_mlp[m].mean() for _, m in groups]

    x = np.arange(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - w / 2, lin, w, color=t.series[0],
           label=label("선형 최소제곱", "Linear"))
    ax.bar(x + w / 2, mlp, w, color=t.series[1],
           label=label("MLP + 선형지름길", "MLP + skip"))
    bar_gap(ax)
    annotate_bars(ax, "{:.2f}%", t)
    ax.set_xticks(x, names)
    ax.set_ylabel(label("유효전력 불일치 / 부하", "P mismatch / load"))
    pct_axis(ax)
    ax.set_ylim(0, max(lin) * 1.18)      # 막대 끝 라벨이 들어갈 자리
    # 범례를 축 안에 두면 막대 끝 값 라벨과 부딪힌다 — 위로 뺀다.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=10)

    ratios = " · ".join(f"{l / max(m, 1e-9):.0f}x" for l, m in zip(lin, mlp))
    fig.suptitle(
        label(f"신경망 개선 배수 — {ratios}", f"NN improvement — {ratios}"),
        color=t.ink, fontsize=12.5, x=0.005, ha="left", y=1.10,
    )
    return save(fig, out)


def plot_per_bus(model, b, idx, t, out: Path) -> Path:
    """오차가 어느 모선에 몰려 있나. 한 계열이므로 색은 하나, 최악만 강조."""
    ii = torch.as_tensor(idx, dtype=torch.long)
    with torch.no_grad():
        Vm, Va = model(b.X[ii])
    dP, _ = b.physics64.residual(
        Vm.double(), Va.double(),
        b.p_spec[ii].double(), b.q_spec[ii].double(), b.outage[ii],
    )
    per_bus = dP.abs().mean(0).numpy()
    worst = int(np.argmax(per_bus))

    fig, ax = plt.subplots(figsize=(10, 3.8))
    colors = [t.grid if i != worst else t.critical for i in range(len(per_bus))]
    colors = [t.series[0] if c == t.grid else c for c in colors]
    ax.bar(np.arange(len(per_bus)), per_bus, color=colors, width=0.85)
    ax.set_xlabel(label("모선 번호 (0부터)", "bus index"))
    ax.set_ylabel(label("평균 |ΔP| [pu]", "mean |ΔP| [pu]"))
    ax.annotate(
        label(f"최악: 모선 {worst}", f"worst: bus {worst}"),
        (worst, per_bus[worst]), xytext=(6, 6), textcoords="offset points",
        color=t.ink, fontsize=10, fontweight="bold",
    )
    share = per_bus[per_bus >= np.percentile(per_bus, 90)].sum() / per_bus.sum()
    ax.set_title(
        label(
            f"모선별 평균 유효전력 잔차 — 상위 10% 모선이 전체의 {share:.0%}",
            f"Mean P residual by bus — top 10% of buses carry {share:.0%}",
        ),
        color=t.ink, fontsize=12, loc="left", pad=12,
    )
    return save(fig, out)


def plot_scatter(model, b, idx, t, out: Path) -> Path:
    """예측 vs 실제. 대각선에서 벗어나는 곳이 무너지는 곳이다."""
    ii = torch.as_tensor(idx, dtype=torch.long)
    with torch.no_grad():
        Vm, Va = model(b.X[ii])
    pq, ns = model.pq_idx, model.va_idx

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, pred, true, name, unit in (
        (axes[0], Vm[:, pq], b.Vm[ii][:, pq], "|V|", "pu"),
        (axes[1], Va[:, ns], b.Va[ii][:, ns], "θ", "rad"),
    ):
        p = pred.detach().numpy().ravel()
        q = true.numpy().ravel()
        k = np.random.default_rng(0).choice(len(p), min(20000, len(p)), replace=False)
        ax.plot(q[k], p[k], ".", ms=2, alpha=0.25, color=t.series[0],
                markeredgewidth=0)
        lo, hi = float(min(q.min(), p.min())), float(max(q.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], lw=1.2, color=t.ink2, alpha=0.7)
        ax.set_xlabel(label(f"실제 {name} [{unit}]", f"true {name} [{unit}]"))
        ax.set_ylabel(label(f"예측 {name} [{unit}]", f"predicted {name} [{unit}]"))
        r2 = 1 - ((p - q) ** 2).sum() / max(((q - q.mean()) ** 2).sum(), 1e-30)
        ax.set_title(f"{name}   R² = {r2:.5f}", color=t.ink, fontsize=11,
                     loc="left", pad=10)
        ax.grid(alpha=0.5)
    fig.suptitle(
        label("예측 대 실제 — 대각선에서 벗어난 점이 실패 사례",
              "Predicted vs true — points off the diagonal are the failures"),
        color=t.ink, fontsize=13, x=0.005, ha="left", y=1.02,
    )
    return save(fig, out)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="실험 결과 그림 그리기")
    ap.add_argument("--case", default="case30")
    ap.add_argument("--tag", default="", help="results/<case><tag>_mlp.json")
    ap.add_argument("--n", type=int, default=None, help="데이터 표본 수")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dark", action="store_true", help="어두운 배경")
    a = ap.parse_args()

    t = setup(a.dark)
    stem = f"{a.case}{a.tag}"
    js = ROOT / "results" / f"{stem}_mlp.json"
    if not js.exists():
        print(f"[중단] 결과 파일이 없습니다: {js.relative_to(ROOT)}")
        print("       먼저 scripts/s03_train_surrogate.py 를 돌리세요.")
        return 1
    payload = json.loads(js.read_text(encoding="utf-8"))
    made: list[Path] = []

    p = plot_learning_curve(payload, t, FIG / f"{stem}_01_learning_curve.png")
    if p:
        made.append(p)
    else:
        print("  (학습곡선 건너뜀 — 이 결과에는 history 가 없습니다)")

    ck = ROOT / "results" / f"{stem}_mlp.pt"
    if not ck.exists():
        print(f"  (체크포인트 없음: {ck.name} — 나머지 그림은 건너뜁니다)")
    else:
        n = a.n or int(payload.get("n_samples", 6000))
        seed = int(payload.get("data_seed", a.seed))
        npz = ROOT / "data" / f"{a.case}_n{n}_s{seed}.npz"
        if not npz.exists():
            print(f"[중단] 데이터가 없습니다: {npz.relative_to(ROOT)}")
            return 1
        ds = PowerFlowDataset.load(str(npz))

        sp = payload["spec"]
        spec = SurrogateSpec(**sp)
        split = (
            ds.split_unseen_n1(seed=0)
            if payload.get("split") == "unseen-n1"
            else ds.split_random(seed=0)
        )
        b, model = prepare(ds, spec, split=split, case=a.case, seed=0)
        sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        sd.setdefault("va_ref", model.va_ref)   # 슬랙 버그 이전 체크포인트 호환
        model.load_state_dict(sd)
        model.eval()

        te = split["test"]
        lin = fit_linear(ds, IOLayout(load_case(a.case)), split["train"], split["val"])
        pct_mlp = per_sample_pct(model, b, te)
        pct_lin = per_sample_pct(lin, b, te)
        n1 = (b.outage[torch.as_tensor(te, dtype=torch.long)].numpy() >= 0)

        # 무작위 분할에서는 시험용 고장이 학습에도 나오므로 "미지" 가 아니다.
        # 같은 그림에 같은 이름을 붙이면 결과를 잘못 읽게 된다.
        unseen = payload.get("split") == "unseen-n1"
        kind = label("미지 N-1", "unseen N-1") if unseen else label("N-1", "N-1")
        if n1.any() and (~n1).any():
            made.append(plot_error_split(pct_mlp, pct_lin, n1, t,
                                         FIG / f"{stem}_02_error_split.png", kind))
            made.append(plot_model_compare(pct_mlp, pct_lin, n1, t,
                                           FIG / f"{stem}_03_compare.png", kind))
        else:
            print("  (분할에 한쪽 표본만 있어 토폴로지 비교는 건너뜁니다)")
        made.append(plot_per_bus(model, b, te, t, FIG / f"{stem}_04_per_bus.png"))
        made.append(plot_scatter(model, b, te, t, FIG / f"{stem}_05_scatter.png"))

    print(f"\n그림 {len(made)}장:")
    for p in made:
        print(f"  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
