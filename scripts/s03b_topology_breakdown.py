r"""미지 N-1 결과를 토폴로지로 쪼개 본다 — 06 문서 §4.1.

    python scripts/s03b_topology_breakdown.py --case case30 --tag _un1
    python scripts/s03b_topology_breakdown.py --case case118 --tag _gpu

전체 평균만 보면 "신경망도 미지 N-1 에서 몇 배 나빠진다" 로 끝나는데, 정상
토폴로지 표본과 본 적 없는 고장 표본을 나눠 재면 이야기가 완전히 달라진다.
열화가 퍼져 있는 게 아니라 **후자에만 뭉쳐 있다.** 그게 M4 에서 그래프
신경망을 쓸 근거이자 목표 수치가 된다.

``--split unseen-n1`` 로 학습한 결과에 쓰는 것이 본래 용도지만, 무작위 분할
결과에도 쓸 수 있다 (그쪽은 시험 표본에 학습에서 본 고장이 섞여 있으므로
"미지" 가 아니라 그냥 "N-1" 이라고 읽어야 한다).
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


def residual_pct(model, b, idx) -> np.ndarray:
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


def main() -> int:
    ap = argparse.ArgumentParser(description="오차를 토폴로지로 분해")
    ap.add_argument("--case", default="case30")
    ap.add_argument("--tag", default="_un1", help="results/<case><tag>_mlp.{json,pt}")
    a = ap.parse_args()

    stem = f"{a.case}{a.tag}"
    js, ck = ROOT / "results" / f"{stem}_mlp.json", ROOT / "results" / f"{stem}_mlp.pt"
    for f in (js, ck):
        if not f.exists():
            print(f"[중단] 파일이 없습니다: {f.relative_to(ROOT)}")
            print("       먼저 그 설정으로 s03_train_surrogate.py 를 돌리세요.")
            return 1

    payload = json.loads(js.read_text(encoding="utf-8"))
    n, seed = int(payload["n_samples"]), int(payload["data_seed"])
    npz = ROOT / "data" / f"{a.case}_n{n}_s{seed}.npz"
    if not npz.exists():
        print(f"[중단] 데이터가 없습니다: {npz.relative_to(ROOT)}")
        return 1

    ds = PowerFlowDataset.load(str(npz))
    spec = SurrogateSpec(**payload["spec"])
    split = (
        ds.split_unseen_n1(seed=0)
        if payload.get("split") == "unseen-n1"
        else ds.split_random(seed=0)
    )
    b, model = prepare(ds, spec, split=split, case=a.case, seed=0)
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    sd.setdefault("va_ref", model.va_ref)     # 슬랙 버그 이전 체크포인트 호환
    model.load_state_dict(sd)
    model.eval()

    te = split["test"]
    lin = fit_linear(ds, IOLayout(load_case(a.case)), split["train"])
    mlp_pct, lin_pct = residual_pct(model, b, te), residual_pct(lin, b, te)
    n1 = b.outage[torch.as_tensor(te, dtype=torch.long)].numpy() >= 0

    kind = "미지 N-1" if payload.get("split") == "unseen-n1" else "N-1"
    print(f"\n{a.case}{a.tag} · 시험 {len(te):,}표본 "
          f"(정상 {(~n1).sum():,} / {kind} {n1.sum():,})\n")

    if not n1.any() or not (~n1).any():
        print("  한쪽 종류만 있어 분해할 수 없습니다.")
        return 0

    hdr = f"{'표본 종류':<14}{'선형':>10}{'MLP':>10}{'개선':>10}"
    print(hdr); print("-" * len(hdr))
    for name, m in ((" 정상 토폴로지", ~n1), (f" {kind}", n1),
                    (" 전체 평균", np.ones_like(n1, bool))):
        L, M = lin_pct[m].mean(), mlp_pct[m].mean()
        print(f"{name:<14}{L:>9.2f}%{M:>9.2f}%{L / max(M, 1e-9):>9.1f}배")

    q99 = np.percentile(mlp_pct, 99)
    bad = mlp_pct >= q99
    print(f"\n  MLP 상위 1% 오차 표본 중 {kind} 비율: {n1[bad].mean() * 100:.1f}%")
    print(f"  MLP 표본별 중앙값 {np.median(mlp_pct):.2f}%  ·  "
          f"90분위 {np.percentile(mlp_pct, 90):.2f}%  ·  최대 {mlp_pct.max():.2f}%")
    if n1[bad].mean() > 0.9:
        print(f"\n  → 열화가 퍼져 있지 않고 {kind} 표본에 뭉쳐 있습니다.")
        print("    '정확도 부족' 이 아니라 '토폴로지 외삽 불가' 입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
