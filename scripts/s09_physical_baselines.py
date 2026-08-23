r"""물리 기반 선형 비교군을 신경망과 같은 지표로 잰다.

    python scripts/s09_physical_baselines.py
    python scripts/s09_physical_baselines.py --case case30 -n 30000

07 부록 §5 가 비교군으로 지목한 것은 **선형화 계열**인데, 우리가 그동안 쓴
비교군은 데이터에서 배운 **최소제곱 선형** 하나뿐이었다. 성격이 다른 셋을
같은 시험 집합에서 나란히 놓는다.

    최소제곱 선형   학습 분할을 보고 계수를 맞춤     데이터 필요
    DC 조류계산     |V|=1, 손실 0, Q 무시            데이터 불필요
    야코비안 선형화 기저해 근처 1차 테일러 전개      데이터 불필요

뒤의 둘은 **표본을 한 건도 안 본다.** 대신 계통 모델(Ybus)과, 상정사고마다
기저 케이스 조류계산 한 번을 쓴다. 그래서 "미지 N-1" 이라는 말이 이 둘에는
적용되지 않는다 — 학습이 없으니 일반화 격차도 없다. 비교표를 읽을 때
반드시 같이 봐야 하는 점이다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnopf import generate_dataset, load_case  # noqa: E402
from nnopf.baselines import DCPowerFlow, JacobianLinear, fit_linear  # noqa: E402
from nnopf.models import IOLayout, SurrogateSpec  # noqa: E402
from nnopf.train import evaluate, prepare  # noqa: E402

DEFAULT = [("case30", 30_000), ("case118", 60_000)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=None)
    ap.add_argument("-n", "--n-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.workers))
    jobs = DEFAULT
    if a.case:
        jobs = [(a.case, a.n_samples or dict(DEFAULT).get(a.case, 30_000))]

    out: list[dict] = []
    for case, n in jobs:
        ds = generate_dataset(case, n_samples=n, seed=a.seed,
                              workers=a.workers, verbose=False)
        sysm = load_case(case)
        layout = IOLayout(sysm)

        for split_name in ("random", "unseen-n1"):
            sp = (ds.split_random(seed=0) if split_name == "random"
                  else ds.split_unseen_n1(seed=0))
            b, _ = prepare(ds, SurrogateSpec(hidden=64, layers=2), split=sp, seed=0)

            models = [
                ("선형 최소제곱", lambda: fit_linear(ds, layout, sp["train"], sp["val"])),
                ("DC 조류계산", lambda: DCPowerFlow(sysm, layout)),
                ("야코비안 선형화", lambda: JacobianLinear(sysm, layout)),
            ]
            print(f"\n=== {case} · {n:,} · {split_name} "
                  f"(시험 {len(sp['test']):,}) ===")
            print(f"  {'모델':22s} {'P/부하 %':>10} {'Vm MAE':>11} {'Va MAE':>11} "
                  f"{'ΔP [pu]':>9} {'V위반':>7} {'초':>7}")
            for name, build in models:
                t0 = time.time()
                m = build()
                r = evaluate(m, b, sp["test"])
                dt = time.time() - t0
                print(f"  {name:20s} {r['p_over_load_pct']:10.4f} "
                      f"{r['vm_mae']:11.4e} {r['va_mae']:11.4e} "
                      f"{r['p_mismatch']:9.4f} {r['vlim_viol_pct']:6.2f}% {dt:7.1f}")
                out.append({"case": case, "n": n, "split": split_name,
                            "model": name, "seconds": dt, "metrics": r})

    dest = ROOT / "results" / "physical_baselines.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
