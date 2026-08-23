r"""학습을 **한 걸음도 시키기 전에** 모델이 얼마나 잘하는지 잰다.

    python scripts/s08_init_quality.py
    python scripts/s08_init_quality.py --case case30 -n 30000

왜 따로 재야 하나. 학습 루프는 한 epoch 안에서 **학습을 먼저 하고 그다음에
검증**한다. 그래서 학습 이력(``history``)의 첫 줄은 초기화 직후가 아니라
**1 epoch 을 이미 돌린 뒤**다. 초기화 품질은 이력에 아예 남지 않는다.

06 문서 §6.1 에서 이 사실을 모르고 이력 첫 줄을 "학습 전" 이라고 적었다가
고쳤다. 이 스크립트가 그 표의 "진짜 학습 전" 열을 만든다.

세 가지를 같은 시험 집합에서 비교한다:

- **선형 기준선** — 최소제곱으로 푼 선형 모델
- **``lstsq`` 초기화 직후** — 지름길 가중치를 그 선형 해로 채운 신경망
- **``zero`` 초기화 직후** — 기본값(지름길 0)

``lstsq`` 가 선형과 거의 같게 나오는 것이 정상이다. 본체가 아직 아무 말도
안 하는 상태에서는 모델이 곧 선형 모델이기 때문이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnopf import generate_dataset, load_case  # noqa: E402
from nnopf.baselines import fit_linear  # noqa: E402
from nnopf.models import IOLayout, SurrogateSpec  # noqa: E402
from nnopf.train import evaluate, prepare  # noqa: E402

# 06 문서 §6.1 이 인용하는 크기. 실제 실험과 같아야 비교가 의미 있다.
DEFAULT = [("case30", 30_000), ("case118", 60_000)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=None, help="하나만 잴 때 (기본: 둘 다)")
    ap.add_argument("-n", "--n-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026, help="데이터 생성 씨앗")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.workers))
    jobs = DEFAULT
    if a.case:
        n = a.n_samples or dict(DEFAULT).get(a.case, 30_000)
        jobs = [(a.case, n)]

    for case, n in jobs:
        ds = generate_dataset(case, n_samples=n, seed=a.seed,
                              workers=a.workers, verbose=False)
        # 미지 N-1: 시험 고장이 학습·검증 어디에도 없다 (06 문서 §4.1)
        sp = ds.split_unseen_n1(seed=0)
        layout = IOLayout(load_case(case))

        def spec(skip_init: str | None = None) -> SurrogateSpec:
            kw = {"skip_init": skip_init} if skip_init else {}
            return SurrogateSpec(hidden=a.hidden, layers=a.layers, **kw)

        # prepare() 는 초기화까지만 한다 — 학습은 train() 이 따로 한다.
        b_lin, _ = prepare(ds, spec(), split=sp, seed=0)
        lin = evaluate(fit_linear(ds, layout, sp["train"], sp["val"]),
                       b_lin, sp["test"])
        b_ls, m_ls = prepare(ds, spec("lstsq"), split=sp, seed=0)
        ini = evaluate(m_ls, b_ls, sp["test"])
        b_z, m_z = prepare(ds, spec(), split=sp, seed=0)
        zer = evaluate(m_z, b_z, sp["test"])

        print(f"\n=== {case} · {n:,} 표본 · 미지 N-1 · 진짜 학습 전 ===")
        print(f"  시험 표본            {len(sp['test']):,} 개")
        print(f"  선형 기준선          {lin['p_over_load_pct']:8.3f} %")
        print(f"  lstsq 초기화 직후    {ini['p_over_load_pct']:8.3f} %")
        print(f"  zero  초기화 직후    {zer['p_over_load_pct']:8.3f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
