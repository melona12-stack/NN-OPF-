r"""데이터셋이 다른 컴퓨터의 것과 같은지 확인한다 — 지문 찍기.

    python scripts/s06_data_fingerprint.py

시드를 고정했으니 어느 컴퓨터에서 만들어도 **비트 단위로 같아야** 한다.
그런데 실제로는 병렬 워커 수, 라이브러리 버전, 부동소수점 처리 방식 때문에
갈라질 수 있다. 데이터가 다르면 "같은 실험"이 아니게 되므로, 결과를 비교하기
전에 이걸 먼저 맞춰야 한다.

닫힌 해로 풀리는 선형 최소제곱을 같이 재는 것이 요령이다. 시드도 epoch 도
없어서, **데이터가 같으면 반드시 같은 값**이 나온다. 값이 다르면 데이터가
다른 것이고, 같으면 데이터도 같다고 볼 수 있다.
"""

from __future__ import annotations

import hashlib
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
from nnopf.train import evaluate, prepare  # noqa: E402


def fingerprint(a: np.ndarray) -> str:
    """배열의 짧은 지문. 바이트를 그대로 해싱한다."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def main() -> int:
    files = sorted((ROOT / "data").glob("*.npz"))
    if not files:
        print("[중단] data/ 에 데이터셋이 없습니다.")
        return 1

    for f in files:
        ds = PowerFlowDataset.load(str(f))
        case = ds.case
        print(f"\n{'=' * 64}\n{f.name}")
        print(f"  표본 {ds.n_samples:,} · 모선 {ds.n_bus} · "
              f"N-1 {(ds.outage >= 0).mean() * 100:.1f}%")
        print(f"  Vm  {fingerprint(ds.Vm)}   Va  {fingerprint(ds.Va)}")
        print(f"  Pd  {fingerprint(ds.Pd)}   outage {fingerprint(ds.outage)}")
        print(f"  Vm 범위 {ds.Vm.min():.6f} ~ {ds.Vm.max():.6f}")
        print(f"  Va 범위 {ds.Va.min():.6f} ~ {ds.Va.max():.6f} rad")

        # 닫힌 해라 데이터가 같으면 반드시 같은 값이 나온다
        try:
            split = ds.split_random(seed=0)
            layers = 4 if ds.n_bus > 60 else 3
            b, _ = prepare(ds, SurrogateSpec(hidden=256, layers=layers),
                           split=split, case=case)
            lin = fit_linear(ds, IOLayout(load_case(case)), split["train"])
            m = evaluate(lin, b, split["test"])
            print(f"  선형 기준선(닫힌 해): Vm MAE {m['vm_mae']:.6e} · "
                  f"P/부하 {m['p_over_load_pct']:.4f}%")
        except Exception as e:  # noqa: BLE001
            print(f"  선형 기준선 계산 실패: {type(e).__name__}: {e}")

    print(f"\n{'=' * 64}")
    print("이 값들을 다른 컴퓨터의 것과 맞춰 보세요. 지문이 다르면 데이터가")
    print("다른 것이고, 그러면 결과를 나란히 놓고 비교할 수 없습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
