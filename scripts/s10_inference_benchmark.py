#!/usr/bin/env python
"""추론 속도 — 뉴턴-랩슨 대비, CPU 와 GPU 를 나눠 잰다.

06 문서 §8 의 표가 **CPU 기준으로만** 채워져 있었다. 04 문서 §2.8 이
"CPU 단건 / GPU 단건 / GPU 배치를 반드시 나눠 재라" 고 했는데, 그중
GPU 두 칸이 비어 있었다. 이 스크립트가 그 칸을 채운다.

**가중치는 학습하지 않는다.** 순전파 비용은 구조(층 수·너비·헤드)로 정해지고
가중치 값과는 무관하다. 그래서 같은 설정으로 모델을 세우기만 하고 잰다.
정확도를 재는 것이 아니라 **시간**을 재는 것이므로 이게 맞다.

    python scripts/s10_inference_benchmark.py --case case30
    python scripts/s10_inference_benchmark.py --case case118 --model gat --layers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nnopf.gnn import GATSpec                                # noqa: E402
from nnopf.models import SurrogateSpec                       # noqa: E402
from nnopf.train import benchmark, prepare                   # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from s03_train_surrogate import load_or_make                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="case30")
    ap.add_argument("-n", type=int, default=None, help="표본 수 (기본: 계통별 프리셋)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="unseen-n1", choices=["random", "unseen-n1"])
    ap.add_argument("--model", default="mlp", choices=["mlp", "gat"])
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-single", type=int, default=200, help="단건 추론 표본 수")
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 256, 1024],
                    help="배치 추론에서 재 볼 배치 크기들")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    n = a.n or {"case30": 30_000, "case118": 60_000}.get(a.case, 10_000)
    hidden = a.hidden or (128 if a.model == "gat" else 256)
    layers = a.layers or (8 if a.model == "gat" else 4)
    spec = (GATSpec(hidden=hidden, layers=layers, heads=a.heads)
            if a.model == "gat" else
            SurrogateSpec(hidden=hidden, layers=layers))

    ds = load_or_make(a.case, n, a.seed, a.workers)
    split = (ds.split_unseen_n1(seed=0) if a.split == "unseen-n1"
             else ds.split_random(seed=0))
    test = split["test"]

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    if "cuda" not in devices:
        print("경고: GPU 를 못 찾았습니다. CPU 만 잽니다.")

    print(f"\n{a.case} · {a.model.upper()} hidden {hidden} x {layers}층 · "
          f"시험 {len(test):,} 표본 · 단건 {a.n_single}회\n")
    head = (f"{'장치':<26} {'배치':>6} {'뉴턴-랩슨':>10} {'단건':>10} "
            f"{'배치추론':>10} {'단건 배속':>10} {'배치 배속':>10}")
    print(head)
    print("-" * len(head))

    rows = []
    for dev in devices:
        b, model = prepare(ds, spec, split=split, device=dev, seed=0)
        name = (torch.cuda.get_device_name(0) if dev == "cuda" else "CPU")
        for bs in a.batches:
            r = benchmark(model, b, test, n_single=a.n_single, batch=bs)
            r["device"] = name
            rows.append(r)
            print(f"{name:<26} {bs:6d} {r['nr_ms']:9.3f}ms {r['single_ms']:9.4f}ms "
                  f"{r['batch_ms']:9.4f}ms {r['speedup_single']:9.1f}배 "
                  f"{r['speedup_batch']:9.1f}배")

    print("\n뉴턴-랩슨은 장치와 무관합니다 (CPU numpy). 배치별로 다시 재는 것은"
          "\n같은 조건에서 흔들림을 보기 위해서입니다.")

    if not a.no_save:
        out = ROOT / "results" / f"{a.case}_inference_benchmark.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"case": a.case, "model": a.model, "hidden": hidden, "layers": layers,
             "n_samples": int(ds.n_samples), "split": a.split,
             "n_test": int(len(test)), "n_single": a.n_single, "rows": rows},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
