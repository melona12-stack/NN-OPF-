r"""CPU 와 GPU 중 어느 쪽이 빠른가 — 배치 크기를 바꿔 가며 잰다.

    python scripts/s05_device_benchmark.py --case case30
    python scripts/s05_device_benchmark.py --case case118 --epochs 10

**GPU 가 항상 빠르지는 않다.** case30 · 배치 64 에서는 CPU 가 이긴다. 배치
하나가 :math:`64\times161` 행렬을 256 차원으로 두어 번 곱하는 정도라, GPU 는
계산보다 **호출 준비**에 더 오래 걸리기 때문이다.

    GPU 한 번 호출 = 커널 실행 준비 ~10 µs + 계산 ~2 µs
    CPU 한 번 호출 = 계산 ~20 µs

300 epoch × 66 배치 = 2만 번을 호출하면 그 준비 시간이 전부를 지배한다.
GPU 는 **한 번에 큰 일**을 줄 때 이긴다.

이 스크립트는 정확도를 재지 않는다. 배치를 바꾸면 최적화 자체가 달라져서
정확도 비교가 성립하지 않기 때문이다. 여기서 재는 것은 **처리량 하나**다 —
epoch 당 몇 밀리초인가.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnopf.models import SurrogateSpec  # noqa: E402
from nnopf.train import prepare, resolve_device, supervised_loss  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from s03_train_surrogate import PRESET, load_or_make  # noqa: E402


def time_epochs(ds, spec, split, case, device, batch, epochs, warmup=2) -> float:
    """epoch 하나에 걸리는 시간 [ms]. 학습 루프와 같은 연산만 남긴다."""
    b, model = prepare(ds, spec, split=split, case=case, seed=0, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tr = torch.as_tensor(split["train"], dtype=torch.long, device=b.device)

    def one_epoch() -> None:
        model.train()
        perm = tr[torch.randperm(len(tr), device=b.device)]
        for s in range(0, len(perm), batch):
            j = perm[s : s + batch]
            Vm, Va = model(b.X[j])
            loss = supervised_loss(model, Vm, Va, b.Vm[j], b.Va[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    for _ in range(warmup):        # 첫 몇 번은 커널 컴파일·캐시 준비라 버린다
        one_epoch()
    if b.device.type == "cuda":
        torch.cuda.synchronize()   # GPU 는 비동기라 동기화 없이 재면 거짓말이 나온다

    t0 = time.perf_counter()
    for _ in range(epochs):
        one_epoch()
    if b.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / epochs * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description="장치·배치별 처리량 측정")
    ap.add_argument("--case", default="case30")
    ap.add_argument("-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5, help="측정용 epoch 수")
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[64, 256, 1024, 4096])
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    pre = PRESET.get(a.case, PRESET["case30"])
    ds = load_or_make(a.case, a.n or pre["n"], a.seed, a.workers)
    spec = SurrogateSpec(hidden=pre["hidden"], layers=pre["layers"])
    split = ds.split_random(seed=0)

    gpu = resolve_device("auto")
    devices = [("CPU", torch.device("cpu"))]
    if gpu.type == "cuda":
        devices.append((torch.cuda.get_device_name(0), gpu))
    else:
        print("\n[주의] GPU 를 쓸 수 없어 CPU 만 잽니다.\n")

    print(f"\n{a.case} · 학습 {len(split['train']):,}표본 · "
          f"hidden {spec.hidden} x {spec.layers}층 · "
          f"CPU {a.threads} 스레드 · epoch {a.epochs}회 평균\n")
    hdr = f"{'배치':>6}" + "".join(f"{n[:22]:>24}" for n, _ in devices)
    if len(devices) == 2:
        hdr += f"{'빠른 쪽':>12}"
    print(hdr)
    print("-" * len(hdr))

    for batch in a.batches:
        row, times = f"{batch:>6}", []
        for _, dev in devices:
            ms = time_epochs(ds, spec, split, a.case, dev, batch, a.epochs)
            times.append(ms)
            row += f"{ms:>21.1f} ms"
        if len(times) == 2:
            cpu, gpu_ms = times
            faster = "GPU" if gpu_ms < cpu else "CPU"
            row += f"{faster} {max(cpu, gpu_ms) / min(cpu, gpu_ms):>6.2f}배"
        print(row)

    print("\n배치를 키우면 GPU 쪽이 유리해집니다 — 호출 준비 시간이 한 번에")
    print("처리하는 표본 수로 나눠지기 때문입니다. 다만 배치를 바꾸면 최적화")
    print("자체가 달라지므로, 정확도는 §8 이 아니라 따로 확인해야 합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
