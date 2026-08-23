#!/usr/bin/env python
"""4단계 — 조류계산 대체모델(MLP 기준선) 학습.

사용 예::

    # case30 기준선 (P2 §4.1 설정)
    .venv/bin/python scripts/s03_train_surrogate.py --case case30

    # 물리정보 손실 켜기
    .venv/bin/python scripts/s03_train_surrogate.py --case case30 --lam 3e-3

    # λ 민감도 곡선 (P2 Table 9)
    .venv/bin/python scripts/s03_train_surrogate.py --case case30 --lam-sweep 0 1e-4 5e-4 1e-3 5e-3

    # 학습곡선 — 몇 표본이 필요한지 실측
    .venv/bin/python scripts/s03_train_surrogate.py --case case30 --curve 500 1000 2000 4000 6000

데이터셋은 ``data/`` 에 캐시된다(커밋하지 않음). 없으면 시드로 재생성한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nnopf.baselines import fit_linear  # noqa: E402
from nnopf.dataset import PowerFlowDataset, generate_dataset  # noqa: E402
from nnopf.gnn import GATSpec  # noqa: E402
from nnopf.models import SurrogateSpec  # noqa: E402
from nnopf.train import (  # noqa: E402
    TrainConfig,
    evaluate,
    prepare,
    resolve_device,
    save_run,
    spec_config_dict,
    train,
)

# 계통별 기본값. lr 은 P2 값이 아니라 우리 MLP 에 맞춰 올린 값이다
# (train.TrainConfig 의 경고 참조).
PRESET = {
    "case30": dict(n=6000, hidden=256, layers=3, epochs=300, lr=2e-3,
                   lam=3e-3, lam_warmup=20, lam_ramp=50),
    "case118": dict(n=20000, hidden=256, layers=4, epochs=500, lr=2e-3,
                    lam=5e-4, lam_warmup=300, lam_ramp=50),
    "case300": dict(n=20000, hidden=256, layers=4, epochs=500, lr=2e-3,
                    lam=5e-4, lam_warmup=300, lam_ramp=50),
}


def load_or_make(case: str, n: int, seed: int, workers: int) -> PowerFlowDataset:
    path = ROOT / "data" / f"{case}_n{n}_s{seed}.npz"
    if path.exists():
        print(f"데이터셋 캐시 사용: {path.name}")
        return PowerFlowDataset.load(str(path))
    print(f"데이터셋 생성 중: {case} x {n} (seed {seed})")
    ds = generate_dataset(case, n_samples=n, seed=seed, workers=workers, verbose=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save(str(path))
    return ds


def fmt_row(tag: str, m: dict, secs: float, extra: str = "") -> str:
    return (
        f"{tag:<14s} {m['vm_mae']:.3e}  {m['vm_max']:.3e}  {m['va_mae']:.3e}  "
        f"{m['p_mismatch']:8.4f}  {m['q_mismatch']:8.4f}  "
        f"{m['p_over_load_pct']:6.2f}  {m['vlim_viol_pct']:6.2f}  {secs:6.1f}s {extra}"
    )


HEAD = (
    f"{'실험':<14s} {'Vm MAE':>9s}  {'Vm 최대':>9s}  {'Va MAE':>9s}  "
    f"{'ΔP[pu]':>8s}  {'ΔQ[pu]':>8s}  {'P/부하%':>6s}  {'V위반%':>6s}  {'시간':>7s}"
)


DEVICE = "cpu"     # main() 이 --device 로 정한다


def run_one(ds, spec: SurrogateSpec, cfg: TrainConfig, split=None, quiet=False):
    b, model = prepare(ds, spec, split=split, device=DEVICE,
                       jac_alpha=cfg.jac_alpha)
    r = train(model, b, cfg, verbose=not quiet)
    m = evaluate(model, b, b.split["test"])
    return model, b, r, m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="case30")
    p.add_argument("-n", type=int, default=None, help="표본 수 (기본: 계통별 프리셋)")
    p.add_argument("--seed", type=int, default=2026, help="데이터 생성 시드")
    p.add_argument("--train-seed", type=int, default=0, help="학습 시드")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=2, help="torch 스레드 (작은 배치라 2가 최적)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="auto 는 쓸 수 있으면 GPU. 잔차 측정은 항상 CPU float64")

    p.add_argument("--model", default="mlp", choices=["mlp", "gat"],
                   help="gat = 그래프 어텐션 (M4). 입출력 계약은 같다")
    p.add_argument("--heads", type=int, default=4, help="gat 전용")
    p.add_argument("--agg", default="softmax", choices=["softmax", "sum"],
                   help="gat 전용: sum 은 소프트맥스 없이 합산 (조류방정식과 같은 꼴)")
    p.add_argument("--no-gate", action="store_true",
                   help="gat 전용 절제 실험: 끊긴 선로의 어텐션을 막지 않는다")
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--activation", default="silu")
    p.add_argument("--vm-head", default="scaled", choices=["scaled", "raw"])
    p.add_argument("--no-residual", action="store_true",
                   help="선형 지름길 끄기 (순수 MLP 비교용)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=256,
                   help="GPU 에서 64 는 손해다 (06 문서 §8.1)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lam", type=float, default=None, help="물리손실 가중치 (0 = 순수 지도학습)")

    p.add_argument("--split", default="random", choices=["random", "unseen-n1"])
    p.add_argument("--patience", type=int, default=200,
                   help="검증이 이만큼 안 좋아지면 멈춘다")
    p.add_argument("--skip-init", default="zero", choices=["zero", "lstsq"],
                   help="선형 지름길의 출발점. zero=0 에서 함께 학습(기본), "
                        "lstsq=학습 분할의 최소제곱 해에서 출발 "
                        "(그러면 학습 첫 순간의 모델이 곧 선형 기준선이다)")
    p.add_argument("--skip-freeze", action="store_true",
                   help="지름길을 얼려 둔다. 신경망은 보정만 배운다")
    p.add_argument("--lr-patience", type=int, default=30,
                   help="검증이 이만큼 안 좋아지면 학습률을 절반으로. 깊은 GAT 는 "
                        "초반에 출렁여서 30 이면 LR 이 먼저 말라 버린다")
    p.add_argument("--jac-alpha", type=float, default=0.0,
                   help="손실을 야코비안 민감도로 가중하는 세기. "
                        "0=균등(기본), 1=민감도 그대로. 오차가 전력으로 크게 "
                        "증폭되는 모선에 벌점을 몰아준다")
    p.add_argument("--amp", default="off", choices=["off", "bf16", "fp16"],
                   help="학습 순전파를 반정밀도로. GAT 의 간선 텐서가 절반이 된다. "
                        "손실은 언제나 float32 로 되돌린 뒤 계산한다")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="GAT 층의 활성값을 들고 있지 않고 역전파 때 다시 계산한다. "
                        "계산이 1/3 늘지만 18층 활성값이 11.6 GB -> 0.6 GB")
    p.add_argument("--val-chunk", type=int, default=0,
                   help="검증을 한 번에 몇 표본씩 볼지. 0=배치와 같게(기본)")
    p.add_argument("--select", default="loss", choices=["loss", "phys"],
                   help="되돌릴 가중치를 고르는 기준. loss=표준화 지도손실, "
                        "phys=검증 분할의 P/부하 %% (미지 N-1 에서는 이쪽)")
    p.add_argument("--lam-sweep", type=float, nargs="+", default=None)
    p.add_argument("--curve", type=int, nargs="+", default=None,
                   help="학습곡선: 이 표본 수들로 각각 학습")
    p.add_argument("--tag", default="", help="결과 파일 이름에 붙일 꼬리표")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    global DEVICE
    torch.set_num_threads(a.threads)
    DEVICE = resolve_device(a.device)
    pre = PRESET.get(a.case, PRESET["case30"])
    n = a.n or pre["n"]
    if a.model == "gat":
        # GAT 는 노드마다 같은 가중치를 쓰므로 폭이 덜 필요하다. MLP 와 같은
        # 256 을 주면 파라미터가 훨씬 커져 공정한 비교가 아니게 된다.
        spec = GATSpec(
            hidden=a.hidden or 128,
            layers=a.layers or pre["layers"],
            heads=a.heads,
            vm_head=a.vm_head,
            residual=not a.no_residual,
            gate=not a.no_gate,
            agg=a.agg,
            skip_init=a.skip_init, skip_freeze=a.skip_freeze,
            checkpoint=a.grad_checkpoint,
        )
    else:
        spec = SurrogateSpec(
            hidden=a.hidden or pre["hidden"],
            layers=a.layers or pre["layers"],
            activation=a.activation,
            vm_head=a.vm_head,
            residual=not a.no_residual,
            skip_init=a.skip_init, skip_freeze=a.skip_freeze,
        )
    base_cfg = dict(
        epochs=a.epochs or pre["epochs"], batch=a.batch, lr=a.lr or pre["lr"],
        lam_warmup=pre["lam_warmup"], lam_ramp=pre["lam_ramp"], seed=a.train_seed,
        patience=a.patience, lr_patience=a.lr_patience,
        select=a.select, jac_alpha=a.jac_alpha,
        amp=a.amp, val_chunk=a.val_chunk,
    )

    ds = load_or_make(a.case, n, a.seed, a.workers)
    split_full = (ds.split_unseen_n1(seed=0) if a.split == "unseen-n1"
                  else ds.split_random(seed=0))
    print(f"\n{a.case}: 표본 {ds.n_samples:,} · 모선 {ds.n_bus} · "
          f"분할 {a.split} (train {len(split_full['train']):,} / "
          f"val {len(split_full['val']):,} / test {len(split_full['test']):,})")
    dev_name = (torch.cuda.get_device_name(0) if DEVICE.type == "cuda"
                else f"CPU ({a.threads} 스레드)")
    kind = ("GAT · 헤드 %d · 집계 %s · 게이팅 %s"
            % (spec.heads, spec.agg, "켬" if spec.gate else "끔")
            if a.model == "gat" else "MLP · %s" % spec.activation)
    print(f"모델: {kind} · hidden {spec.hidden} x {spec.layers}층 · "
          f"vm_head={spec.vm_head} · 선형지름길 "
          f"{'없음' if not spec.residual else spec.skip_init}"
          f"{' (얼림)' if spec.residual and spec.skip_freeze else ''}")
    print(f"장치: {dev_name}"
          + ("  (잔차 측정은 CPU float64)" if DEVICE.type == "cuda" else ""))
    # 학습 설정을 눈에 보이게 찍는다. 06 문서 §5.3 의 비교 규칙("같은 표본 수 ·
    # 같은 분할 · 같은 평가 코드 · 같은 선택 규칙")을 지켰는지는 두 실행의 이
    # 줄을 나란히 놓고 확인하는 것이 제일 빠르다. 실제로 --epochs 를 빠뜨려
    # 기본값 300 으로 돌린 것을 결과가 나온 뒤에야 알아챈 적이 있다.
    print(f"학습: epoch {base_cfg['epochs']} · 배치 {base_cfg['batch']} · "
          f"lr {base_cfg['lr']:g} · patience {base_cfg['patience']}"
          f"(lr {base_cfg['lr_patience']}) · select {base_cfg['select']} · "
          f"jac_alpha {base_cfg['jac_alpha']:g} · 학습시드 {base_cfg['seed']} · "
          f"amp {base_cfg['amp']} · 검증청크 "
          f"{base_cfg['val_chunk'] or base_cfg['batch']}"
          f"{' · 체크포인팅' if getattr(spec, 'checkpoint', False) else ''}\n")

    results: list[dict] = []
    print(HEAD)
    print("-" * len(HEAD))

    # 비교군: 최소제곱 선형 대체모델. 06 부록 §4.1 의 기준선이고 닫힌 형태라
    # 시드도 epoch 도 없다. 신경망은 이걸 넘어야 의미가 있다.
    # 선형 비교군에는 --jac-alpha 를 넘기지 않는다. fit_linear 은 절단선을
    # 고를 때 자기 가중치(1/표준편차)를 따로 계산하므로 넘겨도 효과가 없고,
    # 무엇보다 **기준선은 고정돼 있어야** 가중을 켠 효과를 읽을 수 있다.
    b0, _ = prepare(ds, spec, split=split_full, device=DEVICE)
    t0 = time.time()
    # 비교군도 같은 장치로. 신경망만 옮기고 여기를 빠뜨리면 평가에서 터진다.
    lin = fit_linear(ds, b0.layout, split_full["train"],
                     split_full["val"]).to(DEVICE)
    m_lin = evaluate(lin, b0, split_full["test"])
    print(fmt_row("선형(최소제곱)", m_lin, time.time() - t0, "닫힌해"))
    results.append({"model": "linear", "metrics": m_lin})

    # ------------------------------------------------------------ 학습곡선
    if a.curve:
        lam = a.lam if a.lam is not None else 0.0
        rng = np.random.default_rng(0)
        for size in a.curve:
            if size > len(split_full["train"]):
                print(f"{'n=' + str(size):<14s} (건너뜀 — 학습 분할 {len(split_full['train'])} 보다 큼)")
                continue
            sub = dict(split_full)
            sub["train"] = rng.permutation(split_full["train"])[:size]
            cfg = TrainConfig(lam=lam, **base_cfg)
            t = time.time()
            _, _, r, m = run_one(ds, spec, cfg, split=sub, quiet=True)
            print(fmt_row(f"n={size}", m, time.time() - t, f"ep{r['epochs_run']}"))
            results.append({"n_train": size, "lam": lam, "metrics": m,
                            "best_epoch": r["best_epoch"], "seconds": r["seconds"]})

    # ---------------------------------------------------------- λ 민감도
    elif a.lam_sweep:
        for lam in a.lam_sweep:
            cfg = TrainConfig(lam=lam, **base_cfg)
            t = time.time()
            _, _, r, m = run_one(ds, spec, cfg, split=split_full, quiet=True)
            print(fmt_row(f"λ={lam:g}", m, time.time() - t, f"ep{r['epochs_run']}"))
            results.append({"lam": lam, "metrics": m,
                            "best_epoch": r["best_epoch"], "seconds": r["seconds"]})

    # ------------------------------------------------------------- 단일 학습
    else:
        lam = a.lam if a.lam is not None else 0.0
        cfg = TrainConfig(lam=lam, **base_cfg)
        t = time.time()
        model, b, r, m = run_one(ds, spec, cfg, split=split_full)
        print()
        print(HEAD)
        print("-" * len(HEAD))
        print(fmt_row(f"λ={lam:g}", m, time.time() - t, f"ep{r['epochs_run']}"))
        results.append({"lam": lam, "metrics": m,
                        "best_epoch": r["best_epoch"], "seconds": r["seconds"],
                        # 학습곡선을 나중에 그리려면 이게 있어야 한다.
                        # 실험이 끝난 뒤 "그때 손실이 어땠지" 를 다시 물을 수 없다.
                        "history": r["history"]})
        if not a.no_save:
            ck = ROOT / "results" / f"{a.case}{a.tag}_{a.model}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "spec": spec, "case": a.case}, ck)
            print(f"\n체크포인트: {ck.relative_to(ROOT)}")

    if not a.no_save:
        out = ROOT / "results" / f"{a.case}{a.tag}_{a.model}.json"
        save_run(out, {
            "case": a.case, "n_samples": int(ds.n_samples), "data_seed": a.seed,
            "split": a.split, **spec_config_dict(spec, TrainConfig(**base_cfg)),
            "results": results,
        })
        print(f"결과: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
