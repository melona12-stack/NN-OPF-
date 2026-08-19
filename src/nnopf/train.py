r"""대체모델 학습·평가 (4단계).

데이터가 작아서(case118 20,000 표본 = 48 MB) ``DataLoader`` 대신 전체를
텐서로 올려 두고 ``randperm`` 으로 배치를 자른다. CPU 4코어에서 워커 오버헤드가
연산보다 큰 규모라 이쪽이 훨씬 빠르다.

두 가지 원칙
------------
**① 통계는 학습 분할에서만 뽑는다.** 정규화 평균·표준편차, 전압 박스 모두.
검증·시험 분할을 보고 정한 값이 하나라도 섞이면 성능이 부풀려진다
(03 문서 §3.2).

**② 최종 잔차는 float64 로 잰다.** 전압 오차는 :math:`Y_{bus}` 를 거치며
:math:`\max|Y|` 배로 증폭되므로, float32 로 재면 측정 바닥이
case118 기준 4.6e-05 까지 올라온다 (05 문서 §6.1). 학습은 float32,
보고는 float64.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from nnopf.case import PowerSystem, load_case
from nnopf.dataset import PowerFlowDataset
from nnopf.models import IOLayout, PowerFlowMLP, SurrogateSpec
from nnopf.physics_torch import ACPhysics

__all__ = ["TrainConfig", "prepare", "train", "evaluate", "lambda_at",
           "supervised_loss", "input_stats"]


@dataclass(frozen=True)
class TrainConfig:
    """학습 하이퍼파라미터.

    .. warning::
       ``lr`` 은 P2 의 5e-4 가 아니라 **2e-3** 이다. P2 의 값은 정규화 층이 있는
       GAT 기준이고, 우리 MLP 에서 5e-4 를 쓰면 학습이 MSE 1e-4 에서 멈춘다
       (선형회귀보다도 못한 :math:`R^2` 0.87). 게다가 ``ReduceLROnPlateau`` 가
       그 조기 플래토를 감지해 LR 을 더 깎으면서 악순환이 된다.
       2e-3 으로 올리면 같은 모델·같은 epoch 수에서 검증 MSE 가 **45배** 내려간다.
       ``lr_patience`` 를 30 으로 늘린 것도 같은 이유다.
    """

    epochs: int = 300
    batch: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip: float = 2.0

    # 물리정보 손실 (P2 §3.3, §4.1). lam=0 이면 순수 지도학습 기준선.
    lam: float = 0.0
    lam_warmup: int = 20
    lam_ramp: int = 50

    # 조기 종료는 넉넉하게. 검증 표본이 900개뿐이라 검증 손실이 잡음을 타는데,
    # patience 60 이면 개선이 이어지는 중에도 끊긴다 (case30 에서 epoch 184 에
    # 끊겨 Vm MAE 가 1.8배 나빴다). 최고 검증 시점 가중치를 되돌리므로
    # 끝까지 돌려도 과적합 위험은 없다.
    patience: int = 200
    lr_patience: int = 30
    seed: int = 0


def supervised_loss(
    model, Vm: torch.Tensor, Va: torch.Tensor,
    Vm_true: torch.Tensor, Va_true: torch.Tensor,
) -> torch.Tensor:
    r"""**표준화 공간**의 지도 손실.

    오차를 모선별 학습 표준편차로 나눈 뒤 MSE 를 잰다. 이유는 두 가지다.

    1. 원단위 MSE 는 :math:`10^{-4}` 규모라 기울기도 그만큼 작아서, 정규화
       항이 과제 기울기를 눌러 버린다 (models.py 상단의 실측표 참조).
    2. Vm 과 Va 의 분산이 10배 차이라 원단위로는 Va 가 손실을 지배한다.
       표준화하면 둘이 대등하게 반영된다.
    """
    dv = (Vm[:, model.pq_idx] - Vm_true[:, model.pq_idx]) * model.vm_w
    da = (Va[:, model.va_idx] - Va_true[:, model.va_idx]) * model.va_w
    return (dv**2).mean() + (da**2).mean()


def lambda_at(epoch: int, cfg: TrainConfig) -> float:
    r"""λ 워밍업-램프 스케줄 (P2 §4.1).

    초기에는 전압 예측이 엉망인데 그 값을 조류방정식에 넣으면 잔차 기울기가
    폭주한다. 그래서 warmup 동안 λ=0 으로 지도학습만 하고, ramp 구간에서
    선형으로 올린다.
    """
    if cfg.lam <= 0 or epoch < cfg.lam_warmup:
        return 0.0
    if epoch >= cfg.lam_warmup + cfg.lam_ramp:
        return cfg.lam
    return cfg.lam * (epoch - cfg.lam_warmup + 1) / cfg.lam_ramp


# --------------------------------------------------------------------------
# 준비
# --------------------------------------------------------------------------
@dataclass
class Bundle:
    """학습에 필요한 텐서와 메타를 한 덩어리로."""

    sys: PowerSystem
    layout: IOLayout
    split: dict[str, np.ndarray]
    X: torch.Tensor          # (N, in_dim)
    Vm: torch.Tensor         # (N, nb) 라벨
    Va: torch.Tensor
    p_spec: torch.Tensor     # (N, nb)
    q_spec: torch.Tensor
    outage: torch.Tensor     # (N,)
    physics: ACPhysics       # float32 (학습용)
    physics64: ACPhysics     # float64 (평가용)


def input_stats(X: np.ndarray, tr: np.ndarray, n_phys: int) -> tuple[np.ndarray, np.ndarray]:
    r"""입력 정규화 통계. **선로상태 열은 정규화하지 않는다.**

    상태 열은 이미 0/1 이라 스케일이 맞고, 표준화하면 다음 함정에 빠진다.

    .. danger::
       어떤 선로가 **학습 분할에서 한 번도 고장 나지 않으면** 그 열의 표준편차가
       0 이다. 여기에 하한 :math:`10^{-6}` 을 씌워 나누면, 그 선로가 고장 난
       시험 표본에서 정규화 입력이 :math:`10^{6}` 으로 폭주한다.

       그리고 이건 드문 사고가 아니다 — **미지 N-1 분할은 정의상 시험용 고장이
       학습에 한 번도 안 나오게 만든다**(05 문서 §9). 즉 P2 의 핵심 일반화
       실험이 통째로 망가진다. 실제로 case30 · 400 표본에서 41개 선로 중
       12개가 학습 분할에서 상수였고, 그중 4개가 시험에서 변했다.

    물리량 열은 그대로 표준화하되, 상수 열은 나누지 않고 중심화만 한다.
    """
    mean = X[tr].mean(0)
    std = X[tr].std(0)
    std = np.where(std > 1e-8, std, 1.0)      # 상수 열: 중심화만, 확대 금지
    mean[n_phys:] = 0.0                        # 선로상태: 손대지 않는다
    std[n_phys:] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def prepare(
    ds: PowerFlowDataset,
    spec: SurrogateSpec,
    split: dict[str, np.ndarray] | None = None,
    case: str | None = None,
    seed: int = 0,
) -> tuple[Bundle, PowerFlowMLP]:
    """데이터셋에서 텐서 묶음과 (초기화된) 모델을 만든다.

    ``seed`` 는 **가중치 초기화**를 고정한다. 학습 시드(``TrainConfig.seed``)는
    배치 순서만 정하므로, 둘 다 고정해야 완전히 재현된다.
    """
    torch.manual_seed(seed)
    sysm = load_case(case or ds.case)
    layout = IOLayout(sysm)
    split = split if split is not None else ds.split_random(seed=0)
    tr = split["train"]

    X = layout.inputs(ds)
    # p_spec 은 float64 로 캐스팅한 뒤 더한다. float32 로 더하면 반올림이
    # 1e-8 수준으로 남아 잔차 측정 바닥을 올린다.
    f64 = lambda a: np.asarray(a, dtype=np.float64)
    p_spec = f64(ds.p_gen) + f64(ds.p_ren) - f64(ds.Pd)
    q_spec = -f64(ds.Qd)

    # --- 정규화·박스 통계: 학습 분할에서만 ---
    in_mean, in_std = input_stats(X, tr, n_phys=4 * layout.nb)

    vm_tr = ds.Vm[tr][:, layout.pq]
    va_tr = ds.Va[tr][:, layout.nonslack]
    if spec.vm_head == "scaled":
        lo, hi = vm_tr.min(0), vm_tr.max(0)
        pad = np.maximum((hi - lo) * spec.vm_margin, 1e-3)
        vm_lo, vm_hi = lo - pad, hi + pad
    else:
        vm_lo = vm_tr.mean(0)                     # raw 모드: (평균, 표준편차)
        vm_hi = np.maximum(vm_tr.std(0), 1e-6)

    model = PowerFlowMLP(
        layout, spec,
        in_mean=in_mean, in_std=in_std,
        v_set=ds.v_set,
        vm_lo=vm_lo, vm_hi=vm_hi,
        va_mean=va_tr.mean(0), va_std=np.maximum(va_tr.std(0), 1e-6),
        vm_w=1.0 / np.maximum(vm_tr.std(0), 1e-6),
        va_w=1.0 / np.maximum(va_tr.std(0), 1e-6),
    )

    t = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a), dtype=d)
    # 물리 손실 무차원화 기준: 학습 분할의 모선별 지정주입 RMS
    ph32 = ACPhysics(sysm, torch.float32)
    ph32.set_scale(
        t(np.sqrt((p_spec[tr] ** 2).mean(0))), t(np.sqrt((q_spec[tr] ** 2).mean(0)))
    )
    bundle = Bundle(
        sys=sysm, layout=layout, split=split,
        X=t(X), Vm=t(ds.Vm), Va=t(ds.Va),
        p_spec=t(p_spec), q_spec=t(q_spec),
        outage=t(ds.outage, torch.long),
        physics=ph32,
        physics64=ACPhysics(sysm, torch.float64),
    )
    return bundle, model


# --------------------------------------------------------------------------
# 평가
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: PowerFlowMLP, b: Bundle, idx: np.ndarray, chunk: int = 2048
) -> dict:
    """전압 오차와 **물리 잔차**를 잰다. 잔차는 float64.

    보고 항목은 P2 Table 6/7 과 맞췄다 — 전압 MAE, P/Q 불일치, 부하 대비 비율.
    """
    model.eval()
    ii = torch.as_tensor(idx, dtype=torch.long)
    ph = b.physics64

    dvm, dva, dP, dQ = [], [], [], []
    for s in range(0, len(ii), chunk):
        j = ii[s : s + chunk]
        Vm, Va = model(b.X[j])
        Vm64, Va64 = Vm.double(), Va.double()
        dvm.append((Vm64 - b.Vm[j].double()).abs())
        dva.append((Va64 - b.Va[j].double()).abs())
        rp, rq = ph.residual(
            Vm64, Va64, b.p_spec[j].double(), b.q_spec[j].double(), b.outage[j]
        )
        dP.append(rp.abs())
        dQ.append(rq.abs())

    dvm, dva = torch.cat(dvm), torch.cat(dva)
    dP, dQ = torch.cat(dP), torch.cat(dQ)
    load = b.p_spec[ii].double().abs().sum(-1).mean().clamp(min=1e-9)

    # 전압 한계 위반 (계통 기준)
    Vmin = torch.as_tensor(b.sys.Vmin, dtype=torch.float64)
    Vmax = torch.as_tensor(b.sys.Vmax, dtype=torch.float64)
    Vm_all = torch.cat(
        [model(b.X[ii[s : s + chunk]])[0].double() for s in range(0, len(ii), chunk)]
    )
    viol = ((Vm_all < Vmin) | (Vm_all > Vmax)).double()

    return {
        "n": int(len(idx)),
        "vm_mae": dvm.mean().item(),
        "vm_max": dvm.max().item(),
        "va_mae": dva.mean().item(),
        "va_max": dva.max().item(),
        "p_mismatch": dP.sum(-1).mean().item(),   # 표본당 |ΔP| 합 [pu]
        "q_mismatch": dQ.sum(-1).mean().item(),
        "p_mismatch_max": dP.max().item(),
        "q_mismatch_max": dQ.max().item(),
        "p_over_load_pct": (dP.sum(-1).mean() / load).item() * 100,
        "residual_max": max(dP.max().item(), dQ.max().item()),
        "vlim_viol_pct": viol.mean().item() * 100,
    }


# --------------------------------------------------------------------------
# 학습
# --------------------------------------------------------------------------
def train(
    model: PowerFlowMLP,
    b: Bundle,
    cfg: TrainConfig,
    verbose: bool = True,
    log_every: int = 25,
) -> dict:
    """학습 루프. 최고 검증 손실 시점의 가중치를 되돌려 놓고 이력을 반환한다."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    tr = torch.as_tensor(b.split["train"], dtype=torch.long)
    va = torch.as_tensor(b.split["val"], dtype=torch.long)
    # AdamW(분리형 감쇠). 일반 Adam 의 weight_decay 는 L2 를 기울기에 더하는
    # 방식이라 손실이 작을 때 과제 기울기를 눌러 버린다 (models.py 상단 주석).
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=cfg.lr_patience
    )

    best = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    hist: list[dict] = []
    t0 = time.time()

    for ep in range(cfg.epochs):
        lam = lambda_at(ep, cfg)
        model.train()
        perm = tr[torch.randperm(len(tr))]
        tot = n_seen = 0.0

        for s in range(0, len(perm), cfg.batch):
            j = perm[s : s + cfg.batch]
            Vm, Va = model(b.X[j])
            sup = supervised_loss(model, Vm, Va, b.Vm[j], b.Va[j])
            loss = sup
            if lam > 0:
                loss = loss + lam * b.physics.loss(
                    Vm, Va, b.p_spec[j], b.q_spec[j], b.outage[j]
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += sup.item() * len(j)
            n_seen += len(j)

        # 검증은 항상 순수 지도손실로 — λ 가 바뀌어도 비교 가능해야 한다
        model.eval()
        with torch.no_grad():
            Vm, Va = model(b.X[va])
            vloss = supervised_loss(model, Vm, Va, b.Vm[va], b.Va[va]).item()
        sched.step(vloss)
        hist.append(
            {"epoch": ep, "train": tot / n_seen, "val": vloss, "lam": lam,
             "lr": opt.param_groups[0]["lr"]}
        )

        if vloss < best * (1 - 1e-5):
            best, best_epoch = vloss, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif ep - best_epoch >= cfg.patience:
            if verbose:
                print(f"  조기 종료 (epoch {ep}, 최고 {best_epoch})")
            break

        if verbose and (ep % log_every == 0 or ep == cfg.epochs - 1):
            print(
                f"  ep {ep:4d}  train {tot/n_seen:.3e}  val {vloss:.3e}"
                f"  λ {lam:.1e}  lr {opt.param_groups[0]['lr']:.1e}"
            )

    model.load_state_dict(best_state)
    return {
        "history": hist,
        "best_epoch": best_epoch,
        "best_val": best,
        "seconds": time.time() - t0,
        "epochs_run": len(hist),
    }


def save_run(path: str | Path, payload: dict) -> None:
    """결과 JSON 저장. 설정과 지표를 항상 같이 남긴다 (00 문서 §9)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def spec_config_dict(spec: SurrogateSpec, cfg: TrainConfig) -> dict:
    return {"spec": asdict(spec), "train": asdict(cfg)}


# --------------------------------------------------------------------------
# 추론 속도
# --------------------------------------------------------------------------
@torch.no_grad()
def benchmark(
    model, b: Bundle, idx: np.ndarray, n_single: int = 200, batch: int = 64
) -> dict:
    """뉴턴-랩슨 대비 추론 속도. **단건과 배치를 나눠 잰다.**

    P2 가 관찰한 대로 그래프가 작으면 단건 추론에서 가속 효과가 크게 줄어든다
    (04 문서 §2.8). 하나로 뭉뚱그린 배속 숫자는 리뷰어가 바로 지적한다.
    """
    from nnopf.powerflow import solve_power_flow
    from nnopf.ybus import make_ybus

    model.eval()
    ii = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    sub = ii[:n_single]

    # 뉴턴-랩슨 — 상정사고가 없는 표본만 (Ybus 재사용이 공정)
    import dataclasses

    Ybus = make_ybus(b.sys)
    plain = [int(k) for k in sub.tolist() if int(b.outage[k]) < 0][:n_single]
    zero_g = np.zeros(len(b.sys.Pg0))
    t0 = time.perf_counter()
    for k in plain:
        # 지정주입을 '음의 부하'로 넣는다. solve_power_flow 는
        #   Psp = Cg @ Pg - Pd  로 지정값을 만들므로 Pg=0, Pd=-p_spec 이면
        # 정확히 데이터셋과 같은 문제가 된다 (발전/부하 분해는 조류계산에 무관).
        sysk = dataclasses.replace(
            b.sys,
            Pd=-b.p_spec[k].numpy().astype(float),
            Qd=-b.q_spec[k].numpy().astype(float),
            Qg0=zero_g,
        )
        solve_power_flow(sysk, Pg=zero_g, Ybus=Ybus, tol=1e-8)
    nr_ms = (time.perf_counter() - t0) / max(len(plain), 1) * 1e3

    # 대체모델 단건
    t0 = time.perf_counter()
    for k in sub.tolist():
        model(b.X[k : k + 1])
    single_ms = (time.perf_counter() - t0) / len(sub) * 1e3

    # 대체모델 배치
    reps = max(1, 2000 // batch)
    t0 = time.perf_counter()
    for r in range(reps):
        s = (r * batch) % max(len(ii) - batch, 1)
        model(b.X[ii[s : s + batch]])
    batch_ms = (time.perf_counter() - t0) / (reps * batch) * 1e3

    return {
        "nr_ms": nr_ms,
        "single_ms": single_ms,
        "batch_ms": batch_ms,
        "speedup_single": nr_ms / max(single_ms, 1e-9),
        "speedup_batch": nr_ms / max(batch_ms, 1e-9),
        "batch_size": batch,
    }
