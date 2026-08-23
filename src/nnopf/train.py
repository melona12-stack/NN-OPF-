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

import contextlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from nnopf.case import PowerSystem, load_case
from nnopf.dataset import PowerFlowDataset
from nnopf.models import IOLayout, PowerFlowMLP, SurrogateSpec
from nnopf.physics_torch import ACPhysics

__all__ = ["TrainConfig", "prepare", "train", "evaluate", "lambda_at",
           "resolve_device",
           "supervised_loss", "input_stats", "jacobian_weights",
           "init_skip_lstsq"]


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
    # 배치 256. GPU 에서 64 는 손해다 — 갱신 한 번 시간이 배치에 거의
    # 무관해서(06 문서 §8.1), 작은 배치는 그 고정비만 여러 번 낸다.
    # 그렇다고 4096 까지 올리면 epoch 당 갱신이 몇 번 안 남는다.
    batch: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip: float = 2.0

    # 물리정보 손실 (P2 §3.3, §4.1). lam=0 이면 순수 지도학습 기준선.
    #
    # lam 은 **지도 항 대비 상대 가중치**다. 램프가 시작되는 시점에 두 항의
    # 비를 재서 나눠 주므로 lam=1 이면 그 순간 둘이 같은 크기가 된다.
    # 이 환산이 없으면 같은 숫자가 계통마다 전혀 다른 뜻이 된다 — 실측으로
    # 물리/지도 비가 case30 은 2.2e3, case118 은 2.1e7 이었다.
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

    # 중간 저장. 긴 학습(GAT 8층이 55분, 12층이 2.5시간)이 끊기면 통째로
    # 날아간다 — 실제로 터미널을 잘못 닫아 55분을 잃은 적이 있다.
    # ckpt_path 를 주면 ckpt_every epoch 마다 상태를 저장하고, 같은 경로에
    # 파일이 있으면 **거기서 이어서** 돌린다. 학습이 끝나면 지운다.
    #
    # 재현성을 지키려고 난수 상태(torch·numpy)까지 같이 저장한다. 안 그러면
    # 이어달린 결과가 한 번에 돌린 결과와 달라져 비교가 깨진다.
    ckpt_path: str | None = None
    ckpt_every: int = 50
    # 중간 저장본이 있어도 **명시할 때만** 이어서 돌린다. 자동으로 이으면
    # 하이퍼파라미터를 바꾸고 같은 태그로 돌렸을 때 옛 학습이 조용히
    # 되살아난다. 그래도 설정이 다르면 거부하도록 지문을 같이 저장한다.
    resume: bool = False

    # 학습을 멈출 시점과 되돌릴 가중치를 **무엇으로 고를 것인가**.
    #
    #   "loss" — 표준화 지도손실. 무작위 분할에서는 이걸로 충분하다.
    #   "phys" — 검증 분할의 P/부하 % (float32). 우리가 실제로 보고하는 값이다.
    #
    # 미지 N-1 에서는 "loss" 가 망가진다. 검증 분할에도 학습에서 못 본 고장이
    # 들어 있어(dataset.py::split_unseen_n1) 검증 손실이 몇 epoch 만에 바닥에
    # 닿고 그 뒤로 안 움직인다. 그러면 "최고 검증 시점" 이 거의 학습되지 않은
    # 초반 epoch 이 되어 버린다 — GAT 는 1,500 중 **34** 가 뽑혔고, 그때 학습
    # 손실은 끝까지 갔을 때보다 33배 나빴다. 검증이 2% 나빠지는 것을 아끼려고
    # 33배를 버린 셈이다.
    #
    # 06 문서 §4.1 이 "검증 손실은 성능을 읽는 지표로 쓰면 안 된다" 고 적었는데,
    # 미지 N-1 에서는 **멈출 시점을 정하는 지표로도** 못 쓴다.
    select: str = "loss"

    # 손실을 **야코비안 민감도**로 가중할 세기 (:func:`jacobian_weights`).
    #   0 — 지금까지의 균등 가중. 기본값이자 절제 실험의 대조군이다.
    #   1 — 민감도를 그대로. 오차가 전력으로 크게 증폭되는 모선에 벌점을 몰아준다.
    # 06 문서 §2.2·§7.7.1·§7.8.1 이 세 번 같은 곳을 가리켜서 넣었다 —
    # "위상은 더 정확한데 물리 잔차는 더 나쁘다"가 세 실험에서 반복됐다.
    jac_alpha: float = 0.0

    # 검증을 한 번에 몇 표본씩 볼 것인가. **0 이면 batch 와 같게 쓴다.**
    #
    # 예전에는 검증 분할 전체를 한 방에 넣었다. MLP 에서는 아무 문제가
    # 없었지만 GAT 에서는 치명적이다 — case118 미지 N-1 의 검증 분할이
    # 9,000 표본이고, 간선 텐서 ``(B, E, H, D)`` 가 그 크기면 **한 개에
    # 4.5 GB** 다. 동시에 대여섯 개가 살아 있으니 22 GB 를 요구한다.
    # 8 GB 카드에서 이게 매 epoch 반복되면 할당기가 계속 캐시를 비운다.
    # 학습 배치가 들어가는 크기면 검증도 들어간다 — 그래서 기본이 batch 다.
    val_chunk: int = 0

    # 학습 순전파를 반정밀도로 돌린다. "off" | "bf16".
    #
    # 간선 텐서가 절반이 되므로 GAT 의 활성값이 그대로 절반이 된다.
    # **손실 계산은 항상 float32 로 되돌린 뒤에 한다** — 물리 잔차는 전압
    # 오차가 max|Y| 배로 증폭되는 양이라 bf16(유효숫자 약 3자리)으로는
    # 아예 잴 수가 없다 (05 문서 §6.1 이 float32 로도 겪은 문제다).
    # **fp16 은 일부러 넣지 않았다.** fp16 은 지수부가 좁아 기울기가 조용히
    # 언더플로하므로 ``GradScaler`` 가 필수인데, 스케일러는 손실 스케일을
    # 자동 조정하다 가끔 배치를 통째로 건너뛴다 — 재현성이 깨진다.
    # bf16 은 지수부가 float32 와 같아서 스케일러 없이 그냥 된다.
    amp: str = "off"


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


def autocast_ctx(device, amp: str):
    """``cfg.amp`` 를 :func:`torch.autocast` 문맥으로 바꾼다. "off" 면 무동작."""
    if amp == "off":
        return contextlib.nullcontext()
    if amp != "bf16":
        raise ValueError(f'amp 는 "off" | "bf16" 중 하나여야 한다: {amp!r}')
    if torch.device(device).type != "cuda":
        # CPU autocast 는 이득이 없고 bf16 커널이 없는 연산에서 느려지기만 한다.
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


@torch.no_grad()
def validate(model, b: "Bundle", va: torch.Tensor, cfg: TrainConfig,
             val_load: torch.Tensor) -> tuple[float, float]:
    """검증 분할의 (지도손실, P/부하 %) 를 **청크로 나눠** 잰다.

    한 방에 넣지 않는 이유는 :attr:`TrainConfig.val_chunk` 주석에 있다.
    청크로 나눠도 값은 정확히 같다 — 지도손실은 청크 크기로 가중평균하고,
    물리 잔차는 표본별 합을 누적한 뒤 마지막에 한 번만 나눈다.
    """
    chunk = cfg.val_chunk or cfg.batch
    n = len(va)
    sup_sum = 0.0
    res_sum = torch.zeros((), device=b.p_spec.device, dtype=torch.float32)
    for s in range(0, n, chunk):
        k = va[s : s + chunk]
        Vm, Va = model(b.X[k])
        Vm, Va = Vm.float(), Va.float()
        sup_sum += supervised_loss(model, Vm, Va, b.Vm[k], b.Va[k]).item() * len(k)
        if cfg.select == "phys":
            rp, _ = b.physics.residual(
                Vm, Va, b.p_spec[k], b.q_spec[k], b.outage[k]
            )
            res_sum = res_sum + rp.abs().sum(-1).sum()
    vloss = sup_sum / max(n, 1)
    vphys = float("nan")
    if cfg.select == "phys":
        vphys = (res_sum / max(n, 1) / val_load).item() * 100.0
    return vloss, vphys


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
    physics: ACPhysics       # float32 (학습용) — device 위
    physics64: ACPhysics     # float64 (평가용) — 항상 CPU
    device: torch.device = torch.device("cpu")
    # 지름길을 최소제곱으로 초기화했으면 그 기록 (고른 절단선 등). 안 했으면 빈 dict.
    skip_init: dict = field(default_factory=dict)


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


def jacobian_weights(
    sysm: PowerSystem,
    layout: IOLayout,
    Vm_ref: np.ndarray,
    Va_ref: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""전압 오차가 **전력 잔차로 얼마나 증폭되는지**를 재서 손실 가중치를 만든다.

    지금 손실은 모든 모선의 전압 오차를 똑같이 벌한다. 그런데 06 문서 §2.2 에서
    야코비안으로 분해해 보니, 같은 크기의 오차라도 **어느 모선에 있느냐**에 따라
    전력 잔차 기여가 1.7 배까지 달라졌다. 균등 가중 MSE 는 그걸 모른다.

    그래서 각 예측 변수의 **민감도**를 잰다. 변수 :math:`x_j` (모선 j 의 위상
    또는 전압크기) 를 조금 움직였을 때 잔차 벡터 전체가 얼마나 움직이는가 —
    즉 잔차 야코비안의 **j 번째 열의 크기**다.

    .. math::

        s_j = \left\| \frac{\partial r}{\partial x_j} \right\|_2 ,\qquad
        r = \big[\Delta P_{\text{비슬랙}};\ \Delta Q_{PQ}\big]

    잔차를 거는 자리는 05 문서 §7 과 같다 — 슬랙에는 :math:`\Delta P` 를 걸지
    않고, PV 에는 :math:`\Delta Q` 를 걸지 않는다. 두 블록은 단위가 다르므로
    **각각 RMS 로 나눠** 대등하게 만든 뒤 합친다.

    Parameters
    ----------
    Vm_ref, Va_ref
        민감도를 잴 기준 운전점. **학습 분할 라벨의 평균**을 넣는다.
        야코비안은 운전점마다 달라지지만, 정상 운전 영역에서 조류방정식이
        거의 선형이라(06 문서 §2.1) 구조적 민감도는 기준점 하나로 충분하다.
    alpha
        가중 세기. ``0`` 이면 곱수가 전부 1 이라 **기존 손실과 완전히 같다**
        (절제 실험의 대조군). ``1`` 이면 민감도를 그대로 쓴다.

    Returns
    -------
    (m_vm, m_va)
        각각 ``len(layout.pq)``, ``len(layout.nonslack)`` 길이의 곱수.
        :math:`\overline{m^2} = 1` 로 맞춰 두므로 **손실의 크기가 변하지 않는다**
        — 06 문서 §7.2 의 정규화 함정을 다시 밟지 않기 위해서다.

    Notes
    -----
    N-1 표본에서는 실제 토폴로지가 다르지만 기저 :math:`Y_{bus}` 로 잰다.
    선로 하나가 빠져도 "어느 모선이 뻣뻣한가"라는 구조는 거의 그대로이고,
    표본마다 다시 재면 학습 전 준비가 계통 크기에 비례해 무거워진다.
    """
    from nnopf.powerflow import dSbus_dV

    Ybus = sysm.ybus()
    V = np.asarray(Vm_ref, float) * np.exp(1j * np.asarray(Va_ref, float))
    dS_dVa, dS_dVm = dSbus_dV(Ybus, V)

    non_slack = layout.nonslack          # ΔP 를 거는 행
    pq = layout.pq                       # ΔQ 를 거는 행

    def _sens(dS) -> np.ndarray:
        M = np.asarray(dS.todense()) if hasattr(dS, "todense") else np.asarray(dS)
        blk_p = np.real(M[non_slack, :])
        blk_q = np.imag(M[pq, :])
        rp = float(np.sqrt((blk_p**2).mean())) or 1.0
        rq = float(np.sqrt((blk_q**2).mean())) or 1.0
        return np.sqrt(((blk_p / rp) ** 2).sum(0) + ((blk_q / rq) ** 2).sum(0))

    s_va = _sens(dS_dVa)[non_slack]       # θ 는 비슬랙만 예측한다
    s_vm = _sens(dS_dVm)[pq]              # |V| 는 PQ 만 예측한다

    def _norm(s: np.ndarray) -> np.ndarray:
        m = np.power(np.maximum(s, 1e-12), float(alpha))
        rms = float(np.sqrt((m**2).mean()))
        return (m / rms).astype(np.float32) if rms > 0 else np.ones_like(m, np.float32)

    return _norm(s_vm), _norm(s_va)


def _zero_output_head(model) -> bool:
    """본체의 **마지막 층**을 0 으로 만든다. 그러면 출력이 지름길 하나만 남는다.

    :func:`init_skip_lstsq` 가 부르는 보조 함수다. 지름길만 최소제곱으로 채우고
    본체를 무작위 초기화 그대로 두면, 학습 첫 순간의 출력이 **선형 해 + 무작위
    잡음** 이 된다. 실측하면 case30 미지 N-1 에서 P/부하 17.9 % 로, 선형 기준선
    2.95 % 와 한참 멀다. 본체 마지막 층까지 0 으로 눌러야 출발점이 정확히
    선형 해가 된다.
    """
    with torch.no_grad():
        if hasattr(model, "net"):                    # PowerFlowMLP
            last = [m for m in model.net if isinstance(m, torch.nn.Linear)][-1]
            last.weight.zero_(); last.bias.zero_()
            return True
        if hasattr(model, "head_w"):                 # PowerFlowGAT
            model.head_w.zero_(); model.head_b.zero_()
            return True
    return False


def init_skip_lstsq(
    model, X: np.ndarray, Vm: np.ndarray, Va: np.ndarray,
    tr: np.ndarray, va: np.ndarray | None = None,
    zero_body: bool = True,
) -> dict:
    r"""선형 지름길을 **최소제곱 해로 초기화**한다.

    ``models.py`` 는 지름길의 목적을 이렇게 적어 두었다 — "신경망은 사상 전체가
    아니라 **선형에 대한 보정만** 배우면 된다". 그런데 지금 지름길은 **0 으로
    초기화되어 본체와 함께 학습**된다. 즉 그 분업은 **구조가 아니라 희망**이다.
    신경망은 여전히 선형 부분을 처음부터 다시 배워야 하고, 실제로 case30 미지
    N-1 에서 MLP 4.00 % 가 선형 2.05 % 를 못 이긴다 (06 문서 §7.8.1).

    그런데 그 선형 해는 **이미 닫힌 형태로 갖고 있다.** 여기서 출발시킨다.

    핵심은 **모델 자신의 출력 공간에서** 최소제곱을 푼다는 것이다. 지름길은
    정규화 입력 :math:`z` 를 받아 헤드 **직전** 값 ``(vm_raw, va_raw)`` 를
    내놓으므로, 목표도 그 자리로 옮겨 놓아야 한다.

    .. math::

        t^{vm} = \mathrm{logit}\!\left(\frac{|V| - V^{lo}}{V^{hi}-V^{lo}}\right),
        \qquad
        t^{va} = \frac{\theta - \bar\theta}{\sigma_\theta}

    ``vm_head="scaled"`` 의 시그모이드를 거꾸로 통과시키는 것이다. 상자에 여유가
    5 % 있어서(``vm_margin``) 라벨이 항상 안쪽이므로 로짓이 발산하지 않는다.

    절단선은 ``val`` 로 고른다 — 06 문서 §7.6 에서 상수열 때문에 설계행렬이
    무너지는 것을 겪었고, 여기 :math:`z` 에도 같은 상수열이 그대로 있다.

    Parameters
    ----------
    tr, va
        학습 / 검증 분할 인덱스. **학습 분할로만 적합하고 검증으로 절단선만
        고른다.** 시험 분할은 보지 않는다.

    Returns
    -------
    dict
        ``{"rcond": 고른 절단선, "val_mse": 그때 검증 오차, "n_col": 쓴 열 수}``.
        아무 일도 안 했으면 빈 dict (지름길이 없는 모델).
    """
    if getattr(model, "skip", None) is None:
        return {}

    dev = model.skip.weight.device
    f64 = lambda t: t.detach().double().cpu().numpy()
    Xa = np.asarray(X, np.float64)
    z = (Xa - f64(model.in_mean)) / f64(model.in_std)

    pq = f64(model.pq_idx).astype(int)
    vi = f64(model.va_idx).astype(int)
    lo, hi = f64(model.vm_lo), f64(model.vm_hi)
    am, asd = f64(model.va_mean), f64(model.va_std)

    def targets(idx: np.ndarray) -> np.ndarray:
        vm = np.asarray(Vm, np.float64)[idx][:, pq]
        if model.spec.vm_head == "scaled":
            u = np.clip((vm - lo) / np.maximum(hi - lo, 1e-12), 1e-6, 1 - 1e-6)
            t_vm = np.log(u / (1.0 - u))          # 시그모이드의 역함수
        else:
            t_vm = (vm - lo) / np.maximum(hi, 1e-12)
        t_va = (np.asarray(Va, np.float64)[idx][:, vi] - am) / asd
        return np.concatenate([t_vm, t_va], axis=1)

    # **상수열을 뺀다.** 06 문서 §7.6 에서 겪은 그대로다 — 상수열을 그냥 두면
    # 설계행렬 조건수가 무너지고, 검증 손실은 거의 같은데 물리 잔차만 수십 배
    # 커지는 방향이 살아남는다. 실제로 이 함수도 처음엔 상수열을 안 뺐고,
    # 그 결과 초기화 직후 P/부하가 7.3 % (상수열 제거 후 3.0 %) 였다.
    keep = np.flatnonzero(z[tr].std(0) > 0)
    A = np.c_[z[tr][:, keep], np.ones(len(tr))]
    T = targets(tr)

    best_W, best_rc, best_mse = None, None, np.inf
    if va is None or len(va) == 0:
        best_W, *_ = np.linalg.lstsq(A, T, rcond=None)
        best_rc = None
    else:
        Av, Tv = np.c_[z[va][:, keep], np.ones(len(va))], targets(va)
        for rc in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
            W, *_ = np.linalg.lstsq(A, T, rcond=rc)
            mse = float(((Av @ W - Tv) ** 2).mean())
            if mse < best_mse:
                best_W, best_rc, best_mse = W, rc, mse

    # 뺐던 상수열 자리는 0 으로 되돌려 넣는다 (그 열은 어차피 z=0 이라 무해하다).
    Wfull = np.zeros((z.shape[1] + 1, best_W.shape[1]))
    Wfull[keep] = best_W[:-1]
    Wfull[-1] = best_W[-1]

    with torch.no_grad():
        model.skip.weight.copy_(
            torch.as_tensor(Wfull[:-1].T, dtype=model.skip.weight.dtype, device=dev)
        )
        model.skip.bias.copy_(
            torch.as_tensor(Wfull[-1], dtype=model.skip.bias.dtype, device=dev)
        )
    zeroed = _zero_output_head(model) if zero_body else False
    return {"rcond": best_rc, "val_mse": None if best_mse == np.inf else best_mse,
            "n_col": int(A.shape[1]), "n_dropped": int(z.shape[1] - len(keep)),
            "body_zeroed": zeroed}


def resolve_device(spec: str = "auto") -> torch.device:
    """``"auto" | "cpu" | "cuda"`` 를 실제 장치로.

    ``auto`` 는 쓸 수 있으면 GPU 를 쓴다. ``is_available()`` 만 믿지 않고
    실제 연산을 한 번 시켜 본다 — 빌드가 이 GPU 의 계산 능력을 지원하지
    않으면 available 은 True 인데 첫 커널에서 터진다 (s00_check_env 참고).
    """
    if spec == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).sum().item()
        return torch.device("cuda")
    except RuntimeError:
        if spec == "cuda":
            raise
        return torch.device("cpu")


def prepare(
    ds: PowerFlowDataset,
    spec: SurrogateSpec,
    split: dict[str, np.ndarray] | None = None,
    case: str | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
    jac_alpha: float = 0.0,
) -> tuple[Bundle, PowerFlowMLP]:
    """데이터셋에서 텐서 묶음과 (초기화된) 모델을 만든다.

    ``seed`` 는 **가중치 초기화**를 고정한다. 학습 시드(``TrainConfig.seed``)는
    배치 순서만 정하므로, 둘 다 고정해야 완전히 재현된다.

    ``device`` 로 GPU 를 지정하면 입력·라벨·물리모듈·모델이 전부 그쪽으로
    간다. 데이터셋이 통째로 VRAM 에 올라가는데, case118 · 20,000 표본이
    60 MB 남짓이라 8 GB 로 충분하다. 매 배치 전송이 없어져서 이 규모에서는
    이게 제일 빠르다.
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

    # 손실 가중치. 기본은 1/표준편차 (06 문서 §7.2 — 이게 없으면 정규화가
    # 학습을 죽인다). jac_alpha > 0 이면 여기에 야코비안 민감도 곱수를 얹는다.
    vm_w = 1.0 / np.maximum(vm_tr.std(0), 1e-6)
    va_w = 1.0 / np.maximum(va_tr.std(0), 1e-6)
    if jac_alpha > 0:
        # 기준 운전점은 **학습 분할 라벨의 평균**. 검증·시험을 보지 않는다.
        m_vm, m_va = jacobian_weights(
            sysm, layout, ds.Vm[tr].mean(0), ds.Va[tr].mean(0), alpha=jac_alpha
        )
        vm_w = vm_w * m_vm
        va_w = va_w * m_va

    # 헤드·정규화 통계는 두 모델이 **똑같이** 쓴다. 여기가 갈리면 비교가
    # 무너지므로 한 곳에서 만들어 넘긴다.
    head_kw = dict(
        in_mean=in_mean, in_std=in_std,
        v_set=ds.v_set,
        vm_lo=vm_lo, vm_hi=vm_hi,
        va_mean=va_tr.mean(0), va_std=np.maximum(va_tr.std(0), 1e-6),
        vm_w=vm_w, va_w=va_w,
    )
    if type(spec).__name__ == "GATSpec":
        from nnopf.gnn import PowerFlowGAT

        model = PowerFlowGAT(
            layout, spec, sysm=sysm,
            edge_index=ds.edge_index, edge_attr_base=ds.edge_attr_base,
            edge_line=ds.edge_line, **head_kw,
        )
    else:
        model = PowerFlowMLP(layout, spec, **head_kw)

    # 지름길을 최소제곱 해에서 출발시킨다 (spec.skip_init == "lstsq").
    # 모델을 장치로 옮기기 **전에** 한다 — 풀이는 CPU float64 라 그게 자연스럽다.
    skip_info: dict = {}
    if getattr(spec, "skip_init", "zero") == "lstsq" and getattr(spec, "residual", False):
        skip_info = init_skip_lstsq(model, X, ds.Vm, ds.Va, tr, split.get("val"))
    if getattr(spec, "skip_freeze", False) and getattr(model, "skip", None) is not None:
        for prm in model.skip.parameters():
            prm.requires_grad_(False)

    dev = torch.device(device)
    t = lambda a, d=torch.float32: torch.as_tensor(
        np.asarray(a), dtype=d).to(dev)
    # 물리 손실 무차원화 기준: 학습 분할의 모선별 지정주입 RMS
    ph32 = ACPhysics(sysm, torch.float32).to(device=dev)
    ph32.set_scale(
        t(np.sqrt((p_spec[tr] ** 2).mean(0))), t(np.sqrt((q_spec[tr] ** 2).mean(0)))
    )
    bundle = Bundle(
        sys=sysm, layout=layout, split=split,
        X=t(X), Vm=t(ds.Vm), Va=t(ds.Va),
        p_spec=t(p_spec), q_spec=t(q_spec),
        outage=t(ds.outage, torch.long),
        physics=ph32,
        # float64 는 CPU 에 둔다 — 소비자용 GPU 는 배정밀도가 1/64 속도라
        # 여기서 재면 오히려 느려진다.
        physics64=ACPhysics(sysm, torch.float64),
        device=dev,
        skip_init=skip_info,
    )
    return bundle, model.to(dev)


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
    ii = torch.as_tensor(idx, dtype=torch.long, device=b.device)
    ph = b.physics64          # 항상 CPU (배정밀도는 소비자용 GPU 에서 느리다)

    dvm, dva, dP, dQ, vms = [], [], [], [], []
    for s in range(0, len(ii), chunk):
        j = ii[s : s + chunk]
        Vm, Va = model(b.X[j])
        # 예측을 CPU float64 로 내린 뒤에 잰다. 전압 오차가 Ybus 를 거치며
        # max|Y| 배로 증폭되므로 잔차는 반드시 배정밀도로 봐야 한다.
        Vm64, Va64 = Vm.double().cpu(), Va.double().cpu()
        vms.append(Vm64)
        dvm.append((Vm64 - b.Vm[j].double().cpu()).abs())
        dva.append((Va64 - b.Va[j].double().cpu()).abs())
        rp, rq = ph.residual(
            Vm64, Va64,
            b.p_spec[j].double().cpu(), b.q_spec[j].double().cpu(),
            b.outage[j].cpu(),
        )
        dP.append(rp.abs())
        dQ.append(rq.abs())

    dvm, dva = torch.cat(dvm), torch.cat(dva)
    dP, dQ = torch.cat(dP), torch.cat(dQ)
    load = b.p_spec[ii].double().cpu().abs().sum(-1).mean().clamp(min=1e-9)

    # 전압 한계 위반 (계통 기준).
    #
    # **허용오차를 반드시 둬야 한다.** case118 의 슬랙(모선 68)은 상자가
    # [1.0349999999, 1.0350000001] 로 폭이 2e-10 인데, 모델은 float32 라
    # 1.035 를 1.03499997 로밖에 못 쓴다. 상자 폭이 float32 해상도
    # (1.035 근처에서 1.2e-7)보다 좁으니 **어떤 모델이든 항상 위반**이었고,
    # 지표가 정확히 100/118 = 0.8475% 에 못박혀 아무것도 구분하지 못했다
    # (06 문서 §7.10). 1e-6 pu 는 운전 관점에서도 위반이 아니고,
    # 우리 전압 오차(7.5e-05)보다 두 자릿수 아래라 진짜 위반은 안 가린다.
    VLIM_TOL = 1e-6
    Vmin = torch.as_tensor(b.sys.Vmin, dtype=torch.float64) - VLIM_TOL
    Vmax = torch.as_tensor(b.sys.Vmax, dtype=torch.float64) + VLIM_TOL
    Vm_all = torch.cat(vms)
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
def _config_fingerprint(cfg: TrainConfig) -> str:
    """이어달리기가 **같은 실험**인지 확인할 지문.

    저장 경로·주기처럼 학습 결과와 무관한 항목은 뺀다.
    """
    d = {k: v for k, v in asdict(cfg).items()
         if k not in ("ckpt_path", "ckpt_every", "resume")}
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


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

    tr = torch.as_tensor(b.split["train"], dtype=torch.long, device=b.device)
    va = torch.as_tensor(b.split["val"], dtype=torch.long, device=b.device)
    # AdamW(분리형 감쇠). 일반 Adam 의 weight_decay 는 L2 를 기울기에 더하는
    # 방식이라 손실이 작을 때 과제 기울기를 눌러 버린다 (models.py 상단 주석).
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=cfg.lr_patience
    )

    # 검증 분할의 평균 부하 — select="phys" 의 분모. 한 번만 잰다.
    val_load = b.p_spec[va].abs().sum(-1).mean().clamp(min=1e-9)

    best = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    hist: list[dict] = []
    t0 = time.time()

    phys_ref = 1.0     # 물리 항을 지도 항과 같은 크기로 맞추는 환산계수
    start_ep = 0
    prior_secs = 0.0
    ck = Path(cfg.ckpt_path) if cfg.ckpt_path else None
    fp = _config_fingerprint(cfg)
    if ck is not None and ck.exists() and cfg.resume:
        st = torch.load(ck, map_location=b.device, weights_only=False)
        if st.get("fingerprint") != fp:
            raise SystemExit(
                f"중간 저장본의 설정이 지금과 다릅니다: {ck}\n"
                f"  저장본: {st.get('fingerprint')}\n"
                f"  지금  : {fp}\n"
                "같은 태그로 다른 설정을 돌리려는 것이면 그 파일을 지우세요."
            )
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best, best_epoch = st["best"], st["best_epoch"]
        best_state = {k: v.to(b.device) for k, v in st["best_state"].items()}
        hist, phys_ref = st["hist"], st["phys_ref"]
        start_ep, prior_secs = st["epoch"] + 1, st["seconds"]
        torch.set_rng_state(st["rng_torch"])
        np.random.set_state(st["rng_numpy"])
        if verbose:
            print(f"  [재개] epoch {start_ep} 부터 (최고 {best_epoch}, "
                  f"이미 {prior_secs/60:.1f}분 돌았음)")

    def save_ckpt(ep: int) -> None:
        if ck is None:
            return
        ck.parent.mkdir(parents=True, exist_ok=True)
        tmp = ck.with_suffix(ck.suffix + ".tmp")
        torch.save({
            "epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "best": best, "best_epoch": best_epoch,
            "best_state": {k: v.cpu() for k, v in best_state.items()},
            "hist": hist, "phys_ref": phys_ref, "fingerprint": fp,
            "seconds": prior_secs + time.time() - t0,
            "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
        }, tmp)
        tmp.replace(ck)     # 저장 중에 죽어도 이전 체크포인트가 남도록

    for ep in range(start_ep, cfg.epochs):
        lam = lambda_at(ep, cfg)
        if lam > 0 and phys_ref == 1.0:
            # λ 를 '지도 항 대비 몇 배' 로 해석되게 만든다. 두 항의 절대 크기가
            # 계통마다 4자리씩 다르기 때문에(case30 2.2e3배, case118 2.1e7배)
            # 이 환산 없이는 같은 숫자 λ 가 전혀 다른 뜻이 된다.
            with torch.no_grad():
                k = tr[: min(1024, len(tr))]
                Vm0, Va0 = model(b.X[k])
                s0 = supervised_loss(model, Vm0, Va0, b.Vm[k], b.Va[k]).item()
                p0 = b.physics.loss(
                    Vm0, Va0, b.p_spec[k], b.q_spec[k], b.outage[k]
                ).item()
            phys_ref = max(p0, 1e-12) / max(s0, 1e-12)
            if verbose:
                print(f"  [λ 환산] 물리/지도 = {phys_ref:.3e} (epoch {ep})")
        model.train()
        perm = tr[torch.randperm(len(tr), device=b.device)]
        tot = n_seen = 0.0

        for s in range(0, len(perm), cfg.batch):
            j = perm[s : s + cfg.batch]
            with autocast_ctx(b.device, cfg.amp):
                Vm, Va = model(b.X[j])
            # 손실은 언제나 float32 에서. 위 주석(TrainConfig.amp) 참조.
            Vm, Va = Vm.float(), Va.float()
            sup = supervised_loss(model, Vm, Va, b.Vm[j], b.Va[j])
            loss = sup
            if lam > 0:
                loss = loss + (lam / phys_ref) * b.physics.loss(
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
        vloss, vphys = validate(model, b, va, cfg, val_load)
        crit = vphys if cfg.select == "phys" else vloss
        sched.step(crit)
        hist.append(
            {"epoch": ep, "train": tot / n_seen, "val": vloss, "val_phys": vphys,
             "lam": lam, "lr": opt.param_groups[0]["lr"]}
        )

        if crit < best * (1 - 1e-5):
            best, best_epoch = crit, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif ep - best_epoch >= cfg.patience:
            if verbose:
                print(f"  조기 종료 (epoch {ep}, 최고 {best_epoch})")
            break

        if cfg.ckpt_every > 0 and ep % cfg.ckpt_every == 0:
            save_ckpt(ep)

        if verbose and (ep % log_every == 0 or ep == cfg.epochs - 1):
            extra = f"  P/부하 {vphys:.2f}%" if cfg.select == "phys" else ""
            print(
                f"  ep {ep:4d}  train {tot/n_seen:.3e}  val {vloss:.3e}{extra}"
                f"  λ {lam:.1e}  lr {opt.param_groups[0]['lr']:.1e}"
            )

    model.load_state_dict(best_state)
    if ck is not None and ck.exists():
        ck.unlink()            # 끝났으니 중간 저장본은 지운다
    return {
        "phys_ref": phys_ref,
        "history": hist,
        "best_epoch": best_epoch,
        "best_val": best,          # cfg.select 가 가리키는 기준의 최고값
        "select": cfg.select,
        "seconds": prior_secs + time.time() - t0,
        "epochs_run": len(hist),
    }


def save_run(path: str | Path, payload: dict) -> None:
    """결과 JSON 저장. 설정과 지표를 항상 같이 남긴다 (00 문서 §9)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def spec_config_dict(spec, cfg: TrainConfig) -> dict:
    """결과 JSON 에 넣을 설정. 어떤 모델이었는지도 같이 남긴다 — 나중에
    비교표를 만들 때 파일 이름만으로는 부족하다."""
    kind = "gat" if type(spec).__name__ == "GATSpec" else "mlp"
    return {"model": kind, "spec": asdict(spec), "train": asdict(cfg)}


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
