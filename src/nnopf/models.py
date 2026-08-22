r"""조류계산 대체모델 — MLP 기준선 (4단계).

설계는 03 문서 §5 의 입출력 계약을 그대로 따른다.

**무엇을 예측하지 않는가가 무엇을 예측하는가보다 중요하다.**

=========  =======================  ==============================================
값          예측?                    이유
=========  =======================  ==============================================
슬랙 θ      ❌ 항상 0                 위상 기준점 (01 문서 §2.3)
PV·슬랙 |V| ❌ 설정값 그대로           발전기가 유지하는 값. 데이터에서 확인: 오차 7e-16
PQ |V|      ✅                        조류방정식이 결정
비슬랙 θ    ✅                        조류방정식이 결정
P, Q        ❌ 닫힌 식으로 계산        :math:`S = V\overline{YV}` (01 문서 §6.3)
=========  =======================  ==============================================

case30 이면 예측 대상이 60개가 아니라 **53개**(PQ 24 + 비슬랙 29)로 줄고,
줄어든 7개는 근사가 아니라 **정확한 값**이 들어간다.

왜 선형 지름길(``residual=True``)이 기본인가
------------------------------------------
조류방정식은 정상 운전 영역에서 **거의 선형**이다 — DC 조류계산과 고정점
선형화가 실무에서 오래 쓰인 이유다. 실제로 case30 에서 최소제곱 선형모델이
:math:`R^2 = 0.987` 을 낸다.

그래서 MLP 에게 사상 전체를 처음부터 배우게 하면, 비선형 보정을 배우기 전에
**선형 부분을 재현하는 데만 용량과 epoch 을 다 쓴다**. 실측으로도 순수 MLP 가
선형 최소제곱보다 못했다 (Vm MAE 1.7e-4 vs 6.0e-5).

입력에서 출력으로 가는 선형 층을 하나 더해 두면 신경망은 **잔차만** 배우면
된다. 이게 로드맵 M4 의 '잔차연결 MLP' 이고, 기본값으로 둔다.

전압 출력 헤드
--------------
``vm_head="scaled"`` 는 P1 §5.2.2 의 스케일링 인자다.

.. math::

    |V| = \alpha\,(V^{hi} - V^{lo}) + V^{lo}, \qquad \alpha = \sigma(\cdot) \in [0,1]

무슨 값이 나오든 :math:`[V^{lo}, V^{hi}]` 를 벗어날 수 없다.

.. warning::
   박스를 **계통 전압한계** :math:`[V^{min}, V^{max}]` 로 잡으면 안 된다.
   조류계산은 전압한계를 강제하지 않으므로 라벨이 그 밖으로 나갈 수 있다
   (case30 기준 PQ 표본의 0.63% 가 0.95 pu 미만). 그러면 모델이 도달할 수 없는
   정답이 생겨 계통적 오차가 남는다. 그래서 박스는 **학습 데이터 범위에
   여유를 더해** 잡는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from nnopf.case import PQ, SLACK, PowerSystem

__all__ = ["SurrogateSpec", "PowerFlowMLP", "IOLayout"]

# 손실 표준화가 왜 필요한가 — 실측으로 확인한 것
#
#   라벨 분산이 Vm 8.1e-05, Va 8.1e-04 라 원단위 MSE 손실은 1e-4 규모이고
#   기울기도 그 규모다. PyTorch ``Adam(weight_decay=)`` 는 L2 를 **기울기에
#   더하는** 방식이라, wd=1e-5 x |w| 가 과제 기울기와 맞먹어 버린다.
#   그러면 Adam 이 사실상 가중치 축소만 하고 학습이 MSE 1e-4 에서 멈춘다
#   (선형회귀보다도 못한 지점).
#
#     lr      wd      train MSE
#     5e-4    0       9.3e-06
#     5e-4    1e-5    1.0e-04   <- 갇힘
#     2e-3    0       2.2e-06
#     2e-3    1e-5    1.0e-04   <- 갇힘  (lr 과 무관)
#
#   그래서 손실을 **표준화 공간에서** 계산한다. 손실이 O(1) 이 되어 정규화가
#   의도대로 동작하고, 덤으로 Vm 과 Va 가 대등하게 반영된다 (원단위로는 분산이
#   10배 차이라 Va 가 손실을 지배했다).

_ACT = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
}


@dataclass(frozen=True)
class SurrogateSpec:
    """대체모델 하이퍼파라미터."""

    hidden: int = 256
    layers: int = 3
    activation: str = "silu"
    dropout: float = 0.0
    vm_head: str = "scaled"     # "scaled" (제약 보장) | "raw" (표준화)
    vm_margin: float = 0.05     # scaled 박스에 더할 여유 (데이터 범위 대비 비율)
    residual: bool = True       # 입력->출력 선형 지름길 (아래 설명)

    # 지름길의 **출발점**. 06 문서 §6.2 에서 잰다.
    #   "zero"  — 0 에서 시작해 본체와 함께 배운다 (지금까지의 기본값).
    #   "lstsq" — 학습 분할의 최소제곱 해에서 시작한다. 그러면 학습 첫 순간의
    #             모델이 곧 선형 기준선이고, 신경망은 진짜로 **보정만** 배운다.
    skip_init: str = "zero"
    skip_freeze: bool = False   # True 면 지름길을 얼려 둔다 (본체만 학습)

    def __post_init__(self) -> None:
        if self.activation not in _ACT:
            raise ValueError(f"activation 은 {sorted(_ACT)} 중 하나여야 합니다")
        if self.vm_head not in ("scaled", "raw"):
            raise ValueError('vm_head 는 "scaled" 또는 "raw" 여야 합니다')
        if self.skip_init not in ("zero", "lstsq"):
            raise ValueError('skip_init 은 "zero" 또는 "lstsq" 여야 합니다')


class IOLayout:
    """어떤 모선의 무엇을 예측하는지 — 모델과 학습 코드가 공유하는 계약.

    입력 벡터는 ``[Pd | Qd | p_ren | p_gen | line_status]`` 를 이어붙인 것이다.
    노드 특징 8개 중 정적인 4개(``V_set``, 모선종류 원-핫)는 **넣지 않는다** —
    MLP 에서는 모선 위치가 곧 인덱스라 상수 입력이 되어 정보가 0 이다.
    (그래프 신경망에서는 노드마다 필요하므로 M4 에서 다시 들어온다.)
    """

    def __init__(self, sys: PowerSystem) -> None:
        self.nb = sys.nb
        self.nl = len(sys.f_bus)
        self.pq = np.flatnonzero(sys.bus_type == PQ)
        self.nonslack = np.flatnonzero(sys.bus_type != SLACK)
        # 슬랙 기준위상. **0 이라고 가정하면 안 된다** — pandapower ``case118``
        # 은 슬랙(모선 68)의 기준위상이 30° 다. 자세한 사연은 06 문서 §7.4.
        self.va_ref = np.zeros(self.nb)
        self.va_ref[sys.bus_type == SLACK] = sys.Va0[sys.bus_type == SLACK]
        self.in_dim = 4 * self.nb + self.nl
        self.out_dim = len(self.pq) + len(self.nonslack)

    def inputs(self, ds, idx=None) -> np.ndarray:
        """데이터셋에서 입력 행렬 ``(N, in_dim)`` 을 만든다."""
        sl = slice(None) if idx is None else idx
        status = np.ones((len(np.atleast_1d(ds.outage[sl])), self.nl), np.float32)
        out = np.atleast_1d(ds.outage[sl])
        hit = np.flatnonzero(out >= 0)
        status[hit, out[hit]] = 0.0
        return np.concatenate(
            [ds.Pd[sl], ds.Qd[sl], ds.p_ren[sl], ds.p_gen[sl], status], axis=1
        ).astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"IOLayout(nb={self.nb}, nl={self.nl}, "
            f"in={self.in_dim}, out={self.out_dim} "
            f"[Vm@PQ {len(self.pq)} + Va@비슬랙 {len(self.nonslack)}])"
        )


class PowerFlowMLP(nn.Module):
    """부하·발전·선로상태 → 모선 전압 ``(Vm, Va)``.

    정규화 통계와 출력 박스는 **학습 분할에서만** 계산해 버퍼로 들고 있는다.
    체크포인트 하나로 추론이 재현되고, 시험 분할 정보가 새어 들어가지 않는다.
    """

    def __init__(
        self,
        layout: IOLayout,
        spec: SurrogateSpec,
        in_mean: np.ndarray,
        in_std: np.ndarray,
        v_set: np.ndarray,
        vm_lo: np.ndarray,
        vm_hi: np.ndarray,
        va_mean: np.ndarray,
        va_std: np.ndarray,
        vm_w: np.ndarray,
        va_w: np.ndarray,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.nb = layout.nb

        def buf(name: str, x: np.ndarray) -> None:
            self.register_buffer(name, torch.as_tensor(x, dtype=torch.float32))

        buf("in_mean", in_mean)
        buf("in_std", in_std)
        buf("v_set", v_set)
        buf("va_ref", layout.va_ref)
        buf("vm_lo", vm_lo)
        buf("vm_hi", vm_hi)
        buf("va_mean", va_mean)
        buf("va_std", va_std)
        # 손실 표준화 가중치 = 1/표준편차. 아래 이유로 필수다.
        buf("vm_w", vm_w)
        buf("va_w", va_w)
        self.register_buffer("pq_idx", torch.as_tensor(layout.pq, dtype=torch.long))
        self.register_buffer(
            "va_idx", torch.as_tensor(layout.nonslack, dtype=torch.long)
        )

        act = _ACT[spec.activation]
        dims = [layout.in_dim] + [spec.hidden] * spec.layers
        body: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            body += [nn.Linear(a, b), act()]
            if spec.dropout > 0:
                body.append(nn.Dropout(spec.dropout))
        body.append(nn.Linear(dims[-1], layout.out_dim))
        self.net = nn.Sequential(*body)
        # 선형 지름길: 신경망은 '전체 사상'이 아니라 '선형에 대한 보정'만 배운다.
        # 0 으로 초기화해 학습 시작점이 순수 선형모델이 되게 한다.
        self.skip = nn.Linear(layout.in_dim, layout.out_dim) if spec.residual else None
        if self.skip is not None:
            nn.init.zeros_(self.skip.bias)
        self.n_vm = len(layout.pq)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, in_dim)`` -> ``(Vm, Va)`` 각각 ``(B, nb)``."""
        z = (x - self.in_mean) / self.in_std
        h = self.net(z)
        if self.skip is not None:
            h = h + self.skip(z)
        vm_raw, va_raw = h[:, : self.n_vm], h[:, self.n_vm :]

        if self.spec.vm_head == "scaled":
            vm = self.vm_lo + torch.sigmoid(vm_raw) * (self.vm_hi - self.vm_lo)
        else:
            vm = self.vm_lo + vm_raw * self.vm_hi   # raw 모드에서는 (평균, 표준편차)
        va = self.va_mean + va_raw * self.va_std

        b = x.shape[0]
        Vm = self.v_set.expand(b, self.nb).index_copy(1, self.pq_idx, vm)
        Va = self.va_ref.expand(b, self.nb).index_copy(1, self.va_idx, va)
        return Vm, Va
