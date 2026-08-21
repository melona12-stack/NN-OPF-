r"""비교군(baseline) 대체모델.

06 부록 §4.1 이 못 박은 기준선이 있다 — **선형근사의 오차 수준(전압 0.2%,
전력 1.5%)을 넘지 못하면 신경망을 쓸 이유가 없다.** 그래서 선형 대체모델을
신경망과 **똑같은 입출력 계약·똑같은 지표**로 잴 수 있게 구현해 둔다.

.. note::
   조류방정식은 정상 운전 영역에서 **거의 선형**이다. DC 조류계산과 고정점
   선형화가 수십 년간 실무에서 쓰인 이유가 그것이다. 실제로 case30 에서
   최소제곱 선형모델이 :math:`R^2 = 0.987` 을 낸다. 신경망의 값어치는
   평균 :math:`R^2` 가 아니라 **꼬리(최대 오차)와 물리 잔차**에서 나온다.
   비교표를 만들 때 이 점을 분명히 해야 한다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nnopf.models import IOLayout

__all__ = ["LinearSurrogate", "fit_linear"]


class LinearSurrogate(nn.Module):
    """최소제곱 선형 대체모델.

    ``PowerFlowMLP`` 와 인터페이스가 같아서 ``train.evaluate`` 를 그대로 쓴다.
    아는 값(슬랙 위상, PV·슬랙 전압)은 신경망과 **똑같이** 정확히 채운다 —
    그래야 비교가 공정하다.
    """

    def __init__(self, layout: IOLayout, W: np.ndarray, v_set: np.ndarray) -> None:
        super().__init__()
        self.nb = layout.nb
        self.n_vm = len(layout.pq)
        self.register_buffer("W", torch.as_tensor(W, dtype=torch.float32))
        self.register_buffer("v_set", torch.as_tensor(v_set, dtype=torch.float32))
        self.register_buffer("pq_idx", torch.as_tensor(layout.pq, dtype=torch.long))
        self.register_buffer("va_idx", torch.as_tensor(layout.nonslack, dtype=torch.long))
        self.register_buffer("va_ref", torch.as_tensor(layout.va_ref, dtype=torch.float32))

    @property
    def n_params(self) -> int:
        return int(self.W.numel())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 절편 열은 **입력과 같은 장치**에 만든다. device 를 빼먹으면 입력이
        # GPU 일 때 여기서 터진다 — 신경망 쪽만 옮기고 비교군을 빠뜨리기 쉽다.
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
        h = torch.cat([x, ones], 1) @ self.W
        vm, va = h[:, : self.n_vm], h[:, self.n_vm :]
        b = x.shape[0]
        Vm = self.v_set.expand(b, self.nb).index_copy(1, self.pq_idx, vm)
        Va = self.va_ref.expand(b, self.nb).index_copy(1, self.va_idx, va)
        return Vm, Va


def fit_linear(ds, layout: IOLayout, train_idx: np.ndarray) -> LinearSurrogate:
    """학습 분할에서 닫힌 형태로 푼다. 시드도 epoch 도 없다."""
    X = layout.inputs(ds)[train_idx].astype(np.float64)
    Y = np.concatenate(
        [ds.Vm[train_idx][:, layout.pq], ds.Va[train_idx][:, layout.nonslack]], axis=1
    )
    A = np.c_[X, np.ones(len(X))]
    W, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return LinearSurrogate(layout, W, ds.v_set)
