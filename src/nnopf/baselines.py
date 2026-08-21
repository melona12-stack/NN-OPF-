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

    def __init__(self, layout: IOLayout, W: np.ndarray, v_set: np.ndarray,
                 keep: np.ndarray | None = None) -> None:
        super().__init__()
        self.nb = layout.nb
        # 학습 분할에서 상수였던 열은 아예 빼고 푼다 (fit_linear 참고).
        # keep=None 이면 전부 쓴다 — 예전 체크포인트 호환.
        self.register_buffer(
            "keep",
            torch.as_tensor(
                np.arange(layout.in_dim) if keep is None else keep, dtype=torch.long
            ),
        )
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
        h = torch.cat([x[:, self.keep], ones], 1) @ self.W
        vm, va = h[:, : self.n_vm], h[:, self.n_vm :]
        b = x.shape[0]
        Vm = self.v_set.expand(b, self.nb).index_copy(1, self.pq_idx, vm)
        Va = self.va_ref.expand(b, self.nb).index_copy(1, self.va_idx, va)
        return Vm, Va


def fit_linear(
    ds, layout: IOLayout, train_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
) -> LinearSurrogate:
    r"""학습 분할에서 닫힌 형태로 푼다. 시드도 epoch 도 없다.

    .. warning::
       **상수열을 빼지 않으면 이 함수는 컴퓨터마다 다른 답을 낸다.**

       입력에는 정보가 0 인 열이 많다 — 부하가 없는 모선의 ``Pd``, 발전기가
       없는 모선의 ``p_gen``, 끊으면 계통이 갈라져 상정사고에서 제외된 선로의
       ``status``. case118 에서 658 열 중 **239 열**이 그렇다.

       그 열들을 그대로 두면 설계행렬 조건수가 :math:`2.7\times10^{58}` 이 되고,
       ``lstsq`` 의 특이값 절단선이 어디에 걸리느냐로 답이 통째로 달라진다.
       실측한 값이다 (case118, 무작위 분할):

       ===========  ==========  ===========
       rcond        Vm MAE      P/부하
       ===========  ==========  ===========
       1e-10 이하   4.79e-05    30.80 %
       1e-08 이상   3.53e-05     0.90 %
       ===========  ==========  ===========

       numpy 기본값(``rcond=None``)은 :math:`6.2\times10^{-9}` 로 **그 전환
       구간 한가운데**다. LAPACK 구현이 조금만 달라도 답이 튄다. 실제로 같은
       데이터·같은 코드로 두 컴퓨터에서 30.80 % 와 22.99 % 가 나왔다.

       상수열을 빼도 case118 은 조건수가 :math:`8.3\times10^{8}` 로 남는다.
       거기서 성분을 **하나 더** 버리면(420 → 419) 검증 손실은 거의 그대로인데
       물리 잔차가 33.76 % → 0.90 % 로 떨어진다. 전압에는 기여하지 않으면서
       잔차만 키우는 방향이 하나 있다는 뜻이다.

       그래서 절단선을 ``val_idx`` 로 **고른다.** 신경망이 조기 종료에 쓰는
       것과 같은 표준화 지도손실을 기준으로 삼아, 두 모델의 선택 기준을
       맞춘다. "닫힌 해라 재현된다" 는 말은 **행렬이 제대로 세워졌을 때만**
       참이다.
    """
    Xall = layout.inputs(ds).astype(np.float64)
    X = Xall[train_idx]
    keep = np.flatnonzero(X.std(0) > 0)      # 상수열 제거 — 정보가 없다
    tgt = lambda idx: np.concatenate(
        [ds.Vm[idx][:, layout.pq], ds.Va[idx][:, layout.nonslack]], axis=1
    )
    A, Y = np.c_[X[:, keep], np.ones(len(X))], tgt(train_idx)

    if val_idx is None:                       # 고를 근거가 없으면 기본값
        W, *_ = np.linalg.lstsq(A, Y, rcond=None)
        return LinearSurrogate(layout, W, ds.v_set, keep=keep)

    # 절단선을 **검증 분할로** 고른다. 신경망이 조기 종료에 쓰는 것과 같은
    # 표준화 지도손실을 기준으로 삼아야 두 모델의 선택 기준이 같아진다.
    Av = np.c_[Xall[val_idx][:, keep], np.ones(len(val_idx))]
    Yv, w = tgt(val_idx), 1.0 / np.maximum(Y.std(0), 1e-6)
    best, best_W = np.inf, None
    for rc in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
        W, *_ = np.linalg.lstsq(A, Y, rcond=rc)
        loss = float((((Av @ W - Yv) * w) ** 2).mean())
        if loss < best:
            best, best_W = loss, W
    return LinearSurrogate(layout, best_W, ds.v_set, keep=keep)
