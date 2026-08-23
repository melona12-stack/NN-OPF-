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

from nnopf.case import SLACK
from nnopf.models import IOLayout

__all__ = ["LinearSurrogate", "fit_linear", "DCPowerFlow", "JacobianLinear"]


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


# --------------------------------------------------------------------------
# 물리 기반 선형 비교군 — DC 조류계산과 야코비안 1차 선형화
# --------------------------------------------------------------------------
#
# 위 ``LinearSurrogate`` 는 **데이터에서 배운** 선형모델이다. 아래 둘은
# 반대로 **데이터를 한 건도 안 보고** 물리에서 바로 유도한 선형모델이다.
# 07 부록 §4.1 이 비교군으로 지목한 것이 이쪽이고, 전력계통 문헌에서
# 수십 년간 표준으로 쓰인 근사이므로 벤치마크에 반드시 있어야 한다.
#
# 셋의 성격이 다르다는 점이 표를 읽을 때 중요하다.
#
#   최소제곱 선형   학습 분할을 보고 계수를 맞춤     → 데이터가 필요
#   DC 조류계산     |V|=1, 손실 0, Q 무시            → 데이터 불필요
#   야코비안 선형화 기저해 근처 1차 테일러 전개      → 데이터 불필요


def _susceptance_matrix(sysm, alive: np.ndarray) -> "sp.csr_matrix":
    r"""DC 조류계산용 서셉턴스 행렬 :math:`B'`.

    저항과 충전 서셉턴스를 버리고 선로를 :math:`1/x` 하나로만 본다.
    ``alive`` 는 선로별 0/1 로, N-1 상정사고를 그대로 반영한다.
    """
    import scipy.sparse as sp

    b = alive / (sysm.br_x * sysm.br_tap)     # 탭비까지는 반영한다
    nb = sysm.nb
    f, t = sysm.f_bus, sysm.t_bus
    rows = np.concatenate([f, t, f, t])
    cols = np.concatenate([f, t, t, f])
    vals = np.concatenate([b, b, -b, -b])
    return sp.csr_matrix((vals, (rows, cols)), shape=(nb, nb))


class DCPowerFlow(nn.Module):
    r"""DC 조류계산 — 전력계통에서 가장 오래되고 가장 단순한 선형근사.

    세 가지를 버린다.

    1. **전압 크기** — 모든 모선을 :math:`|V| = 1.0` 으로 본다
    2. **손실** — 선로 저항 :math:`r` 을 0 으로 본다
    3. **무효전력** — Q 방정식을 아예 안 푼다

    남는 것은 선형 방정식 하나다.

    .. math::
        P = B'\,\theta,\qquad B'_{ik} = -\frac{1}{x_{ik}}

    위상만 풀고 전압 크기는 지정값(슬랙·PV) 또는 1.0(PQ)으로 둔다.
    그래서 **Vm 오차는 구조적으로 클 수밖에 없다** — 그게 이 근사의 정체다.
    비교표에서 "가장 단순한 기준선" 자리를 채운다.

    상정사고마다 :math:`B'` 가 달라지므로 **고장 종류별로 한 번씩만** 분해해
    두고 재사용한다 (case118 이면 178가지).
    """

    def __init__(self, sysm, layout: IOLayout) -> None:
        super().__init__()
        self.sysm, self.layout = sysm, layout
        self.nb, self.nl = layout.nb, layout.nl
        self.register_buffer("pq_idx", torch.as_tensor(layout.pq, dtype=torch.long))
        self.register_buffer("va_idx",
                             torch.as_tensor(layout.nonslack, dtype=torch.long))
        self.register_buffer("va_ref",
                             torch.as_tensor(layout.va_ref, dtype=torch.float32))
        # PQ 모선 전압은 1.0, 나머지는 지정값 — 신경망과 같은 자리를 채운다
        vm = np.asarray(sysm.Vm0, float).copy()
        vm[layout.pq] = 1.0
        self.register_buffer("vm_flat", torch.as_tensor(vm, dtype=torch.float32))
        self.slack_i = int(np.flatnonzero(sysm.bus_type == SLACK)[0])
        self._cache: dict[int, object] = {}

    @property
    def n_params(self) -> int:
        return 0                      # 학습 파라미터가 없다

    def _solver(self, out: int):
        """상정사고 ``out`` 에 대한 :math:`B'` 의 LU 분해 (없으면 만든다)."""
        import scipy.sparse.linalg as spla

        if out not in self._cache:
            alive = self.sysm.br_status.astype(float).copy()
            if out >= 0:
                alive[out] = 0.0
            B = _susceptance_matrix(self.sysm, alive)
            ns = self.layout.nonslack
            self._cache[out] = (spla.factorized(B[ns][:, ns].tocsc()), ns)
        return self._cache[out]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nb, nl = self.nb, self.nl
        xn = x.detach().cpu().numpy().astype(np.float64)
        # 입력 레이아웃: [Pd, Qd, p_ren, p_gen, status]  (IOLayout.inputs)
        Pd = xn[:, :nb]
        p_ren = xn[:, 2 * nb : 3 * nb]
        p_gen = xn[:, 3 * nb : 4 * nb]
        status = xn[:, 4 * nb : 4 * nb + nl]
        Psp = p_gen + p_ren - Pd                   # 모선 순주입 [pu]

        # 어느 선로가 끊겼는지 — status 열에서 0 을 찾는다 (정상이면 -1)
        outage = np.full(len(xn), -1, dtype=np.int64)
        off = np.argwhere(status < 0.5)
        outage[off[:, 0]] = off[:, 1]

        # DC 는 위상차만 정한다. 슬랙을 0 으로 놓고 푼 뒤 기준위상을 더한다
        # — 그 값이 0 이라고 가정하면 case118 에서 통째로 틀린다 (06 §7.4).
        ref = float(self.layout.va_ref[self.slack_i])
        Va = np.full((len(xn), nb), ref)
        for out in np.unique(outage):
            solve, ns = self._solver(int(out))
            rows = np.flatnonzero(outage == out)
            for i in rows:
                Va[i, ns] = solve(Psp[i, ns]) + ref

        dev, dt = x.device, x.dtype
        Vm = self.vm_flat.to(dev).expand(len(xn), nb).clone()
        return Vm, torch.as_tensor(Va, dtype=dt, device=dev)


class JacobianLinear(nn.Module):
    r"""기저해 근처 1차 테일러 전개 — "고정점 선형화" 계열.

    조류방정식은 정상 운전 영역에서 거의 선형이다(§2.1). 그렇다면 **한 점에서
    한 번만 미분해 두고** 그 접평면을 계속 쓰면 어떨까 — 그게 이 근사다.

    .. math::
        \begin{bmatrix}\Delta\theta \\ \Delta|V|\end{bmatrix}
        = J(V_0)^{-1}
        \begin{bmatrix}\Delta P \\ \Delta Q\end{bmatrix}

    뉴턴-랩슨이 **매번 새로 만드는** 야코비안을 기저 케이스에서 **한 번만**
    만들어 재사용하는 것이다. 반복이 없으니 한 번의 선형 풀이로 끝난다.

    Bolognani & Zampieri(2015)의 고정점 선형화는 배전계통(방사형·불평형)을
    겨냥한 형태라 식이 조금 다르다. 여기 구현한 것은 **같은 아이디어를 송전
    계통 표준형(극좌표 야코비안)으로 옮긴 것**이다. 07 부록 §5 의 비교군
    자리를 채우되, 원논문 그대로는 아니라는 점을 밝혀 둔다.

    DC 와 달리 **전압 크기도 푼다.** 그래서 Vm 오차가 DC 보다 훨씬 작아야
    정상이고, 그 차이가 "무효전력을 버리는 대가"를 숫자로 보여 준다.
    """

    def __init__(self, sysm, layout: IOLayout) -> None:
        super().__init__()
        self.sysm, self.layout = sysm, layout
        self.nb, self.nl = layout.nb, layout.nl
        self.slack_i = int(np.flatnonzero(sysm.bus_type == SLACK)[0])
        self._cache: dict[int, object] = {}

    @property
    def n_params(self) -> int:
        return 0

    def _base(self, out: int):
        """상정사고 ``out`` 의 기저해와 야코비안 분해 (없으면 만든다).

        기저 케이스(데이터셋의 기준 부하)에서 **진짜 뉴턴-랩슨을 한 번 풀고**
        그 점의 야코비안을 분해해 둔다. 상정사고마다 토폴로지가 다르므로
        고장 종류별로 하나씩 필요하다 (case118 이면 178가지).
        """
        import dataclasses

        import scipy.sparse.linalg as spla

        from nnopf.powerflow import _build_jacobian, sbus_from_V, solve_power_flow
        from nnopf.ybus import make_ybus

        if out not in self._cache:
            sysm = self.sysm
            if out >= 0:
                st = sysm.br_status.copy()
                st[out] = 0
                sysm = dataclasses.replace(sysm, br_status=st)
            pf = solve_power_flow(sysm, tol=1e-10)
            Ybus = make_ybus(sysm)
            pvpq = self.layout.nonslack
            pq = self.layout.pq
            J = _build_jacobian(Ybus, pf.V, pvpq, pq)
            S0 = sbus_from_V(Ybus, pf.V)
            self._cache[out] = (
                spla.factorized(J.tocsc()), pf.Vm.copy(), pf.Va.copy(),
                np.real(S0).copy(), np.imag(S0).copy(), pvpq, pq,
            )
        return self._cache[out]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nb, nl = self.nb, self.nl
        xn = x.detach().cpu().numpy().astype(np.float64)
        Pd, Qd = xn[:, :nb], xn[:, nb : 2 * nb]
        p_ren, p_gen = xn[:, 2 * nb : 3 * nb], xn[:, 3 * nb : 4 * nb]
        status = xn[:, 4 * nb : 4 * nb + nl]
        Psp = p_gen + p_ren - Pd
        Qsp = -Qd                                  # 부하만 무효전력 지정값

        outage = np.full(len(xn), -1, dtype=np.int64)
        off = np.argwhere(status < 0.5)
        outage[off[:, 0]] = off[:, 1]

        Vm = np.empty((len(xn), nb))
        Va = np.empty((len(xn), nb))
        for out in np.unique(outage):
            solve, vm0, va0, p0, q0, pvpq, pq = self._base(int(out))
            for i in np.flatnonzero(outage == out):
                # F 의 순서는 뉴턴-랩슨과 같아야 한다: [ΔP@비슬랙, ΔQ@PQ]
                F = np.concatenate([Psp[i, pvpq] - p0[pvpq], Qsp[i, pq] - q0[pq]])
                dx = solve(F)
                Vm[i], Va[i] = vm0.copy(), va0.copy()
                Va[i, pvpq] += dx[: len(pvpq)]
                Vm[i, pq] += dx[len(pvpq) :]

        dev, dt = x.device, x.dtype
        return (torch.as_tensor(Vm, dtype=dt, device=dev),
                torch.as_tensor(Va, dtype=dt, device=dev))
