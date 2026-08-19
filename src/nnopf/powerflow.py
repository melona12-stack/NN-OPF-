r"""뉴턴-랩슨 조류계산 (극좌표 형식).

문제 정의
---------
모선 :math:`i` 의 주입 복소전력은

.. math::
    S_i = V_i \overline{\left(\sum_k Y_{ik} V_k\right)},
    \qquad V_i = |V_i| e^{j\theta_i}

미지수/기지수는 모선 종류로 나뉜다.

======  ================  ================
종류    주어진 값          구할 값
======  ================  ================
슬랙    :math:`|V|, \theta`  :math:`P, Q`
PV      :math:`P, |V|`       :math:`Q, \theta`
PQ      :math:`P, Q`         :math:`|V|, \theta`
======  ================  ================

미지수 :math:`x = [\theta_{PV,PQ};\ |V|_{PQ}]` 에 대해
불일치(mismatch) :math:`f(x) = 0` 을 뉴턴법으로 푼다.

.. math::
    f(x) = \begin{bmatrix}
        P^{spec}_{PV,PQ} - P(x) \\ Q^{spec}_{PQ} - Q(x)
    \end{bmatrix},
    \qquad
    J = \begin{bmatrix} J_{11} & J_{12} \\ J_{21} & J_{22} \end{bmatrix}

여기서 :math:`J` 는 :func:`dSbus_dV` 로 얻은 해석적 야코비안이다.
수치 미분을 쓰지 않으므로 대형 계통에서도 반복 5~6회 안에 수렴한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from nnopf.ybus import branch_flows, make_ybus

__all__ = ["solve_power_flow", "PowerFlowResult", "dSbus_dV", "sbus_from_V"]


@dataclass
class PowerFlowResult:
    """조류계산 결과."""

    converged: bool
    iterations: int
    Vm: np.ndarray            # 모선 전압 크기 [pu]
    Va: np.ndarray            # 모선 전압 위상 [rad]
    Pg: np.ndarray            # 발전기 유효출력 [pu] (슬랙/PV 는 계산값)
    Qg: np.ndarray            # 발전기 무효출력 [pu]
    max_mismatch: float       # 최종 최대 전력 불일치 [pu]
    history: list[float] = field(default_factory=list)

    @property
    def V(self) -> np.ndarray:
        """복소 전압 벡터."""
        return self.Vm * np.exp(1j * self.Va)

    def va_deg(self) -> np.ndarray:
        """위상각 [deg]."""
        return np.rad2deg(self.Va)


def sbus_from_V(Ybus, V: np.ndarray) -> np.ndarray:
    r"""주어진 전압에서의 모선 주입 복소전력 :math:`S = V \overline{Y V}` [pu]."""
    return V * np.conj(Ybus @ V)


def dSbus_dV(Ybus, V: np.ndarray) -> tuple[sp.spmatrix, sp.spmatrix]:
    r"""주입전력의 전압에 대한 편미분 ``(dS_dVa, dS_dVm)``.

    복소수 형태로 한 번에 계산한 뒤, 실수부/허수부를 취하면
    :math:`\partial P/\partial\theta`, :math:`\partial Q/\partial |V|` 등을
    모두 얻을 수 있다. (MATPOWER ``dSbus_dV`` 와 동일한 유도)

    .. math::
        \frac{\partial S}{\partial |V|} =
            \mathrm{diag}(V)\,\overline{Y\,\mathrm{diag}(V/|V|)}
            + \overline{\mathrm{diag}(YV)}\,\mathrm{diag}(V/|V|)

        \frac{\partial S}{\partial \theta} =
            j\,\mathrm{diag}(V)\,\overline{\mathrm{diag}(YV) - Y\,\mathrm{diag}(V)}
    """
    Ibus = Ybus @ V
    diagV = sp.diags(V)
    diagIbus = sp.diags(Ibus)
    diagVnorm = sp.diags(V / np.abs(V))

    dS_dVm = diagV @ np.conj(Ybus @ diagVnorm) + np.conj(diagIbus) @ diagVnorm
    dS_dVa = 1j * diagV @ np.conj(diagIbus - Ybus @ diagV)
    return dS_dVa, dS_dVm


def _build_jacobian(Ybus, V, pvpq, pq) -> sp.csr_matrix:
    """뉴턴법 야코비안 ``J`` 를 조립한다 (크기 ``(npvpq+npq, npvpq+npq)``)."""
    dS_dVa, dS_dVm = dSbus_dV(Ybus, V)

    J11 = np.real(dS_dVa[np.ix_(pvpq, pvpq)])   # dP/dTheta
    J12 = np.real(dS_dVm[np.ix_(pvpq, pq)])     # dP/dVm
    J21 = np.imag(dS_dVa[np.ix_(pq, pvpq)])     # dQ/dTheta
    J22 = np.imag(dS_dVm[np.ix_(pq, pq)])       # dQ/dVm

    return sp.bmat([[J11, J12], [J21, J22]], format="csr")


def solve_power_flow(
    sys,
    Pg: np.ndarray | None = None,
    Vm_set: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 30,
    Vm0: np.ndarray | None = None,
    Va0: np.ndarray | None = None,
    enforce_q_limits: bool = False,
    Ybus=None,
    verbose: bool = False,
) -> PowerFlowResult:
    """뉴턴-랩슨 조류계산.

    Parameters
    ----------
    sys
        :class:`nnopf.case.PowerSystem`
    Pg
        발전기 유효출력 지령 ``(ng,)`` [pu]. ``None`` 이면 케이스의 ``Pg0``.
        슬랙 모선 발전기 값은 무시된다(슬랙이 수급 균형을 맞추므로).
    Vm_set
        발전기 단자 전압 지령 ``(ng,)`` [pu]. ``None`` 이면 케이스의 ``Vg``.
    tol
        수렴 판정 기준 (최대 전력 불일치, pu).
    enforce_q_limits
        True 면 PV 모선의 무효출력 한계 위반 시 PQ 로 전환해 다시 푼다.
        (OPF 검증 목적에서는 보통 False 로 두고 비교한다.)
    Ybus
        미리 만들어 둔 어드미턴스 행렬. 부하만 바뀌고 **토폴로지가 같은** 시나리오를
        수천 건 풀 때 재사용하면 건당 약 10% 가 줄어든다 (데이터 생성기용).
        토폴로지가 달라지면(N-1 등) 반드시 새로 넘겨야 한다.

    Returns
    -------
    PowerFlowResult
    """
    Pg = sys.Pg0.copy() if Pg is None else np.asarray(Pg, dtype=float).copy()
    Vm_set = sys.Vg.copy() if Vm_set is None else np.asarray(Vm_set, dtype=float).copy()

    Ybus = make_ybus(sys) if Ybus is None else Ybus
    Cg = sys.Cg
    on = sys.gen_status.astype(bool)

    bus_type = sys.bus_type.copy()

    for _outer in range(20):  # 무효전력 한계 처리를 위한 바깥 루프
        slack = int(np.flatnonzero(bus_type == SLACK_)[0])
        pv = np.flatnonzero(bus_type == PV_)
        pq = np.flatnonzero(bus_type == PQ_)
        pvpq = np.sort(np.concatenate([pv, pq]))

        # ---- 초기치 -------------------------------------------------------
        Vm = np.ones(sys.nb) if Vm0 is None else np.asarray(Vm0, float).copy()
        Va = np.zeros(sys.nb) if Va0 is None else np.asarray(Va0, float).copy()
        Vm[sys.gen_bus[on]] = Vm_set[on]
        Vm[slack] = sys.Vm0[slack]
        Va[slack] = sys.Va0[slack]
        V = Vm * np.exp(1j * Va)

        # ---- 지령값 -------------------------------------------------------
        # PV/PQ 모선의 P, PQ 모선의 Q 는 알려진 값이다.
        Psp = (Cg @ (Pg * on)) - sys.Pd
        Qsp = (Cg @ (sys.Qg0 * on)) - sys.Qd

        history: list[float] = []
        converged = False
        it = 0

        for it in range(1, max_iter + 1):
            S = sbus_from_V(Ybus, V)
            mis_p = Psp[pvpq] - np.real(S)[pvpq]
            mis_q = Qsp[pq] - np.imag(S)[pq]
            F = np.concatenate([mis_p, mis_q])

            norm = float(np.max(np.abs(F))) if F.size else 0.0
            history.append(norm)
            if verbose:
                print(f"  [PF] iter {it - 1:2d}  max|mismatch| = {norm:.3e}")
            if norm < tol:
                converged = True
                it -= 1
                break

            J = _build_jacobian(Ybus, V, pvpq, pq)
            dx = spla.spsolve(J.tocsc(), F)

            npvpq = len(pvpq)
            Va[pvpq] += dx[:npvpq]
            Vm[pq] += dx[npvpq:]
            V = Vm * np.exp(1j * Va)
            Vm = np.abs(V)
            Va = np.angle(V)
        else:
            S = sbus_from_V(Ybus, V)
            norm = float(
                np.max(np.abs(np.concatenate([Psp[pvpq] - np.real(S)[pvpq],
                                              Qsp[pq] - np.imag(S)[pq]])))
            )
            history.append(norm)

        # ---- 발전기 출력 역산 ---------------------------------------------
        S = sbus_from_V(Ybus, V)
        Pg_out = Pg.copy()
        Qg_out = sys.Qg0.copy()
        # 슬랙: P, Q 모두 계산값. PV: Q 만 계산값.
        for b in np.concatenate([[slack], pv]).astype(int):
            gens = np.flatnonzero((sys.gen_bus == b) & on)
            if gens.size == 0:
                continue
            # 같은 모선에 여러 대면 잔여분을 첫 번째 기기에 몰아준다(관례).
            if b == slack:
                total_p = np.real(S)[b] + sys.Pd[b]
                Pg_out[gens] = 0.0
                Pg_out[gens[0]] = total_p - Pg[gens[1:]].sum() if gens.size > 1 else total_p
                if gens.size > 1:
                    Pg_out[gens[1:]] = Pg[gens[1:]]
            total_q = np.imag(S)[b] + sys.Qd[b]
            Qg_out[gens] = 0.0
            Qg_out[gens[0]] = total_q

        if not enforce_q_limits or not converged:
            break

        # ---- PV -> PQ 전환 ------------------------------------------------
        viol = np.zeros(sys.nb, dtype=bool)
        for b in pv:
            gens = np.flatnonzero((sys.gen_bus == b) & on)
            if gens.size == 0:
                continue
            q = Qg_out[gens].sum()
            qmax, qmin = sys.Qmax[gens].sum(), sys.Qmin[gens].sum()
            if q > qmax + 1e-9:
                bus_type[b] = PQ_
                sys.Qg0[gens[0]] = qmax - sys.Qg0[gens[1:]].sum()
                viol[b] = True
            elif q < qmin - 1e-9:
                bus_type[b] = PQ_
                sys.Qg0[gens[0]] = qmin - sys.Qg0[gens[1:]].sum()
                viol[b] = True
        if not viol.any():
            break

    return PowerFlowResult(
        converged=converged,
        iterations=it,
        Vm=np.abs(V),
        Va=np.angle(V),
        Pg=Pg_out,
        Qg=Qg_out,
        max_mismatch=history[-1] if history else 0.0,
        history=history,
    )


# case 모듈과의 순환 참조를 피하려고 상수만 로컬에 둔다.
PQ_, PV_, SLACK_ = 1, 2, 3


def line_loading(sys, V: np.ndarray) -> np.ndarray:
    """브랜치 조류 이용률 [%]. ``rate_a`` 가 0 인 브랜치는 ``nan``."""
    Sf, St = branch_flows(sys, V)
    smax = np.maximum(np.abs(Sf), np.abs(St))
    with np.errstate(divide="ignore", invalid="ignore"):
        loading = np.where(sys.rate_a > 0, 100.0 * smax / sys.rate_a, np.nan)
    return loading
