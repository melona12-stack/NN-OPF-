r"""교류 최적조류계산 (AC-OPF) — 비선형계획법 직접 구현.

정식화
------
결정변수

.. math::
    x = [\;\theta_1 \dots \theta_{nb},\;
          |V|_1 \dots |V|_{nb},\;
          P^g_1 \dots P^g_{ng},\;
          Q^g_1 \dots Q^g_{ng}\;]

목적함수 (2차 발전비용)

.. math::
    \min_x \; \sum_{i=1}^{ng}
        \left( c_{2,i} P_{g,i}^2 + c_{1,i} P_{g,i} + c_{0,i} \right)
    \quad [P_{g}\ \text{단위: MW}]

등식제약 — 모든 모선의 전력수급 균형 (조류방정식)

.. math::
    P_i(\theta, |V|) - \left(\textstyle\sum_{g \in i} P^g_g - P^d_i\right) = 0 \\
    Q_i(\theta, |V|) - \left(\textstyle\sum_{g \in i} Q^g_g - Q^d_i\right) = 0

부등식제약

.. math::
    V^{min}_i \le |V|_i \le V^{max}_i, \qquad
    P^{min}_g \le P^g_g \le P^{max}_g, \qquad
    Q^{min}_g \le Q^g_g \le Q^{max}_g \\
    |S_f|^2 \le (S^{max})^2, \qquad |S_t|^2 \le (S^{max})^2
    \quad (\text{선로 조류 한계, 선택})

슬랙 모선의 위상각은 :math:`\theta_{slack} = 0` 으로 고정한다.

**왜 이 정식화가 중요한가** — 4단계에서 신경망이 대체하는 대상이 바로
위의 *등식제약(조류방정식)* 이다. 즉 :math:`P_i(\theta,|V|)`,
:math:`Q_i(\theta,|V|)` 를 학습된 함수로 갈아끼우는 것이 목표이므로,
이 부분을 명시적으로 분리해 두었다 (:func:`power_flow_residual`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import NonlinearConstraint, minimize

from nnopf.powerflow import dSbus_dV, sbus_from_V
from nnopf.ybus import make_ybus, make_connection_matrices, make_branch_admittance

__all__ = ["solve_acopf", "ACOPFResult", "OPFVariables"]


@dataclass
class ACOPFResult:
    """AC-OPF 해."""

    success: bool
    status: str
    cost: float               # 총 발전비용 [비용단위/h]
    Vm: np.ndarray
    Va: np.ndarray
    Pg: np.ndarray            # [pu]
    Qg: np.ndarray            # [pu]
    n_iter: int
    max_eq_violation: float   # 조류방정식 최대 잔차 [pu]
    solve_time: float = 0.0
    raw: object = field(default=None, repr=False)

    @property
    def V(self) -> np.ndarray:
        return self.Vm * np.exp(1j * self.Va)

    def pg_mw(self, base_mva: float) -> np.ndarray:
        return self.Pg * base_mva

    def qg_mvar(self, base_mva: float) -> np.ndarray:
        return self.Qg * base_mva


class OPFVariables:
    """결정변수 벡터 ``x`` 의 슬라이싱을 한곳에서 관리한다."""

    def __init__(self, nb: int, ng: int):
        self.nb, self.ng = nb, ng
        self.va = slice(0, nb)
        self.vm = slice(nb, 2 * nb)
        self.pg = slice(2 * nb, 2 * nb + ng)
        self.qg = slice(2 * nb + ng, 2 * nb + 2 * ng)
        self.n = 2 * nb + 2 * ng

    def unpack(self, x: np.ndarray):
        """``x`` -> ``(Va, Vm, Pg, Qg)``."""
        return x[self.va], x[self.vm], x[self.pg], x[self.qg]

    def pack(self, Va, Vm, Pg, Qg) -> np.ndarray:
        """``(Va, Vm, Pg, Qg)`` -> ``x``."""
        x = np.empty(self.n)
        x[self.va], x[self.vm], x[self.pg], x[self.qg] = Va, Vm, Pg, Qg
        return x


# --------------------------------------------------------------------------
# 조류방정식 잔차 — 4단계에서 신경망으로 대체될 부분
# --------------------------------------------------------------------------


def power_flow_residual(sys, Ybus, Va, Vm, Pg, Qg) -> np.ndarray:
    r"""조류방정식 잔차 :math:`[\Delta P; \Delta Q]` ``(2*nb,)`` [pu].

    이 함수가 곧 "물리 모델"이다. 신경망 대체모델은 이 함수와 **같은 입출력
    계약**(전압/출력 -> 잔차, 또는 주입 -> 전압)을 만족해야 한다.
    """
    V = Vm * np.exp(1j * Va)
    S_calc = sbus_from_V(Ybus, V)
    S_inj = (sys.Cg @ Pg - sys.Pd) + 1j * (sys.Cg @ Qg - sys.Qd)
    d = S_calc - S_inj
    return np.concatenate([np.real(d), np.imag(d)])


def power_flow_jacobian(sys, Ybus, Va, Vm) -> sp.csr_matrix:
    """:func:`power_flow_residual` 의 ``x`` 에 대한 해석적 야코비안 ``(2*nb, n)``."""
    V = Vm * np.exp(1j * Va)
    dS_dVa, dS_dVm = dSbus_dV(Ybus, V)
    Cg = sp.csr_matrix(sys.Cg)
    Z = sp.csr_matrix((sys.nb, sys.ng))
    return sp.bmat(
        [
            [np.real(dS_dVa), np.real(dS_dVm), -Cg, Z],
            [np.imag(dS_dVa), np.imag(dS_dVm), Z, -Cg],
        ],
        format="csr",
    )


# --------------------------------------------------------------------------
# 선로 조류 한계
# --------------------------------------------------------------------------


def _branch_flow_matrices(sys):
    """조류 계산용 ``(Yf, Yt, Cf, Ct)`` 를 만든다."""
    nl, nb = sys.nl, sys.nb
    Yff, Yft, Ytf, Ytt = make_branch_admittance(sys)
    Cf, Ct = make_connection_matrices(sys)
    rows = np.arange(nl)
    Yf = sp.csr_matrix((Yff, (rows, sys.f_bus)), shape=(nl, nb)) + sp.csr_matrix(
        (Yft, (rows, sys.t_bus)), shape=(nl, nb)
    )
    Yt = sp.csr_matrix((Ytf, (rows, sys.f_bus)), shape=(nl, nb)) + sp.csr_matrix(
        (Ytt, (rows, sys.t_bus)), shape=(nl, nb)
    )
    return Yf, Yt, Cf, Ct


def _dSbr_dV(Y, C, V):
    """브랜치 조류의 전압 편미분 ``(dS_dVa, dS_dVm)`` (MATPOWER ``dSbr_dV``)."""
    I = Y @ V
    Vnorm = V / np.abs(V)
    diagI = sp.diags(I)
    diagV = sp.diags(V)
    diagVbr = sp.diags(C @ V)
    dS_dVa = 1j * (np.conj(diagI) @ C @ diagV - diagVbr @ np.conj(Y @ diagV))
    dS_dVm = diagVbr @ np.conj(Y @ sp.diags(Vnorm)) + np.conj(diagI) @ C @ sp.diags(Vnorm)
    return dS_dVa, dS_dVm


# --------------------------------------------------------------------------
# 메인 솔버
# --------------------------------------------------------------------------


def solve_acopf(
    sys,
    enforce_line_limits: bool = False,
    x0: np.ndarray | None = None,
    warm_start: bool = True,
    method: str = "SLSQP",
    tol: float = 1e-9,
    max_iter: int = 500,
    verbose: bool = False,
) -> ACOPFResult:
    """교류 최적조류계산을 푼다.

    Parameters
    ----------
    sys
        :class:`nnopf.case.PowerSystem`
    enforce_line_limits
        True 면 ``rate_a > 0`` 인 브랜치에 :math:`|S| \\le S^{max}` 를 건다.
        pandapower 는 ``max_loading_percent`` 가 설정된 경우에만 이를 걸므로,
        정합성 비교 시에는 양쪽 설정을 맞춰야 한다.
    method
        ``"SLSQP"`` (기본, 중소 계통에서 빠름) 또는 ``"trust-constr"``.
    x0
        초기 결정변수. ``None`` 이면 ``warm_start`` 설정에 따라 자동 생성.
    warm_start
        True 면 먼저 조류계산을 한 번 풀어 그 전압해를 초기치로 쓴다.
        평기동보다 훨씬 좋은 출발점이라 대형 계통에서 반복 횟수가 크게 준다.

    Returns
    -------
    ACOPFResult
    """
    import time

    nb, ng = sys.nb, sys.ng
    var = OPFVariables(nb, ng)
    Ybus = make_ybus(sys)
    on = sys.gen_status.astype(bool)
    base = sys.base_mva

    # ---------------------------------------------------------------- 목적함수
    # 비용식은 MW 기준이므로 pu -> MW 환산 계수를 미리 접어 둔다.
    a = sys.cost_c2 * base**2          # * Pg_pu^2
    b = sys.cost_c1 * base             # * Pg_pu
    c = np.where(on, sys.cost_c0, 0.0)

    def raw_cost(x: np.ndarray) -> float:
        """실제 비용 [비용단위/h]."""
        pg = x[var.pg]
        return float(np.sum(a * pg**2 + b * pg + c))

    # 목적함수를 O(1) 로 정규화한다.
    # SLSQP 의 ``ftol`` 은 목적함수의 **절대** 변화량 기준이라, 비용이 1e5 규모인
    # 대형 계통에서는 사실상 1e-14 상대정밀도를 요구하게 되어 수렴 직전에
    # "Positive directional derivative for linesearch" 로 멈춘다. 스케일을 나눠
    # 주면 ``ftol`` 이 상대 기준으로 동작한다.
    cost_scale = 1.0

    def objective(x: np.ndarray) -> float:
        pg = x[var.pg]
        return float(np.sum(a * pg**2 + b * pg + c)) / cost_scale

    def objective_grad(x: np.ndarray) -> np.ndarray:
        g = np.zeros(var.n)
        g[var.pg] = (2.0 * a * x[var.pg] + b) / cost_scale
        return g

    # ------------------------------------------------------------ 등식제약
    def eq_fun(x: np.ndarray) -> np.ndarray:
        Va, Vm, Pg, Qg = var.unpack(x)
        return power_flow_residual(sys, Ybus, Va, Vm, Pg, Qg)

    def eq_jac(x: np.ndarray):
        Va, Vm, _, _ = var.unpack(x)
        return power_flow_jacobian(sys, Ybus, Va, Vm)

    # ---------------------------------------------------------- 부등식제약
    limited = np.flatnonzero((sys.rate_a > 0) & (sys.br_status == 1)) if enforce_line_limits else np.array([], dtype=int)
    Yf, Yt, Cf, Ct = _branch_flow_matrices(sys)

    def ineq_fun(x: np.ndarray) -> np.ndarray:
        """``rate^2 - |S|^2 >= 0`` (송단/수단 각각)."""
        Va, Vm, _, _ = var.unpack(x)
        V = Vm * np.exp(1j * Va)
        Sf = (Cf @ V)[limited] * np.conj(Yf @ V)[limited]
        St = (Ct @ V)[limited] * np.conj(Yt @ V)[limited]
        r2 = sys.rate_a[limited] ** 2
        return np.concatenate([r2 - np.abs(Sf) ** 2, r2 - np.abs(St) ** 2])

    def ineq_jac(x: np.ndarray) -> np.ndarray:
        Va, Vm, _, _ = var.unpack(x)
        V = Vm * np.exp(1j * Va)
        rows = []
        for Y, C in ((Yf, Cf), (Yt, Ct)):
            dS_dVa, dS_dVm = _dSbr_dV(Y, C, V)
            S = (C @ V) * np.conj(Y @ V)
            S_l = S[limited]
            dVa = dS_dVa.tocsr()[limited, :]
            dVm = dS_dVm.tocsr()[limited, :]
            # d(|S|^2) = 2*(Re S * Re dS + Im S * Im dS) 이고 제약은 부호가 반대
            gVa = -2.0 * (
                sp.diags(np.real(S_l)) @ np.real(dVa) + sp.diags(np.imag(S_l)) @ np.imag(dVa)
            )
            gVm = -2.0 * (
                sp.diags(np.real(S_l)) @ np.real(dVm) + sp.diags(np.imag(S_l)) @ np.imag(dVm)
            )
            block = np.zeros((len(limited), var.n))
            block[:, var.va] = gVa.toarray()
            block[:, var.vm] = gVm.toarray()
            rows.append(block)
        return np.vstack(rows)

    # ------------------------------------------------------------------ 경계
    lb = np.empty(var.n)
    ub = np.empty(var.n)
    lb[var.va], ub[var.va] = -2 * np.pi, 2 * np.pi
    slack = sys.slack
    lb[var.va.start + slack] = ub[var.va.start + slack] = sys.Va0[slack]
    lb[var.vm], ub[var.vm] = sys.Vmin, sys.Vmax
    lb[var.pg] = np.where(on, sys.Pmin, 0.0)
    ub[var.pg] = np.where(on, sys.Pmax, 0.0)
    lb[var.qg] = np.where(on, sys.Qmin, 0.0)
    ub[var.qg] = np.where(on, sys.Qmax, 0.0)
    bounds = list(zip(lb, ub))

    # ------------------------------------------------------------------ 초기치
    if x0 is None:
        Va0 = np.zeros(nb)
        Va0[slack] = sys.Va0[slack]
        Vm0 = np.clip(np.ones(nb), sys.Vmin, sys.Vmax)
        # 총 부하를 가동 발전기에 균등 배분한 뒤 출력 한계로 잘라 초기 지령으로 쓴다.
        Pg0 = np.clip(sys.Pd.sum() / max(int(on.sum()), 1) * on, lb[var.pg], ub[var.pg])
        Qg0 = np.clip(np.zeros(ng), lb[var.qg], ub[var.qg])

        if warm_start:
            # 같은 출력 지령으로 조류계산을 한 번 풀어 물리적으로 타당한
            # 전압 프로파일에서 출발한다. (실패하면 조용히 평기동으로 되돌림)
            from nnopf.powerflow import solve_power_flow

            try:
                pf = solve_power_flow(sys, Pg=Pg0, tol=1e-8, max_iter=25)
                if pf.converged:
                    Va0, Vm0 = pf.Va.copy(), np.clip(pf.Vm, sys.Vmin, sys.Vmax)
                    Va0[slack] = sys.Va0[slack]
                    Pg0 = np.clip(pf.Pg, lb[var.pg], ub[var.pg])
                    Qg0 = np.clip(pf.Qg, lb[var.qg], ub[var.qg])
            except Exception:
                pass

        x0 = var.pack(Va0, Vm0, Pg0, Qg0)
    x0 = np.clip(np.asarray(x0, dtype=float), lb, ub)
    cost_scale = max(abs(raw_cost(x0)), 1.0)

    # ------------------------------------------------------------------ 풀이
    t0 = time.perf_counter()
    constraints: list = [
        {"type": "eq", "fun": eq_fun, "jac": lambda x: eq_jac(x).toarray()}
    ]
    if limited.size:
        constraints.append({"type": "ineq", "fun": ineq_fun, "jac": ineq_jac})

    if method == "SLSQP":
        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": max_iter, "ftol": tol, "disp": verbose},
        )
    elif method == "trust-constr":
        cons = [
            NonlinearConstraint(eq_fun, 0.0, 0.0, jac=eq_jac),
        ]
        if limited.size:
            cons.append(NonlinearConstraint(ineq_fun, 0.0, np.inf, jac=ineq_jac))
        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            bounds=list(zip(lb, ub)),
            constraints=cons,
            method="trust-constr",
            options={"maxiter": max_iter, "gtol": tol, "xtol": tol, "verbose": 2 if verbose else 0},
        )
    else:
        raise ValueError(f"지원하지 않는 method: {method}")
    dt = time.perf_counter() - t0

    Va, Vm, Pg, Qg = var.unpack(res.x)
    resid = eq_fun(res.x)

    return ACOPFResult(
        success=bool(res.success),
        status=str(getattr(res, "message", "")),
        cost=raw_cost(res.x),
        Vm=Vm.copy(),
        Va=Va.copy(),
        Pg=Pg.copy(),
        Qg=Qg.copy(),
        n_iter=int(getattr(res, "nit", -1)),
        max_eq_violation=float(np.max(np.abs(resid))),
        solve_time=dt,
        raw=res,
    )
