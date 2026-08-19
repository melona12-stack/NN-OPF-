"""2단계: pandapower 대조 정합성 검증.

직접 구현한 :mod:`nnopf.powerflow` / :mod:`nnopf.opf` 결과를
pandapower(``runpp`` / ``runopp``) 결과와 같은 축에 놓고 비교한다.

pandapower 는 내부적으로 PYPOWER 계열 코드를 쓰므로, 같은 ``ppc`` 를 입력하면
**수치적으로 거의 동일한 해**가 나와야 한다. 오차가 크다면 우리 구현에
버그가 있다는 뜻이다.

주의할 점
---------
* ``to_ppc(net, mode="opf")`` 는 ext_grid 전압을 ``Vmax = Vmin = vm_pu`` 로
  고정한다. pandapower 의 ``runopp`` 도 같은 처리를 하므로 비교가 성립한다.
  (MATPOWER 원본 case9 는 슬랙 전압이 자유라 최적비용이 조금 다르다.)
* 선로 한계는 pandapower 에서 ``max_loading_percent`` 가 설정된 경우에만
  걸린다. 양쪽 설정을 반드시 맞춰야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nnopf.case import load_case, load_pandapower_net
from nnopf.opf import solve_acopf
from nnopf.powerflow import solve_power_flow

__all__ = ["compare_power_flow", "compare_opf", "ComparisonReport"]


@dataclass
class ComparisonReport:
    """한 계통에 대한 비교 결과."""

    case: str
    kind: str                      # "pf" 또는 "opf"
    ok: bool
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def format(self) -> str:
        """터미널 출력용 한 줄 요약 + 지표 목록."""
        head = f"[{'PASS' if self.ok else 'FAIL'}] {self.case:>8s}  {self.kind.upper()}"
        body = "".join(f"\n         {k:<26s} {v:.3e}" for k, v in self.metrics.items())
        note = "".join(f"\n         ! {n}" for n in self.notes)
        return head + body + note


def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """위상각 차이를 (-pi, pi] 로 감아서 계산한다."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def compare_power_flow(case: str = "case9", tol: float = 1e-6) -> ComparisonReport:
    """뉴턴-랩슨 조류계산을 pandapower ``runpp`` 와 비교한다."""
    import pandapower as pp

    sysm = load_case(case)
    net = load_pandapower_net(case)
    pp.runpp(net, numba=False, tolerance_mva=1e-10)

    mine = solve_power_flow(sysm, tol=1e-11)

    # pandapower 결과를 ppc 모선 순서로 정렬한다.
    ref_vm = net.res_bus.vm_pu.to_numpy()
    ref_va = np.deg2rad(net.res_bus.va_degree.to_numpy())

    d_vm = float(np.max(np.abs(mine.Vm - ref_vm)))
    d_va = float(np.max(np.abs(_angle_diff(mine.Va, ref_va))))

    # 모선별 주입전력으로도 한 번 더 대조 (발전기 매핑 차이를 우회)
    #
    # 회계 규약을 맞춰야 한다:
    #   내 S_calc = V * conj(Ybus V)  -> 병렬 소자(shunt)는 Ybus 안에 있으므로
    #                                    "외부 소스의 순주입" = P_gen - P_load
    #   pandapower res_bus.p_mw       -> 그 모선에 붙은 모든 요소의 합
    #                                    = P_load - P_gen + P_shunt
    # 따라서 shunt 소비분 S_sh = |V|^2 * conj(Gs + jBs) 를 빼야 같은 축이 된다.
    ref_p = -net.res_bus.p_mw.to_numpy() / sysm.base_mva
    ref_q = -net.res_bus.q_mvar.to_numpy() / sysm.base_mva
    Ybus = sysm.ybus()
    S = mine.V * np.conj(Ybus @ mine.V)
    S_sh = np.abs(mine.V) ** 2 * np.conj(sysm.Gs + 1j * sysm.Bs)
    S_src = S - S_sh
    d_p = float(np.max(np.abs(np.real(S_src) - ref_p)))
    d_q = float(np.max(np.abs(np.imag(S_src) - ref_q)))

    metrics = {
        "max |dVm| [pu]": d_vm,
        "max |dVa| [rad]": d_va,
        "max |dP_bus| [pu]": d_p,
        "max |dQ_bus| [pu]": d_q,
        "my mismatch [pu]": mine.max_mismatch,
        "iterations": float(mine.iterations),
    }
    notes = []
    if not mine.converged:
        notes.append("직접 구현 조류계산이 수렴하지 않았습니다.")
    ok = mine.converged and max(d_vm, d_va, d_p, d_q) < tol
    return ComparisonReport(case, "pf", ok, metrics, notes)


def compare_opf(
    case: str = "case9",
    tol_cost_rel: float = 1e-6,
    tol_vm: float = 1e-4,
    enforce_line_limits: bool = False,
    method: str = "SLSQP",
) -> ComparisonReport:
    """AC-OPF 를 pandapower ``runopp`` 와 비교한다.

    OPF 는 해가 여러 개일 수 있으므로 **비용(목적함수 값)** 을 1차 지표로,
    전압/출력 프로파일을 2차 지표로 본다. 비용이 맞는데 프로파일이 다르면
    최적해가 평평(degenerate)하다는 신호다.
    """
    import pandapower as pp

    sysm = load_case(case)
    net = load_pandapower_net(case)

    if not enforce_line_limits:
        # 양쪽 모두 선로 한계를 걸지 않도록 맞춘다.
        for tbl in ("line", "trafo", "trafo3w"):
            if tbl in net and "max_loading_percent" in net[tbl]:
                net[tbl]["max_loading_percent"] = np.nan

    notes: list[str] = []
    ref_cost = ref_vm = None
    last_exc: Exception | None = None
    # pandapower 의 내부 IPM 은 초기치에 민감해서 한 번에 실패하는 경우가 있다.
    for kwargs in ({}, {"init": "flat"}, {"init": "pf"}):
        try:
            pp.runopp(net, numba=False, **kwargs)
            ref_cost = float(net.res_cost)
            ref_vm = net.res_bus.vm_pu.to_numpy()
            if kwargs:
                notes.append(f"pandapower 는 {kwargs} 옵션에서만 수렴했습니다.")
            break
        except Exception as exc:  # pragma: no cover - 솔버 실패는 환경 의존
            last_exc = exc
            net = load_pandapower_net(case)
            if not enforce_line_limits:
                for tbl in ("line", "trafo", "trafo3w"):
                    if tbl in net and "max_loading_percent" in net[tbl]:
                        net[tbl]["max_loading_percent"] = np.nan
    if ref_cost is None:
        return ComparisonReport(
            case, "opf", False, {}, [f"pandapower runopp 실패(모든 초기치): {last_exc}"]
        )

    mine = solve_acopf(sysm, enforce_line_limits=enforce_line_limits, method=method)

    d_cost = abs(mine.cost - ref_cost)
    rel_cost = d_cost / max(abs(ref_cost), 1e-9)
    d_vm = float(np.max(np.abs(mine.Vm - ref_vm)))

    metrics = {
        "my cost": mine.cost,
        "pandapower cost": ref_cost,
        "abs cost diff": d_cost,
        "rel cost diff": rel_cost,
        "max |dVm| [pu]": d_vm,
        "eq residual [pu]": mine.max_eq_violation,
        "solve time [s]": mine.solve_time,
    }
    # OPF 합격 기준은 "실행가능성 + 목적함수 값" 두 가지다.
    # 전압 프로파일은 최적해가 평평(degenerate)하면 정당하게 달라질 수 있으므로
    # 경고로만 남기고 불합격 사유로 삼지 않는다.
    feasible = mine.max_eq_violation < 1e-6
    if not mine.success:
        notes.append(f"솔버 종료 메시지: {mine.status} (실행가능성/비용은 아래 지표로 판정)")
    if not feasible:
        notes.append("조류방정식 잔차가 큽니다 — 해가 물리적으로 타당하지 않습니다.")
    if rel_cost >= tol_cost_rel:
        notes.append("비용이 다릅니다 — 제약 설정이 어긋났을 가능성이 큽니다.")
    elif d_vm >= tol_vm:
        notes.append(
            f"비용은 같은데 전압이 {d_vm:.2e} pu 다릅니다 — 최적해가 평평할 수 있습니다."
        )

    ok = feasible and rel_cost < tol_cost_rel
    return ComparisonReport(case, "opf", ok, metrics, notes)


DEFAULT_CASES = ["case9", "case14", "case30", "case57", "case118"]


def run_all(cases: list[str] | None = None, do_opf: bool = True) -> list[ComparisonReport]:
    """여러 계통에 대해 조류계산/OPF 정합성 검증을 한 번에 수행한다."""
    cases = cases or DEFAULT_CASES
    reports: list[ComparisonReport] = []
    for c in cases:
        reports.append(compare_power_flow(c))
        if do_opf:
            reports.append(compare_opf(c))
    return reports
