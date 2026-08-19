"""1~2단계 회귀 테스트.

핵심 주장 세 가지를 고정한다.

1. 우리가 만든 Ybus 가 pandapower 내부 Ybus 와 **비트 단위로** 같다.
2. 우리 뉴턴-랩슨 해가 pandapower ``runpp`` 해와 기계정밀도로 같다.
3. 우리 AC-OPF 의 목적함수 값이 pandapower ``runopp`` 와 같고, 해가 실행가능하다.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nnopf.case import PQ, PV, SLACK, load_case
from nnopf.compare import compare_opf, compare_power_flow
from nnopf.opf import solve_acopf
from nnopf.powerflow import solve_power_flow

PF_CASES = ["case9", "case14", "case30", "case57", "case118"]
# case57 은 pandapower 데이터셋 자체가 기저 조류해에서 39개 모선이 0.94 pu 미만
# (최저 0.72 pu)인 병적 케이스라, pandapower 의 runopp 도 수렴하지 않는다.
OPF_CASES = ["case9", "case14", "case30", "case118"]


@pytest.mark.parametrize("name", PF_CASES)
def test_ybus_matches_pandapower(name):
    """Ybus 가 pandapower 내부 Ybus 와 정확히 일치해야 한다."""
    import pandapower as pp
    import pandapower.networks as pn

    sysm = load_case(name)
    net = getattr(pn, name)()
    pp.runpp(net, numba=False)
    ref = net._ppc["internal"]["Ybus"].toarray()
    assert np.max(np.abs(sysm.ybus().toarray() - ref)) == 0.0


@pytest.mark.parametrize("name", PF_CASES)
def test_power_flow_matches_pandapower(name):
    """조류계산 전압/주입전력이 pandapower 와 1e-9 이내로 같아야 한다."""
    rep = compare_power_flow(name, tol=1e-9)
    assert rep.ok, rep.format()


@pytest.mark.parametrize("name", OPF_CASES)
def test_opf_matches_pandapower(name):
    """OPF 목적함수 값이 pandapower 와 1e-6 상대오차 이내여야 한다."""
    rep = compare_opf(name)
    assert rep.ok, rep.format()


@pytest.mark.parametrize("name", ["case9", "case30"])
def test_opf_solution_is_a_true_power_flow_solution(name):
    """OPF 해의 발전 지령을 조류계산에 다시 넣으면 같은 전압이 나와야 한다.

    이 성질이 4단계에서 신경망 대체모델을 평가하는 기준이 된다: 대체모델이
    내놓은 OPF 해도 **진짜 조류방정식**을 만족해야 쓸모가 있다.
    """
    sysm = load_case(name)
    opt = solve_acopf(sysm)
    assert opt.max_eq_violation < 1e-6

    pf = solve_power_flow(sysm, Pg=opt.Pg, Vm_set=opt.Vm[sysm.gen_bus], tol=1e-11)
    assert pf.converged
    assert np.max(np.abs(pf.Vm - opt.Vm)) < 1e-6


@pytest.mark.parametrize("name", OPF_CASES)
def test_opf_respects_bounds(name):
    """OPF 해가 모든 상자제약을 만족해야 한다."""
    sysm = load_case(name)
    opt = solve_acopf(sysm)
    tol = 1e-7
    assert np.all(opt.Vm <= sysm.Vmax + tol)
    assert np.all(opt.Vm >= sysm.Vmin - tol)
    assert np.all(opt.Pg <= sysm.Pmax + tol)
    assert np.all(opt.Pg >= sysm.Pmin - tol)
    assert np.all(opt.Qg <= sysm.Qmax + tol)
    assert np.all(opt.Qg >= sysm.Qmin - tol)


def test_bus_type_partition():
    """슬랙/PV/PQ 분류가 전체 모선을 빠짐없이 한 번씩 덮어야 한다."""
    sysm = load_case("case30")
    assert sysm.bus_type[sysm.slack] == SLACK
    assert set(sysm.bus_type[sysm.pv]) <= {PV}
    assert set(sysm.bus_type[sysm.pq]) <= {PQ}
    assert len(sysm.pv) + len(sysm.pq) + 1 == sysm.nb
    assert np.array_equal(sysm.pvpq, np.setdiff1d(np.arange(sysm.nb), [sysm.slack]))


def test_analytic_jacobian_matches_finite_difference():
    """해석적 야코비안이 수치미분과 일치해야 한다 (뉴턴법 정확성의 근거)."""
    from nnopf.powerflow import dSbus_dV, sbus_from_V

    sysm = load_case("case14")
    Ybus = sysm.ybus()
    rng = np.random.default_rng(0)
    Vm = 1.0 + 0.05 * rng.standard_normal(sysm.nb)
    Va = 0.1 * rng.standard_normal(sysm.nb)

    dS_dVa, dS_dVm = dSbus_dV(Ybus, Vm * np.exp(1j * Va))
    eps = 1e-7
    for k in range(sysm.nb):
        for arr, ana in ((Va, dS_dVa), (Vm, dS_dVm)):
            pert = arr.copy()
            pert[k] += eps
            if arr is Va:
                Sp = sbus_from_V(Ybus, Vm * np.exp(1j * pert))
            else:
                Sp = sbus_from_V(Ybus, pert * np.exp(1j * Va))
            S0 = sbus_from_V(Ybus, Vm * np.exp(1j * Va))
            num = (Sp - S0) / eps
            assert np.max(np.abs(num - np.asarray(ana[:, k].todense()).ravel())) < 1e-4
