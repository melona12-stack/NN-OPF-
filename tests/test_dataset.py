"""3단계 데이터 생성기 회귀 테스트.

가장 중요한 주장 하나: **생성된 라벨이 진짜 조류방정식을 만족한다.**
이게 깨지면 그 위에 올리는 학습은 전부 무의미하므로 제일 먼저 고정한다.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nnopf.case import PQ, PV, SLACK, load_case
from nnopf.dataset import (
    ScenarioConfig,
    PowerFlowDataset,
    build_edges,
    generate_dataset,
    physics_residual,
    renewable_generators,
    valid_contingencies,
)


@pytest.fixture(scope="module")
def ds():
    """작은 데이터셋 하나를 모듈 전체에서 재사용한다."""
    return generate_dataset("case30", n_samples=200, seed=2026, workers=2, verbose=False)


# --------------------------------------------------------------------------
# 핵심: 라벨이 진짜 조류해인가
# --------------------------------------------------------------------------


def test_labels_satisfy_power_flow_equations(ds):
    """저장된 (부하, 발전, 전압)이 조류방정식을 만족해야 한다.

    이 검사가 4~5단계에서 신경망 예측을 평가하는 바로 그 지표다.
    지금은 정답 라벨을 넣으므로 잔차가 수렴 허용오차 수준이어야 한다.
    """
    sysm = load_case(ds.case)
    worst = 0.0
    for i in range(0, ds.n_samples, 7):
        dP, dQ = physics_residual(
            sysm, ds.Pd[i], ds.Qd[i], ds.p_ren[i], ds.p_gen[i],
            ds.Vm[i], ds.Va[i], outage=int(ds.outage[i]),
        )
        worst = max(worst, float(np.max(np.abs(dP))), float(np.max(np.abs(dQ))))
    # 입력을 float32 로 내린 뒤 풀고 라벨은 float64 로 두므로, 잔차는 뉴턴법
    # 수렴 허용오차 수준(1e-8 pu)이어야 한다. 이 값이 오르면 저장 정밀도나
    # 입력/라벨 정합이 깨진 것이다.
    assert worst < 1e-7, f"조류방정식 잔차가 큽니다: {worst:.3e} pu"


def test_residual_is_masked_by_bus_type(ds):
    """슬랙에는 P/Q 잔차가, PV 에는 Q 잔차가 걸리지 않아야 한다."""
    sysm = load_case(ds.case)
    dP, dQ = physics_residual(
        sysm, ds.Pd[0], ds.Qd[0], ds.p_ren[0], ds.p_gen[0], ds.Vm[0], ds.Va[0]
    )
    assert dP[sysm.bus_type == SLACK] == pytest.approx(0.0)
    assert dQ[sysm.bus_type == SLACK] == pytest.approx(0.0)
    assert np.allclose(dQ[sysm.bus_type == PV], 0.0)


# --------------------------------------------------------------------------
# 재현성 — 데이터를 옮기지 않고 재생성하는 전략의 근거
# --------------------------------------------------------------------------


def test_same_seed_gives_identical_data():
    """같은 시드면 비트 단위로 같은 데이터가 나와야 한다."""
    a = generate_dataset("case30", n_samples=60, seed=7, workers=2, verbose=False)
    b = generate_dataset("case30", n_samples=60, seed=7, workers=2, verbose=False)
    for f in ("Pd", "Qd", "p_ren", "p_gen", "Vm", "Va", "outage"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f"{f} 가 다릅니다"


def test_worker_count_does_not_change_data():
    """워커 수를 바꿔도 같은 데이터여야 한다 (표본별 시드 파생의 목적)."""
    a = generate_dataset("case30", n_samples=60, seed=7, workers=1, chunk=60, verbose=False)
    b = generate_dataset("case30", n_samples=60, seed=7, workers=4, chunk=7, verbose=False)
    assert np.array_equal(a.Vm, b.Vm)
    assert np.array_equal(a.outage, b.outage)


def test_different_seed_gives_different_data():
    a = generate_dataset("case30", n_samples=40, seed=1, workers=2, verbose=False)
    b = generate_dataset("case30", n_samples=40, seed=2, workers=2, verbose=False)
    assert not np.array_equal(a.Pd, b.Pd)


# --------------------------------------------------------------------------
# 계통 구조에서 파생되는 것
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["case30", "case118"])
def test_contingencies_never_island_the_network(name):
    """N-1 후보로 고른 고장은 계통을 분리하지 않아야 한다."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    sysm = load_case(name)
    cont = valid_contingencies(sysm)
    assert cont.size > 0
    for line in cont[:: max(1, len(cont) // 12)]:
        mask = sysm.br_status.astype(bool).copy()
        mask[line] = False
        adj = sp.coo_matrix(
            (np.ones(int(mask.sum())), (sysm.f_bus[mask], sysm.t_bus[mask])),
            shape=(sysm.nb, sysm.nb),
        )
        assert connected_components(adj, directed=False)[0] == 1


def test_renewable_selection_matches_paper_ratio():
    """재생에너지 발전기는 슬랙 제외 PV 발전기의 60% 여야 한다 (P2 §4.1)."""
    for name, expected in (("case30", 3), ("case118", 32)):
        sysm = load_case(name)
        cfg = ScenarioConfig()
        ren = renewable_generators(sysm, cfg, seed=2026)
        assert len(ren) == expected, f"{name}: {len(ren)} != {expected}"
        assert sysm.slack not in set(sysm.gen_bus[ren].tolist())


def test_renewable_selection_is_seed_stable():
    sysm = load_case("case118")
    cfg = ScenarioConfig()
    assert np.array_equal(
        renewable_generators(sysm, cfg, 2026), renewable_generators(sysm, cfg, 2026)
    )


# --------------------------------------------------------------------------
# 특징 구성 (P2 §3.1)
# --------------------------------------------------------------------------


def test_node_features_shape_and_order(ds):
    """노드 특징 8개: Pload, Qload, Pren, Pgen, Vset, isPQ, isPV, isref."""
    X = ds.node_features()
    assert X.shape == (ds.n_samples, ds.n_bus, 8)
    assert np.allclose(X[:, :, 0], ds.Pd)
    assert np.allclose(X[:, :, 1], ds.Qd)
    assert np.allclose(X[:, :, 2], ds.p_ren)
    assert np.allclose(X[:, :, 3], ds.p_gen)
    # 모선 종류 원-핫은 정확히 하나만 1
    assert np.all(X[0, :, 5:8].sum(axis=-1) == 1.0)


def test_slack_pgen_is_zero(ds):
    """슬랙 출력은 추론 시점에 알 수 없으므로 입력에서 0 이어야 한다."""
    sysm = load_case(ds.case)
    assert np.allclose(ds.p_gen[:, sysm.slack], 0.0)


def test_edge_features_encode_outage_as_status(ds):
    """고장 선로는 그래프에서 제거하지 않고 status=0 으로만 표시해야 한다."""
    E = ds.edge_attr()
    assert E.shape == (ds.n_samples, ds.edge_index.shape[1], 7)
    faulted = np.flatnonzero(ds.outage >= 0)
    assert faulted.size > 0
    for k in faulted[:5]:
        line = ds.outage[k]
        assert np.all(E[k][ds.edge_line == line, 4] == 0.0)
        assert np.all(E[k][ds.edge_line != line, 4] == 1.0)
    # 정상 표본은 전부 1
    normal = np.flatnonzero(ds.outage < 0)
    assert np.all(E[normal[0]][:, 4] == 1.0)


def test_edges_are_bidirectional():
    """브랜치마다 양방향 엣지가 있어야 한다."""
    sysm = load_case("case30")
    ei, attr, line = build_edges(sysm)
    assert ei.shape == (2, 2 * sysm.nl)
    assert attr.shape == (2 * sysm.nl, 7)
    assert np.array_equal(ei[0, : sysm.nl], ei[1, sysm.nl :])
    assert np.array_equal(ei[1, : sysm.nl], ei[0, sysm.nl :])


# --------------------------------------------------------------------------
# 분할
# --------------------------------------------------------------------------


def test_unseen_n1_split_has_no_overlapping_fault_types(ds):
    """테스트용 고장 유형은 학습·검증에 단 한 번도 등장하면 안 된다 (P2 §4.7)."""
    sp = ds.split_unseen_n1()
    kinds = {k: set(np.unique(ds.outage[v]).tolist()) - {-1} for k, v in sp.items()}
    assert not (kinds["train"] & kinds["test"])
    assert not (kinds["val"] & kinds["test"])
    assert not (kinds["train"] & kinds["val"])


def test_splits_partition_all_samples(ds):
    for sp in (ds.split_random(), ds.split_unseen_n1()):
        allidx = np.sort(np.concatenate(list(sp.values())))
        assert np.array_equal(allidx, np.arange(ds.n_samples))


# --------------------------------------------------------------------------
# 저장/불러오기
# --------------------------------------------------------------------------


def test_save_load_roundtrip(ds, tmp_path):
    path = tmp_path / "ds.npz"
    ds.save(str(path))
    back = PowerFlowDataset.load(str(path))
    assert back.case == ds.case and back.seed == ds.seed
    assert back.config == ds.config
    for f in ("Pd", "Qd", "p_ren", "p_gen", "Vm", "Va", "outage", "edge_index"):
        assert np.array_equal(getattr(back, f), getattr(ds, f)), f


# --------------------------------------------------------------------------
# 발전 배분 — 조용히 슬랙에 떠넘기지 않는가
# --------------------------------------------------------------------------


def test_allocate_dispatch_respects_limits_and_hits_target():
    """한계 안에서 목표를 맞춰야 한다."""
    from nnopf.dataset import allocate_dispatch

    pmin = np.zeros(5)
    pmax = np.array([1.0, 2.0, 3.0, 0.5, 1.5])
    a = allocate_dispatch(6.0, pmin, pmax, weight=pmax, jitter=np.ones(5))
    assert np.all(a <= pmax + 1e-9) and np.all(a >= pmin - 1e-9)
    assert a.sum() == pytest.approx(6.0, abs=1e-8)


def test_allocate_dispatch_redistributes_after_clipping():
    """한 대가 상한에 걸리면 남은 몫을 다른 발전기가 받아야 한다.

    단순 clip 만 하면 잘린 만큼이 전부 슬랙으로 떨어져 조류계산이 발산한다.
    """
    from nnopf.dataset import allocate_dispatch

    pmin = np.zeros(3)
    pmax = np.array([0.1, 5.0, 5.0])   # 첫 발전기가 아주 작다
    # 비중은 균등이라 순진하게 배분하면 첫 대가 상한에 걸린다
    a = allocate_dispatch(6.0, pmin, pmax, weight=np.ones(3), jitter=np.ones(3))
    assert a[0] == pytest.approx(0.1)
    assert a.sum() == pytest.approx(6.0, abs=1e-8), "잘린 몫이 재배분되지 않았습니다"


def test_allocate_dispatch_saturates_when_capacity_is_short():
    """용량 자체가 모자라면 전부 상한에 붙고 그대로 끝나야 한다 (무한루프 금지)."""
    from nnopf.dataset import allocate_dispatch

    pmax = np.array([1.0, 1.0])
    a = allocate_dispatch(10.0, np.zeros(2), pmax, weight=pmax, jitter=np.ones(2))
    assert np.allclose(a, pmax)


# --------------------------------------------------------------------------
# 용량 사전 검사 — 발산 데이터를 조용히 만들지 않는가
# --------------------------------------------------------------------------


def test_capacity_check_blocks_clearly_infeasible_config():
    """용량이 확실히 모자란 설정은 데이터를 만들기 전에 막아야 한다."""
    from nnopf.dataset import check_capacity_feasible

    sysm = load_case("case300")
    bad = ScenarioConfig(ren_capacity_frac=0.15)
    with pytest.raises(ValueError, match="발전 용량이 부족"):
        check_capacity_feasible(sysm, bad, seed=2026)


def test_capacity_check_passes_paper_defaults():
    """P2 기본 설정은 통과해야 한다 (case30/case118)."""
    from nnopf.dataset import check_capacity_feasible

    for name in ("case30", "case118"):
        check_capacity_feasible(load_case(name), ScenarioConfig(), seed=2026)


def test_generate_raises_before_producing_garbage():
    """generate_dataset 도 같은 검사를 거쳐야 한다."""
    with pytest.raises(ValueError, match="발전 용량이 부족"):
        generate_dataset("case300", n_samples=10,
                         config=ScenarioConfig(ren_capacity_frac=0.15),
                         seed=2026, workers=1, verbose=False)
