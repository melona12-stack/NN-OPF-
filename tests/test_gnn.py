"""그래프 어텐션 대체모델 회귀 테스트 — M4.

이 모델의 존재 이유는 하나다: **선로를 지우는 것이 곧 메시지 경로를 지우는
것**이어야 한다. 그게 진짜인지 여기서 못박는다. 나머지는 MLP 와 계약이
같은지 확인하는 것들이다 — 계약이 갈리면 비교가 무너진다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnopf import generate_dataset, load_case  # noqa: E402
from nnopf.case import PQ, SLACK  # noqa: E402
from nnopf.gnn import GATSpec, PowerFlowGAT  # noqa: E402
from nnopf.models import IOLayout, SurrogateSpec  # noqa: E402
from nnopf.train import TrainConfig, evaluate, prepare, train  # noqa: E402

CASE = "case30"


@pytest.fixture(scope="module")
def ds():
    return generate_dataset(CASE, n_samples=300, seed=11, workers=2, verbose=False)


def _spec(**kw):
    return GATSpec(**{"hidden": 32, "layers": 2, "heads": 4, **kw})


# ---------------------------------------------------------------- 핵심 주장
def test_dead_line_carries_no_message(ds):
    """끊긴 선로의 파라미터를 아무리 바꿔도 출력이 그대로여야 한다.

    이것이 "선로를 지우는 것 = 메시지 경로를 지우는 것" 의 조작적 정의다.
    게이팅이 없으면 끊긴 선로도 어텐션에 참여하므로 출력이 흔들린다.
    """
    b, model = prepare(ds, _spec(gate=True), case=CASE)
    model.eval()

    x = b.X[:16].clone()
    dead = 3
    x[:, 4 * model.nb + dead] = 0.0            # 선로 3 을 끊는다
    with torch.no_grad():
        base_vm, base_va = model(x)

    # 그 선로의 임피던스를 100 배로 바꾼다 — 죽은 선로면 영향이 없어야 한다
    with torch.no_grad():
        hit = model.edge_line == dead
        assert hit.sum() == 2, "양방향 엣지 두 개여야 한다"
        model.edge_attr_base[hit, :4] *= 100.0
        moved_vm, moved_va = model(x)

    assert torch.equal(base_vm, moved_vm)
    assert torch.equal(base_va, moved_va)


def test_without_gate_dead_line_still_leaks(ds):
    """대조군 — 게이팅을 끄면 끊긴 선로가 출력에 영향을 준다.

    위 테스트가 "우연히 통과" 한 것이 아님을 보이는 절제 실험이다.
    """
    b, model = prepare(ds, _spec(gate=False), case=CASE)
    model.eval()
    x = b.X[:16].clone()
    dead = 3
    x[:, 4 * model.nb + dead] = 0.0
    with torch.no_grad():
        v0, _ = model(x)
        model.edge_attr_base[model.edge_line == dead, :4] *= 100.0
        v1, _ = model(x)
    assert not torch.equal(v0, v1), "게이팅 없이도 막히면 이 실험이 무의미하다"


def test_self_loop_survives_full_isolation(ds):
    """인접 선로가 **전부** 끊겨도 NaN 이 나오지 않아야 한다.

    소프트맥스 분모가 0 이 되는 경우다. 자기 자신으로 가는 엣지를 넣어 둔
    이유가 이것이고, 미지 N-1 에서 실제로 일어날 수 있다.
    """
    b, model = prepare(ds, _spec(), case=CASE)
    model.eval()
    x = b.X[:4].clone()
    x[:, 4 * model.nb :] = 0.0                 # 모든 선로를 끊는다
    with torch.no_grad():
        Vm, Va = model(x)
    assert torch.isfinite(Vm).all() and torch.isfinite(Va).all()


# ---------------------------------------------------------------- 계약 일치
def test_same_io_contract_as_mlp(ds):
    """MLP 와 입력·출력 모양이 같아야 학습·평가 코드를 공유할 수 있다."""
    b_m, mlp = prepare(ds, SurrogateSpec(hidden=32, layers=2), case=CASE)
    b_g, gat = prepare(ds, _spec(), case=CASE)

    assert torch.equal(b_m.X, b_g.X), "입력 텐서가 같아야 한다"
    for m in (mlp, gat):
        Vm, Va = m(b_m.X[:8])
        assert Vm.shape == Va.shape == (8, b_m.sys.nb)
    assert torch.equal(mlp.pq_idx, gat.pq_idx)
    assert torch.equal(mlp.va_idx, gat.va_idx)


def test_known_values_filled_exactly(ds):
    """아는 값은 예측하지 않는다 — PV·슬랙 전압과 슬랙 위상 (06 문서 §1)."""
    sysm = load_case(CASE)
    b, model = prepare(ds, _spec(), case=CASE)
    with torch.no_grad():
        Vm, Va = model(b.X[:8])

    not_pq = np.flatnonzero(sysm.bus_type != PQ)
    assert torch.allclose(
        Vm[:, not_pq], model.v_set[not_pq].expand(8, len(not_pq)), atol=0
    )
    sl = np.flatnonzero(sysm.bus_type == SLACK)
    assert torch.allclose(Va[:, sl], model.va_ref[sl].expand(8, len(sl)), atol=0)


def test_scaled_head_stays_in_box(ds):
    """스케일링 인자 헤드는 출력을 상자 안에 **구조적으로** 가둔다."""
    b, model = prepare(ds, _spec(vm_head="scaled"), case=CASE)
    with torch.no_grad():
        Vm, _ = model(b.X[:64] * 50.0)         # 말도 안 되는 입력을 넣어도
    vm = Vm[:, model.pq_idx]
    assert (vm >= model.vm_lo - 1e-6).all() and (vm <= model.vm_hi + 1e-6).all()


def test_trains_and_is_reproducible(ds):
    """학습이 손실을 줄이고, 같은 시드면 같은 결과가 나온다."""
    out = []
    for _ in range(2):
        b, model = prepare(ds, _spec(), case=CASE, seed=0)
        r = train(model, b, TrainConfig(epochs=6, batch=64, seed=0), verbose=False)
        out.append((r["history"][0]["train"], r["history"][-1]["train"],
                    evaluate(model, b, b.split["test"])["vm_mae"]))
    assert out[0][1] < out[0][0], "손실이 줄어야 한다"
    assert out[0] == pytest.approx(out[1], rel=1e-9), "같은 시드면 같은 결과"


def test_linear_skip_starts_at_zero(ds):
    """선형 지름길은 0 에서 출발한다 — 처음엔 순수 GAT 와 같아야 한다."""
    _, model = prepare(ds, _spec(residual=True), case=CASE)
    assert torch.count_nonzero(model.skip.weight) == 0
    assert torch.count_nonzero(model.skip.bias) == 0
