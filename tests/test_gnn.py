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
    """학습이 손실을 줄이고, 같은 시드면 실질적으로 같은 결과가 나온다.

    .. note::
       **비트 단위 일치는 요구하지 않는다.** 엣지를 노드로 모을 때 쓰는
       ``scatter_add`` 는 더하는 순서가 스레드마다 달라서, float32 에서
       상대 :math:`10^{-7}` 수준의 차이가 남는다. 순전파 한 번에도 생기므로
       **첫 epoch 부터** 갈릴 수 있고, 학습이 진행되며 증폭된다.

       MLP 는 그런 누적이 없어 비트 단위로 재현된다
       (``test_surrogate.py``). 모델 구조가 다르면 재현성의 뜻도 달라진다 —
       "같은 시드면 같은 결론" 이지 "같은 비트" 가 아니다.
    """
    out = []
    for _ in range(2):
        b, model = prepare(ds, _spec(), case=CASE, seed=0)
        r = train(model, b, TrainConfig(epochs=6, batch=64, seed=0), verbose=False)
        out.append((r["history"][0]["train"], r["history"][-1]["train"],
                    evaluate(model, b, b.split["test"])["vm_mae"]))
    assert out[0][1] < out[0][0], "손실이 줄어야 한다"
    # 우리가 주장할 어떤 효과보다도 훨씬 작은 폭 — 결론이 흔들리지 않는다.
    # 더 조이면 통과했다 실패했다 하는 테스트가 된다. 그건 없는 것만 못하다.
    assert out[0] == pytest.approx(out[1], rel=1e-4)


def test_linear_skip_starts_at_zero(ds):
    """선형 지름길은 0 에서 출발한다 — 처음엔 순수 GAT 와 같아야 한다."""
    _, model = prepare(ds, _spec(residual=True), case=CASE)
    assert torch.count_nonzero(model.skip.weight) == 0
    assert torch.count_nonzero(model.skip.bias) == 0


# --------------------------------------------------------------------------
# 층 수를 정하는 값 — 계통 그래프의 지름 (06 문서 §7.7)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,normal,worst_n1", [("case30", 6, 8),
                                                  ("case118", 14, 18)])
def test_graph_diameter_sets_the_layer_count(name, normal, worst_n1):
    """지름은 GAT 층 수의 하한이다. 바뀌면 모델 형태도 바뀌어야 한다.

    L 층 메시지 전달 신경망은 L-hop 까지만 본다. 조류방정식은
    :math:`Y_{bus}^{-1}` 로 계통 전체가 결합하므로, 층 수가 지름보다 작으면
    모델은 **물리적으로 볼 수 없는 것**을 예측해야 한다 (06 문서 §7.7).

    선로가 끊기면 우회 경로가 길어져 지름이 늘어난다. 미지 N-1 분할에서
    필요한 층 수는 정상 지름이 아니라 **N-1 최악 지름**이다.

    이 숫자가 바뀌면 케이스 로더나 선로 목록이 변했다는 뜻이고, 그때는
    ``--layers`` 기본값을 다시 정해야 한다.
    """
    import sys as _s
    from pathlib import Path
    _s.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from s07_graph_diameter import _adjacency, diameter
    from nnopf.case import load_case

    ps = load_case(name)
    assert diameter(_adjacency(ps.f_bus, ps.t_bus))[0] == normal

    worst = max(
        d for k in range(len(ps.f_bus))
        if (d := diameter(_adjacency(ps.f_bus, ps.t_bus, drop=k))[0]) != float("inf")
    )
    assert worst == worst_n1


# --------------------------------------------------- 메모리 절약 장치 (06 §8.2)
#
# case118 GAT 18층이 8 GB 에 안 들어가서 넣은 것들이다. 셋 다 **결과를 바꾸면
# 안 되는** 종류라, 여기서 그걸 못박는다. 하나라도 값을 바꾸면 지금까지의
# 모든 비교가 무의미해진다.
def test_grad_checkpointing_changes_nothing_but_memory(ds):
    """체크포인팅을 켠 모델과 끈 모델의 **기울기가 같아야** 한다.

    역전파 때 순전파를 다시 도는 것뿐이므로 수학적으로 완전히 같다.
    dropout 이 있으면 재계산에서 다른 마스크가 나올 수 있어 기본값 0 을 쓴다.
    """
    sp = ds.split_random(seed=0)

    def grads(checkpoint: bool):
        b, m = prepare(ds, _spec(checkpoint=checkpoint), split=sp, seed=0)
        m.train()
        idx = torch.arange(32)
        Vm, Va = m(b.X[idx])
        loss = ((Vm - b.Vm[idx]) ** 2).mean() + ((Va - b.Va[idx]) ** 2).mean()
        loss.backward()
        return loss.item(), [p.grad.clone() for p in m.parameters() if p.grad is not None]

    l_off, g_off = grads(False)
    l_on, g_on = grads(True)

    assert l_on == pytest.approx(l_off, rel=1e-12)
    assert len(g_on) == len(g_off) > 0, "기울기가 하나도 안 흐르면 시험이 무의미하다"
    for a, c in zip(g_off, g_on):
        assert torch.allclose(a, c, rtol=1e-5, atol=1e-7)


def test_checkpointing_is_off_during_eval(ds):
    """추론에서는 체크포인팅이 걸리지 않아야 한다 — 켜 봐야 손해다."""
    b, m = prepare(ds, _spec(checkpoint=True), split=ds.split_random(seed=0), seed=0)
    m.eval()
    with torch.no_grad():
        Vm1, Va1 = m(b.X[:16])
    b2, m2 = prepare(ds, _spec(checkpoint=False), split=ds.split_random(seed=0), seed=0)
    m2.eval()
    with torch.no_grad():
        Vm2, Va2 = m2(b2.X[:16])
    assert torch.equal(Vm1, Vm2) and torch.equal(Va1, Va2)


def test_validation_chunking_gives_the_same_numbers(ds):
    """검증을 나눠 봐도 지도손실·P/부하 % 가 같아야 한다.

    예전에는 검증 분할을 한 방에 넣었다. case118 GAT 에서는 간선 텐서가
    한 개에 4.5 GB 라 그게 불가능하다. 나눠 재도 값이 같다는 것이 전제다.

    허용오차가 1e-5 인 것은 **수식이 달라서가 아니라 float32 라서**다.
    나눗셈 순서가 바뀌면 마지막 몇 비트가 흔들린다 (실측 상대차 1.2e-7 =
    float32 엡실론). ``evaluate`` 도 청크 크기에 따라 같은 크기로 흔들린다.
    """
    from nnopf.train import validate

    sp = ds.split_random(seed=0)
    b, m = prepare(ds, _spec(), split=sp, seed=0)
    va = torch.as_tensor(sp["val"], dtype=torch.long, device=b.device)
    val_load = b.p_spec[va].abs().sum(-1).mean().clamp(min=1e-9)

    ref = validate(m, b, va, TrainConfig(select="phys", val_chunk=len(va)), val_load)
    for chunk in (7, 16, 64):
        got = validate(m, b, va, TrainConfig(select="phys", val_chunk=chunk), val_load)
        assert got[0] == pytest.approx(ref[0], rel=1e-5), f"지도손실이 청크 {chunk} 에서 달라짐"
        assert got[1] == pytest.approx(ref[1], rel=1e-5), f"P/부하가 청크 {chunk} 에서 달라짐"


def test_amp_is_a_no_op_on_cpu(ds):
    """CPU 에서는 --amp 를 줘도 학습 결과가 비트 단위로 같아야 한다.

    bf16 커널이 CPU 에 다 있지도 않고 이득도 없어서 무동작으로 두었다.
    이게 깨지면 CPU 로 낸 기존 결과와 GPU 결과를 나란히 놓을 수 없게 된다.
    """
    sp = ds.split_random(seed=0)
    out = []
    for amp in ("off", "bf16"):
        b, m = prepare(ds, _spec(), split=sp, seed=0)
        train(m, b, TrainConfig(epochs=3, batch=64, seed=0, amp=amp), verbose=False)
        out.append(torch.cat([p.detach().reshape(-1) for p in m.parameters()]))
    assert torch.equal(out[0], out[1])
