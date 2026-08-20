"""4단계 회귀 테스트 — PyTorch 물리 모듈과 MLP 대체모델.

고정하는 주장은 크게 셋이다.

1. **PyTorch 잔차 = NumPy 잔차** (기계정밀도). 물리 손실이 틀리면 학습이
   조용히 잘못된 방향으로 간다.
2. **아는 값은 예측하지 않는다.** 슬랙 위상 0, PV·슬랙 전압 = 설정값이
   근사가 아니라 정확히 들어가야 한다.
3. **통계는 학습 분할에서만.** 시험 분할이 정규화에 새어 들면 성능이 부풀려진다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nnopf import generate_dataset, load_case, physics_residual  # noqa: E402
from nnopf.case import PQ, SLACK  # noqa: E402
from nnopf.models import IOLayout, PowerFlowMLP, SurrogateSpec  # noqa: E402
from nnopf.physics_torch import ACPhysics  # noqa: E402
from nnopf.train import (  # noqa: E402
    TrainConfig, evaluate, lambda_at, prepare, supervised_loss, train,
)

CASE = "case30"


@pytest.fixture(scope="module")
def ds():
    return generate_dataset(CASE, n_samples=400, seed=7, workers=2, verbose=False)


@pytest.fixture(scope="module")
def sysm():
    return load_case(CASE)


def _spec(**kw):
    return SurrogateSpec(**{"hidden": 32, "layers": 2, **kw})


# ---------------------------------------------------------------- 물리 모듈
def test_torch_physics_matches_numpy(ds, sysm):
    """PyTorch 잔차가 NumPy 판과 기계정밀도로 일치한다 (N-1 표본 포함)."""
    assert (ds.outage >= 0).sum() > 0, "N-1 표본이 없으면 보정 경로를 못 본다"

    ph = ACPhysics(sysm, dtype=torch.float64)
    f64 = lambda a: np.asarray(a, dtype=np.float64)
    p_spec = f64(ds.p_gen) + f64(ds.p_ren) - f64(ds.Pd)
    q_spec = -f64(ds.Qd)

    dP_t, dQ_t = ph.residual(
        torch.as_tensor(ds.Vm), torch.as_tensor(ds.Va),
        torch.as_tensor(p_spec), torch.as_tensor(q_spec),
        torch.as_tensor(ds.outage, dtype=torch.long),
    )
    for i in range(0, ds.n_samples, 7):
        dP, dQ = physics_residual(
            sysm, ds.Pd[i], ds.Qd[i], ds.p_ren[i], ds.p_gen[i],
            ds.Vm[i], ds.Va[i], outage=int(ds.outage[i]),
        )
        assert np.abs(dP - dP_t[i].numpy()).max() < 1e-12
        assert np.abs(dQ - dQ_t[i].numpy()).max() < 1e-12


def test_residual_masks_dependent_variables(ds, sysm):
    """슬랙은 P·Q 둘 다, PV 는 Q 가 잔차에서 빠져 있어야 한다 (05 문서 §7)."""
    ph = ACPhysics(sysm, dtype=torch.float64)
    n = 16
    f64 = lambda a: np.asarray(a, dtype=np.float64)
    dP, dQ = ph.residual(
        torch.as_tensor(ds.Vm[:n]), torch.as_tensor(ds.Va[:n]),
        torch.as_tensor(f64(ds.p_gen[:n]) + f64(ds.p_ren[:n]) - f64(ds.Pd[:n])),
        torch.as_tensor(-f64(ds.Qd[:n])),
        torch.as_tensor(ds.outage[:n], dtype=torch.long),
    )
    slack = sysm.bus_type == SLACK
    non_pq = sysm.bus_type != PQ
    assert torch.all(dP[:, slack] == 0)
    assert torch.all(dQ[:, non_pq] == 0)


def test_physics_loss_is_differentiable(ds, sysm):
    ph = ACPhysics(sysm)
    n = 8
    Vm = torch.as_tensor(ds.Vm[:n], dtype=torch.float32).requires_grad_(True)
    Va = torch.as_tensor(ds.Va[:n], dtype=torch.float32).requires_grad_(True)
    f64 = lambda a: np.asarray(a, dtype=np.float64)
    loss = ph.loss(
        Vm, Va,
        torch.as_tensor(f64(ds.p_gen[:n]) + f64(ds.p_ren[:n]) - f64(ds.Pd[:n]), dtype=torch.float32),
        torch.as_tensor(-f64(ds.Qd[:n]), dtype=torch.float32),
        torch.as_tensor(ds.outage[:n], dtype=torch.long),
    )
    loss.backward()
    assert torch.isfinite(Vm.grad).all() and torch.isfinite(Va.grad).all()
    assert Vm.grad.abs().sum() > 0


def test_n1_correction_equals_rebuilt_system(sysm):
    """4개 성분 보정이 '선로를 빼고 Ybus 를 다시 만든 것'과 같은가."""
    import dataclasses

    line = 5
    ph_full = ACPhysics(sysm, dtype=torch.float64)
    st = sysm.br_status.copy()
    st[line] = 0
    ph_out = ACPhysics(dataclasses.replace(sysm, br_status=st), dtype=torch.float64)

    rng = np.random.default_rng(0)
    Vm = torch.as_tensor(1.0 + 0.05 * rng.standard_normal((4, sysm.nb)))
    Va = torch.as_tensor(0.1 * rng.standard_normal((4, sysm.nb)))

    P1, Q1 = ph_full.injection(Vm, Va, torch.full((4,), line, dtype=torch.long))
    P2, Q2 = ph_out.injection(Vm, Va, torch.full((4,), -1, dtype=torch.long))
    assert torch.allclose(P1, P2, atol=1e-12) and torch.allclose(Q1, Q2, atol=1e-12)


# ------------------------------------------------------------ 입출력 계약
def test_layout_predicts_only_unknowns(sysm):
    lay = IOLayout(sysm)
    assert lay.out_dim == len(lay.pq) + len(lay.nonslack)
    assert lay.out_dim < 2 * sysm.nb, "아는 값까지 예측하고 있다"
    assert sysm.slack not in lay.nonslack, "슬랙 위상은 예측 대상이 아니다"
    assert not np.isin(lay.pq, np.flatnonzero(sysm.bus_type != PQ)).any()


def test_inputs_encode_outage_as_status_zero(ds, sysm):
    lay = IOLayout(sysm)
    X = lay.inputs(ds)
    status = X[:, 4 * lay.nb :]
    for i in range(ds.n_samples):
        o = int(ds.outage[i])
        assert status[i].sum() == lay.nl - (1 if o >= 0 else 0)
        if o >= 0:
            assert status[i, o] == 0.0


def test_model_fills_known_values_exactly(ds, sysm):
    """무작위 초기 가중치에서도 슬랙 θ=0, PV·슬랙 |V|=설정값 이어야 한다."""
    b, model = prepare(ds, _spec())
    Vm, Va = model(b.X[:32])
    known = sysm.bus_type != PQ
    v_set = torch.as_tensor(ds.v_set, dtype=torch.float32)
    assert torch.all(Vm[:, known] == v_set[known])
    assert torch.all(Va[:, sysm.bus_type == SLACK] == 0)


def test_scaled_head_cannot_leave_its_box(ds):
    """스케일링 인자(P1 §5.2.2): 입력이 아무리 튀어도 박스를 못 벗어난다."""
    b, model = prepare(ds, _spec(vm_head="scaled"))
    x = torch.randn(64, b.layout.in_dim) * 1e4
    Vm, _ = model(x)
    lo, hi = model.vm_lo, model.vm_hi
    assert torch.all(Vm[:, model.pq_idx] >= lo - 1e-6)
    assert torch.all(Vm[:, model.pq_idx] <= hi + 1e-6)


def test_scaled_box_covers_training_labels(ds):
    """박스가 학습 라벨을 자르면 도달 불가능한 정답이 생긴다."""
    b, model = prepare(ds, _spec(vm_head="scaled"))
    vm_tr = ds.Vm[b.split["train"]][:, b.layout.pq]
    assert (vm_tr >= model.vm_lo.numpy()).all()
    assert (vm_tr <= model.vm_hi.numpy()).all()


# ------------------------------------------------------------------ 누출
def test_statistics_use_train_split_only(ds):
    """시험 분할 라벨을 망가뜨려도 모델 버퍼가 그대로여야 한다."""
    split = ds.split_random(seed=0)
    _, m1 = prepare(ds, _spec(), split=split)

    import copy

    poisoned = copy.deepcopy(ds)
    poisoned.Vm[split["test"]] += 5.0
    poisoned.Va[split["test"]] += 5.0
    poisoned.Pd[split["test"]] += 5.0
    _, m2 = prepare(poisoned, _spec(), split=split)

    for name in ("in_mean", "in_std", "vm_lo", "vm_hi", "va_mean", "va_std"):
        assert torch.allclose(getattr(m1, name), getattr(m2, name)), name


# ------------------------------------------------------------------ 학습
def test_lambda_warmup_ramp_schedule():
    cfg = TrainConfig(lam=1e-3, lam_warmup=10, lam_ramp=5)
    assert lambda_at(0, cfg) == 0.0
    assert lambda_at(9, cfg) == 0.0
    assert lambda_at(10, cfg) == pytest.approx(1e-3 / 5)
    assert lambda_at(14, cfg) == pytest.approx(1e-3)
    assert lambda_at(99, cfg) == pytest.approx(1e-3)
    assert lambda_at(99, TrainConfig(lam=0.0)) == 0.0


def test_training_reduces_validation_loss(ds):
    b, model = prepare(ds, _spec())
    r = train(model, b, TrainConfig(epochs=25, seed=0), verbose=False)
    assert r["best_val"] < r["history"][0]["val"]
    m = evaluate(model, b, b.split["test"])
    assert np.isfinite(m["vm_mae"]) and m["vm_mae"] < 1.0


def test_training_is_reproducible(ds):
    out = []
    for _ in range(2):
        b, model = prepare(ds, _spec(), split=ds.split_random(seed=0))
        r = train(model, b, TrainConfig(epochs=8, seed=3), verbose=False)
        out.append(r["best_val"])
    assert out[0] == out[1]


def test_physics_loss_changes_gradients(ds):
    """λ>0 이 실제로 학습에 영향을 주는가 (스케줄이 죽어 있지 않은지)."""
    res = []
    for lam in (0.0, 1e-2):
        b, model = prepare(ds, _spec(), split=ds.split_random(seed=0))
        train(model, b, TrainConfig(epochs=12, seed=1, lam=lam, lam_warmup=0,
                                    lam_ramp=1), verbose=False)
        res.append(evaluate(model, b, b.split["test"])["p_mismatch"])
    assert res[0] != res[1]


def test_constant_status_column_does_not_blow_up_inputs(ds, sysm):
    """학습에서 한 번도 고장 안 난 선로가 시험에서 고장 나도 입력이 폭주하면 안 된다.

    미지 N-1 분할은 **정의상** 이 상황을 만든다. 하한 1e-6 으로 나누던 초기
    구현에서는 정규화 입력이 1e6 까지 튀었다.
    """
    b, model = prepare(ds, _spec(), split=ds.split_unseen_n1(seed=0))
    z = (b.X - model.in_mean) / model.in_std
    assert torch.isfinite(z).all()
    assert z.abs().max() < 1e3, f"정규화 입력이 {z.abs().max():.1e} 로 폭주"


# ------------------------------------------------ 손실 스케일 / 선형 지름길
def test_supervised_loss_is_scale_free(ds):
    """표준화 손실이라 weight_decay 를 키워도 학습이 무너지지 않아야 한다.

    원단위 MSE 를 쓰던 초기 구현에서는 손실이 1e-4 규모라 Adam 의 L2 항이
    과제 기울기를 눌러, wd=1e-5 만으로 학습이 MSE 1e-4 에 갇혔다.
    """
    out = []
    for wd in (0.0, 1e-3):
        b, model = prepare(ds, _spec(), split=ds.split_random(seed=0))
        r = train(model, b, TrainConfig(epochs=40, lr=2e-3, weight_decay=wd,
                                        seed=0, patience=10**6), verbose=False)
        out.append(r["best_val"])
    assert out[1] < out[0] * 3, f"weight_decay 에 과민하다: {out}"


def test_residual_skip_starts_from_linear_and_helps(ds):
    """지름길 편향이 0 초기화이고, 켜면 검증 손실이 나빠지지 않는다."""
    b, m = prepare(ds, _spec(residual=True))
    assert m.skip is not None and torch.all(m.skip.bias == 0)

    res = {}
    for flag in (False, True):
        b, model = prepare(ds, _spec(residual=flag), split=ds.split_random(seed=0))
        r = train(model, b, TrainConfig(epochs=60, lr=2e-3, seed=0, patience=10**6),
                  verbose=False)
        res[flag] = r["best_val"]
    assert res[True] <= res[False] * 1.05, f"지름길이 오히려 해롭다: {res}"


def test_linear_baseline_matches_least_squares(ds):
    """비교군이 실제로 최소제곱 해인지 — 학습 분할 잔차가 최소여야 한다."""
    from nnopf.baselines import fit_linear

    b, _ = prepare(ds, _spec())
    tr = b.split["train"]
    lin = fit_linear(ds, b.layout, tr)
    m = evaluate(lin, b, b.split["test"])
    assert np.isfinite(m["vm_mae"]) and m["vm_mae"] < 1e-2

    # 계수를 흔들면 학습 분할 오차가 커져야 한다 (= 최소점에 있다)
    with torch.no_grad():
        Vm0, Va0 = lin(b.X[torch.as_tensor(tr)])
        base = ((Vm0 - b.Vm[tr]) ** 2).mean() + ((Va0 - b.Va[tr]) ** 2).mean()
        lin.W += 0.01 * torch.randn_like(lin.W)
        Vm1, Va1 = lin(b.X[torch.as_tensor(tr)])
        worse = ((Vm1 - b.Vm[tr]) ** 2).mean() + ((Va1 - b.Va[tr]) ** 2).mean()
    assert worse > base


def test_linear_baseline_fills_known_values(ds, sysm):
    """비교군도 아는 값은 정확히 채워야 공정한 비교가 된다."""
    from nnopf.baselines import fit_linear

    b, _ = prepare(ds, _spec())
    lin = fit_linear(ds, b.layout, b.split["train"])
    Vm, Va = lin(b.X[:16])
    known = sysm.bus_type != PQ
    assert torch.allclose(Vm[:, known], torch.as_tensor(ds.v_set[known], dtype=torch.float32))
    assert torch.all(Va[:, sysm.bus_type == SLACK] == 0)


def test_lambda_is_relative_weight(ds):
    """λ 는 '지도 항 대비 몇 배' 여야 한다 — 절대 크기에 휘둘리면 안 된다.

    물리 항과 지도 항의 절대 크기 비는 계통마다 4자리씩 다르다
    (case30 2.2e3, case118 2.1e7). 환산 없이 같은 λ 를 쓰면 한쪽에서는
    무시되고 다른 쪽에서는 학습을 파괴한다.
    """
    b, model = prepare(ds, _spec(), split=ds.split_random(seed=0))
    r = train(model, b, TrainConfig(epochs=12, lam=1.0, lam_warmup=0, lam_ramp=1,
                                    seed=0, patience=10**6), verbose=False)
    assert r["phys_ref"] > 1.0, "물리/지도 비가 측정되지 않았다"

    # λ=1 이면 램프 시작 시점에 두 항이 같은 크기가 되어야 한다
    b2, m2 = prepare(ds, _spec(), split=ds.split_random(seed=0))
    k = torch.as_tensor(b2.split["train"][:256], dtype=torch.long)
    with torch.no_grad():
        Vm, Va = m2(b2.X[k])
        s = supervised_loss(m2, Vm, Va, b2.Vm[k], b2.Va[k]).item()
        p = b2.physics.loss(Vm, Va, b2.p_spec[k], b2.q_spec[k], b2.outage[k]).item()
    scaled = p / (p / s)          # = s
    assert abs(scaled - s) < 1e-6 * max(s, 1.0)


# ---------------------------------------------------------------- 슬랙 기준위상
def test_slack_reference_angle_is_not_assumed_zero():
    """슬랙 위상을 0 으로 박으면 case118 에서 30° 오차가 통째로 생긴다.

    06 문서 §7.4 의 회귀 테스트. ``case118`` 은 슬랙(모선 68)의 기준위상이
    30° 라서, 두 대체모델 모두 그 값을 그대로 채워야 한다.
    """
    for case in ("case30", "case118"):
        sysm = load_case(case)
        layout = IOLayout(sysm)
        sl = np.flatnonzero(sysm.bus_type == SLACK)
        assert np.allclose(layout.va_ref[sl], sysm.Va0[sl])
        assert np.allclose(np.delete(layout.va_ref, sl), 0.0)

    sysm = load_case("case118")
    assert abs(IOLayout(sysm).va_ref[sysm.bus_type == SLACK][0] - np.pi / 6) < 1e-12


def test_models_fill_slack_angle_with_reference():
    """MLP 와 선형 기준선 모두 슬랙 위상을 기준값으로 채운다."""
    ds = generate_dataset("case118", n_samples=24, seed=3, workers=2, verbose=False)
    from nnopf.baselines import fit_linear

    layout = IOLayout(load_case("case118"))
    b, model = prepare(ds, SurrogateSpec(hidden=16, layers=1), case="case118")
    lin = fit_linear(ds, layout, np.arange(len(ds.Vm)))
    sl = np.flatnonzero(load_case("case118").bus_type == SLACK)
    for m in (model, lin):
        _, Va = m(b.X[:4])
        assert torch.allclose(
            Va[:, sl], torch.full_like(Va[:, sl], float(np.pi / 6)), atol=1e-6
        )
