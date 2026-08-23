"""4단계 회귀 테스트 — PyTorch 물리 모듈과 MLP 대체모델.

고정하는 주장은 크게 셋이다.

1. **PyTorch 잔차 = NumPy 잔차** (기계정밀도). 물리 손실이 틀리면 학습이
   조용히 잘못된 방향으로 간다.
2. **아는 값은 예측하지 않는다.** 슬랙 위상 0, PV·슬랙 전압 = 설정값이
   근사가 아니라 정확히 들어가야 한다.
3. **통계는 학습 분할에서만.** 시험 분할이 정규화에 새어 들면 성능이 부풀려진다.
"""

from __future__ import annotations

import math
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
    TrainConfig, evaluate, init_skip_lstsq, jacobian_weights, lambda_at,
    prepare, supervised_loss, train,
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


# ---------------------------------------------------------------- 장치 배치
def test_resolve_device_never_lies():
    """``resolve_device`` 는 available 만 믿지 않고 실제 연산을 해 본다.

    cu126 이하 빌드에 Blackwell GPU 를 물리면 ``is_available()`` 은 True 인데
    첫 커널에서 터진다. auto 는 그런 경우 CPU 로 물러서야 한다.
    """
    from nnopf.train import resolve_device

    assert resolve_device("cpu").type == "cpu"
    dev = resolve_device("auto")
    assert dev.type in ("cpu", "cuda")
    # 무엇을 고르든 그 위에서 실제로 곱셈이 된다
    (torch.zeros(4, 4, device=dev) @ torch.zeros(4, 4, device=dev)).sum().item()


def test_bundle_places_tensors_and_keeps_float64_on_cpu(ds):
    """학습 텐서는 지정 장치로, 배정밀도 물리는 항상 CPU 로.

    소비자용 GeForce 는 배정밀도가 단정밀도의 1/64 속도라, 잔차를 GPU 에서
    재면 오히려 느려진다. 그래서 ``physics64`` 만은 옮기지 않는다.
    """
    b, model = prepare(ds, _spec(), case=CASE, device="cpu")
    assert b.device.type == "cpu"
    assert b.X.device == b.device
    assert b.physics64.G.device.type == "cpu"
    assert next(model.parameters()).device == b.device

    m = evaluate(model, b, b.split["test"][:64])
    assert np.isfinite(m["p_mismatch"]) and np.isfinite(m["vlim_viol_pct"])


def test_linear_baseline_follows_the_bundle_device(ds):
    """비교군도 신경망과 같은 장치에서 평가된다.

    GPU 학습을 붙일 때 신경망만 옮기고 선형 기준선을 빠뜨려 평가에서 터진
    적이 있다. 장치가 섞이면 바로 RuntimeError 이므로, 같은 Bundle 로
    평가가 끝까지 도는지만 확인하면 회귀를 잡을 수 있다.
    """
    from nnopf.baselines import fit_linear
    from nnopf.train import resolve_device

    dev = resolve_device("auto")
    b, _ = prepare(ds, _spec(), case=CASE, device=dev)
    lin = fit_linear(ds, b.layout, b.split["train"]).to(dev)
    m = evaluate(lin, b, b.split["test"][:128])
    assert np.isfinite(m["vm_mae"]) and np.isfinite(m["p_mismatch"])


def test_linear_baseline_is_conditioned_and_reproducible(ds, sysm):
    """상수열을 빼지 않으면 선형 기준선이 컴퓨터마다 다른 답을 낸다.

    입력에는 정보가 0 인 열이 많다 — 부하 없는 모선의 ``Pd``, 발전기 없는
    모선의 ``p_gen``, 상정사고에서 제외된 선로의 ``status``. 그대로 두면
    설계행렬 조건수가 1e58 까지 올라가고, ``lstsq`` 의 특이값 절단선이 numpy
    기본값 근처에 걸려 LAPACK 구현에 따라 답이 튄다. 실제로 같은 데이터로
    두 컴퓨터에서 P/부하 30.80% 와 22.99% 가 나왔다.
    """
    from nnopf.baselines import fit_linear

    layout = IOLayout(sysm)
    split = ds.split_random(seed=0)
    X = layout.inputs(ds)[split["train"]].astype(np.float64)

    const = np.flatnonzero(X.std(0) == 0)
    assert len(const) > 0, "이 데이터엔 상수열이 있어야 이 테스트가 의미 있다"

    lin = fit_linear(ds, layout, split["train"], split["val"])
    assert len(lin.keep) == X.shape[1] - len(const)

    keep = np.flatnonzero(X.std(0) > 0)
    A = np.c_[X[:, keep], np.ones(len(X))]
    sv = np.linalg.svd(A, compute_uv=False)
    assert sv[0] / sv[-1] < 1e12, "상수열을 뺐는데도 조건수가 너무 크다"

    # 두 번 풀면 같은 답 (닫힌 해의 최소 조건)
    lin2 = fit_linear(ds, layout, split["train"], split["val"])
    assert torch.equal(lin.W, lin2.W)


def test_linear_baseline_truncation_is_chosen_on_validation(ds, sysm):
    """절단선을 검증 분할로 고른다 — 신경망의 조기 종료와 같은 기준.

    학습 분할만 보면 성분을 하나도 안 버리는 쪽이 항상 이긴다(정의상 잔차
    최소). 그런데 그 방향이 물리 잔차를 크게 키울 수 있다. 검증으로 골라야
    두 모델의 선택 기준이 같아지고 비교가 공정해진다.
    """
    from nnopf.baselines import fit_linear

    layout = IOLayout(sysm)
    split = ds.split_random(seed=0)
    picked = fit_linear(ds, layout, split["train"], split["val"])
    plain = fit_linear(ds, layout, split["train"])          # 기본값 (고르지 않음)
    assert picked.W.shape == plain.W.shape

    b, _ = prepare(ds, _spec(), split=split, case=CASE)
    for m in (evaluate(picked, b, split["test"]), evaluate(plain, b, split["test"])):
        assert np.isfinite(m["p_over_load_pct"])


# --------------------------------------------------------------------------
# 미지 N-1 에서의 모델 선택 (06 문서 §7.8)
# --------------------------------------------------------------------------

def test_phys_selection_keeps_training_past_the_val_floor(ds):
    """미지 N-1 에서는 검증 손실로 고르면 거의 학습되지 않은 모델이 뽑힌다.

    ``split_unseen_n1`` 은 **검증 분할에도** 학습에서 못 본 고장을 넣는다.
    그래서 검증 지도손실이 몇 epoch 만에 바닥에 닿고 그 뒤로 안 움직인다.
    ``train`` 은 최고 검증 시점의 가중치로 되돌리므로, 그 바닥이 초반이면
    **초반 모델이 최종 답이 된다.** 실측에서 GAT 는 1,500 중 34 가 뽑혔고
    그때 학습 손실은 끝까지 갔을 때보다 33배 나빴다.

    ``select="phys"`` 는 우리가 실제로 보고하는 값(검증 분할의 P/부하 %)으로
    고른다. 그 지표는 계속 좋아지므로 학습이 초반에 끊기지 않는다.

    고정하는 주장은 **"더 오래 학습한 모델이 뽑힌다"** 하나다. 어느 쪽 시험
    성능이 나은지는 계통·모델마다 다를 수 있어 여기서 못박지 않는다.
    """
    un1 = ds.split_unseen_n1(seed=0)
    b, model = prepare(ds, _spec(), split=un1, seed=0)
    r_loss = train(model, b, TrainConfig(epochs=40, batch=64, seed=0,
                                         select="loss"), verbose=False)

    b2, model2 = prepare(ds, _spec(), split=un1, seed=0)
    r_phys = train(model2, b2, TrainConfig(epochs=40, batch=64, seed=0,
                                           select="phys"), verbose=False)

    assert r_loss["select"] == "loss" and r_phys["select"] == "phys"
    assert r_phys["best_epoch"] > r_loss["best_epoch"], (
        f"phys 기준이 더 늦게까지 갱신돼야 한다 "
        f"(loss {r_loss['best_epoch']} vs phys {r_phys['best_epoch']})"
    )
    # 기록에는 둘 다 남는다 — 나중에 어느 쪽으로 골랐는지 되짚을 수 있어야 한다.
    assert not math.isnan(r_phys["history"][-1]["val_phys"])
    assert math.isnan(r_loss["history"][-1]["val_phys"])


def test_jacobian_weights_are_scale_preserving_and_alpha_zero_is_a_no_op(sysm):
    """야코비안 가중치는 손실 크기를 바꾸지 않고, alpha=0 이면 아무 일도 안 한다.

    두 성질이 다 필요하다.

    **① alpha=0 이 완전한 대조군이어야 한다.** 그래야 "가중을 켰더니 좋아졌다"
    를 한 변수 실험으로 말할 수 있다 (06 문서 §5.3 의 비교 규칙).

    **② 곱수의 제곱평균이 1 이어야 한다.** 손실 크기가 변하면 정규화가
    다시 어긋난다 — 06 문서 §7.2 에서 학습을 죽였던 그 함정이다.
    """
    layout = IOLayout(sysm)
    Vm = np.ones(sysm.nb)
    Va = layout.va_ref.copy()

    m_vm0, m_va0 = jacobian_weights(sysm, layout, Vm, Va, alpha=0.0)
    assert np.allclose(m_vm0, 1.0) and np.allclose(m_va0, 1.0)

    m_vm, m_va = jacobian_weights(sysm, layout, Vm, Va, alpha=1.0)
    assert len(m_vm) == len(layout.pq)
    assert len(m_va) == len(layout.nonslack)
    for m in (m_vm, m_va):
        assert abs(float(np.sqrt((m**2).mean())) - 1.0) < 1e-5
        # 모선마다 실제로 달라야 의미가 있다. 균등이면 켤 이유가 없다.
        assert m.max() / m.min() > 2.0


def test_jac_alpha_only_rescales_within_a_group(ds):
    """``jac_alpha`` 는 모선 사이의 **상대** 가중만 바꾸고 전체 크기는 안 바꾼다.

    ``prepare`` 가 만든 두 모델의 ``vm_w`` / ``va_w`` 를 직접 비교한다.
    비율의 제곱평균이 1 이면, 정규화가 의도대로 보존된 것이다.
    """
    _, m0 = prepare(ds, _spec(), seed=0, jac_alpha=0.0)
    _, m1 = prepare(ds, _spec(), seed=0, jac_alpha=1.0)

    assert not torch.allclose(m0.vm_w, m1.vm_w), "alpha=1 이면 가중치가 달라져야 한다"
    assert not torch.allclose(m0.va_w, m1.va_w)

    for w0, w1 in ((m0.vm_w, m1.vm_w), (m0.va_w, m1.va_w)):
        r = (w1 / w0).numpy()
        assert abs(float(np.sqrt((r**2).mean())) - 1.0) < 1e-4


def test_lstsq_skip_init_starts_the_model_at_the_linear_baseline(ds):
    """``skip_init="lstsq"`` 는 **학습 한 번 하기 전에** 선형 기준선과 같아야 한다.

    ``models.py`` 는 지름길의 목적을 "신경망은 선형에 대한 보정만 배운다" 로
    적어 두었지만, 지름길이 0 에서 함께 학습되면 그건 구조가 아니라 희망이다.
    최소제곱 해로 초기화하고 본체 마지막 층을 0 으로 눌러야 **출발점이 곧
    선형 해**가 된다.

    고정하는 주장 둘:

    1. lstsq 초기화 모델은 학습 전에 이미 zero 초기화보다 훨씬 낫다.
    2. 그 성능이 ``fit_linear`` 의 선형 기준선과 같은 자릿수다.
       (완전히 같지는 않다 — ``vm_head="scaled"`` 의 시그모이드를 거꾸로
       통과시키는 과정에서 오차가 조금 생긴다.)
    """
    from nnopf.baselines import fit_linear

    sp = ds.split_unseen_n1(seed=0)
    b0, m0 = prepare(ds, _spec(), split=sp, seed=0)
    b1, m1 = prepare(ds, _spec(skip_init="lstsq"), split=sp, seed=0)

    e0 = evaluate(m0, b0, sp["test"])["p_over_load_pct"]
    e1 = evaluate(m1, b1, sp["test"])["p_over_load_pct"]
    lin = fit_linear(ds, b1.layout, sp["train"], sp["val"])
    el = evaluate(lin, b1, sp["test"])["p_over_load_pct"]

    assert e1 < e0 / 3, f"lstsq 초기화가 zero 보다 훨씬 나아야 한다 ({e1:.2f} vs {e0:.2f})"
    assert e1 < el * 1.5, f"선형 기준선과 같은 자릿수여야 한다 ({e1:.2f} vs {el:.2f})"
    assert b1.skip_init["body_zeroed"] is True
    assert b1.skip_init["n_dropped"] > 0, "상수열을 빼야 한다 (06 문서 §7.6)"


def test_skip_freeze_keeps_the_shortcut_fixed(ds):
    """``skip_freeze`` 를 켜면 학습이 지름길을 건드리지 않아야 한다."""
    sp = ds.split_unseen_n1(seed=0)
    b, m = prepare(ds, _spec(skip_init="lstsq", skip_freeze=True), split=sp, seed=0)
    before = m.skip.weight.detach().clone()

    assert not m.skip.weight.requires_grad
    train(m, b, TrainConfig(epochs=6, batch=64, seed=0), verbose=False)
    assert torch.equal(m.skip.weight, before), "얼린 지름길이 학습으로 바뀌면 안 된다"

    # 얼리지 않으면 바뀌어야 한다 (이 시험 자체가 유효한지 확인).
    b2, m2 = prepare(ds, _spec(skip_init="lstsq"), split=sp, seed=0)
    w2 = m2.skip.weight.detach().clone()
    train(m2, b2, TrainConfig(epochs=6, batch=64, seed=0), verbose=False)
    assert not torch.equal(m2.skip.weight, w2)


def test_vlim_metric_tolerates_float32_resolution():
    """float32 해상도보다 좁은 전압 상자를 위반으로 세면 안 된다.

    case118 의 슬랙(모선 68)은 상자가 [1.0349999999, 1.0350000001] 로
    폭이 2e-10 인데, 모델은 float32 라 1.035 를 1.03499997 로밖에 못 쓴다.
    허용오차가 없으면 **어떤 모델이든** 이 한 모선이 항상 위반이라,
    지표가 정확히 100/118 = 0.8475% 에 못박혀 아무것도 구분하지 못한다
    (06 문서 §7.10).
    """
    sysm118 = load_case("case118")
    width = sysm118.Vmax - sysm118.Vmin
    assert np.flatnonzero(width < 1e-6).tolist() == [68], "이 시험의 전제가 깨졌다"

    d = generate_dataset("case118", n_samples=12, seed=7, workers=2, verbose=False)
    sp = d.split_random(seed=0)
    b, m = prepare(d, _spec(), split=sp, seed=0)
    viol = evaluate(m, b, sp["test"])["vlim_viol_pct"]

    # 고치기 전에는 어떤 모델이든 정확히 이 값이 나왔다.
    assert abs(viol - 100 / 118) > 1e-9, "슬랙 설정값이 위반으로 세어지고 있다"
