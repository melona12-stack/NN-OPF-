"""미지 N-1 결과를 토폴로지로 쪼개 본다 — 06 문서 §4.1.

전체 평균만 보면 "신경망도 미지 N-1에서 3배 나빠진다" 로 끝나는데, 정상
토폴로지 표본과 본 적 없는 고장 표본을 나눠 재면 이야기가 완전히 달라진다.
열화가 퍼져 있는 게 아니라 **후자에만 뭉쳐 있다.** 그게 M4 에서 그래프
신경망을 쓸 근거이자 목표 수치가 된다.

    .venv/bin/python scripts/s03b_topology_breakdown.py
"""
import sys, numpy as np, torch
sys.path.insert(0, "src")
from nnopf.dataset import PowerFlowDataset
from nnopf.models import SurrogateSpec, IOLayout
from nnopf.train import prepare
from nnopf.case import load_case

ds = PowerFlowDataset.load("data/case118_n20000_s2026.npz")
spec = SurrogateSpec(hidden=256, layers=4)
split = ds.split_unseen_n1(seed=0)
b, model = prepare(ds, spec, split=split, case="case118", seed=0)
ck = torch.load("results/case118_fix_un1_mlp.pt", map_location="cpu", weights_only=False)
sd = ck["state_dict"]; sd.setdefault("va_ref", model.va_ref)
model.load_state_dict(sd); model.eval()

te = torch.as_tensor(split["test"], dtype=torch.long)
ph = b.physics64
with torch.no_grad():
    Vm, Va = model(b.X[te])
dP, dQ = ph.residual(Vm.double(), Va.double(),
                     b.p_spec[te].double(), b.q_spec[te].double(), b.outage[te])
per = dP.abs().sum(-1).numpy()                    # 표본별 |ΔP| 합
load = b.p_spec[te].double().abs().sum(-1).numpy()
pct = per / load * 100
q = np.percentile(pct, [50, 90, 95, 99, 100])
print(f"표본별 P/부하[%]  중앙값 {q[0]:.3f} · 90% {q[1]:.3f} · 95% {q[2]:.3f} · 99% {q[3]:.3f} · 최대 {q[4]:.2f}")
print(f"평균 {pct.mean():.3f}%  —  상위 1% 표본이 전체 합의 {per[pct>=q[3]].sum()/per.sum()*100:.1f}% 를 차지")
print(f"상위 5% 제외하면 평균 {pct[pct < q[2]].mean():.3f}%")

out = b.outage[te].numpy()
n1 = out >= 0
print(f"\nN-1 표본 {n1.sum()} / 정상 표본 {(~n1).sum()}")
print(f"  정상 토폴로지 평균 {pct[~n1].mean():.3f}%   ·   미지 N-1 평균 {pct[n1].mean():.3f}%")
bad = pct >= q[3]
print(f"  상위 1% 표본 중 N-1 비율 {n1[bad].mean()*100:.1f}%")

# --- 선형 기준선도 같은 방식으로 쪼개 본다 ---
from nnopf.baselines import fit_linear
lin = fit_linear(ds, IOLayout(load_case("case118")), split["train"])
with torch.no_grad():
    Vl, Al = lin(b.X[te])
dPl, _ = ph.residual(Vl.double(), Al.double(),
                     b.p_spec[te].double(), b.q_spec[te].double(), b.outage[te])
pl = dPl.abs().sum(-1).numpy() / load * 100
print(f"\n선형  정상 토폴로지 {pl[~n1].mean():7.3f}%   ·   미지 N-1 {pl[n1].mean():7.3f}%")
print(f"MLP   정상 토폴로지 {pct[~n1].mean():7.3f}%   ·   미지 N-1 {pct[n1].mean():7.3f}%")
print(f"개선배수            {pl[~n1].mean()/pct[~n1].mean():7.1f}배  ·            {pl[n1].mean()/pct[n1].mean():7.1f}배")
