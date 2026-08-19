#!/usr/bin/env python
"""3단계 실행 스크립트: 조류계산 대체모델 학습 데이터 생성.

데이터 자체는 저장소에 넣지 않는다. **시드가 고정되어 있어 이 스크립트만 있으면
어느 컴퓨터에서든 같은 데이터가 몇 분 만에 재생성**되기 때문이다.
(case118 · 20,000표본 기준 4코어에서 약 2분)

사용법::

    # P2 논문과 동일한 설정으로 생성
    .venv/bin/python scripts/s02_generate_dataset.py --case case118 -n 20000

    # 여러 계통을 한 번에
    .venv/bin/python scripts/s02_generate_dataset.py --case case30 case118 -n 6000 20000

    # 다중 시드 (P2 는 2026~2030 다섯 번)
    .venv/bin/python scripts/s02_generate_dataset.py --case case30 -n 6000 --seeds 2026 2027 2028

    # 빠른 확인
    .venv/bin/python scripts/s02_generate_dataset.py --case case30 -n 500 --no-save
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nnopf.dataset import ScenarioConfig, generate_dataset  # noqa: E402


def report(ds) -> dict:
    """생성된 데이터셋의 품질 지표를 찍고 딕셔너리로 돌려준다."""
    div_rate = 100.0 * ds.n_diverged / max(ds.n_attempted, 1)
    n1 = int((ds.outage >= 0).sum())
    kinds = int(len(np.unique(ds.outage[ds.outage >= 0])))

    # 표본 하나 안에서의 값으로 봐야 의미가 있다. 전체 집합의 min~max 는
    # 수만 표본의 합집합이라 실제보다 훨씬 넓게 보여 오해를 부른다.
    vm_lo = ds.Vm.min(axis=1)
    spread = np.rad2deg(ds.Va.max(axis=1) - ds.Va.min(axis=1))

    print(f"    표본        {ds.n_samples:,} / 시도 {ds.n_attempted:,}")
    print(f"    발산        {ds.n_diverged:,} ({div_rate:.2f}%)")
    print(f"    N-1         {n1:,} ({100 * n1 / max(ds.n_samples, 1):.1f}%), 고장유형 {kinds}종")
    print(f"    최저 전압   평균 {vm_lo.mean():.4f} pu, P5 {np.percentile(vm_lo, 5):.4f}, "
          f"최저 {vm_lo.min():.4f} | 0.94 미만 표본 {100 * (vm_lo < 0.94).mean():.1f}%")
    print(f"    위상 스프레드 표본당 평균 {spread.mean():.2f} deg, P95 {np.percentile(spread, 95):.2f}, "
          f"최대 {spread.max():.2f}")
    print(f"    총부하      {ds.Pd.sum(1).min() * 100:.1f} ~ {ds.Pd.sum(1).max() * 100:.1f} MW")
    print(f"    재생E 출력  {ds.p_ren.sum(1).min() * 100:.1f} ~ {ds.p_ren.sum(1).max() * 100:.1f} MW "
          f"(부하 대비 평균 {100 * (ds.p_ren.sum(1) / ds.Pd.sum(1)).mean():.0f}%)")
    print(f"    생성 시간   {ds.gen_seconds:.1f}s ({ds.gen_seconds / max(ds.n_samples, 1) * 1000:.2f} ms/표본)")

    # 라벨이 진짜 조류해인지 표본 추출로 재확인한다.
    # 이 값이 크면 그 위에 올리는 학습이 전부 무의미해지므로 생성할 때마다 본다.
    from nnopf.case import load_case
    from nnopf.dataset import physics_residual

    sysm = load_case(ds.case)
    step = max(1, ds.n_samples // 200)
    worst = 0.0
    for i in range(0, ds.n_samples, step):
        dP, dQ = physics_residual(
            sysm, ds.Pd[i], ds.Qd[i], ds.p_ren[i], ds.p_gen[i],
            ds.Vm[i], ds.Va[i], outage=int(ds.outage[i]),
        )
        worst = max(worst, float(np.abs(dP).max()), float(np.abs(dQ).max()))
    print(f"    물리 검증   조류방정식 잔차 최대 {worst:.2e} pu "
          f"({'OK' if worst < 1e-7 else '!! 라벨 이상'})")

    # 미지 N-1 분할이 실제로 겹치지 않는지 확인 (조용히 실패하면 실험이 무의미해진다)
    sp = ds.split_unseen_n1()
    tr = set(np.unique(ds.outage[sp["train"]]).tolist()) - {-1}
    te = set(np.unique(ds.outage[sp["test"]]).tolist()) - {-1}
    overlap = len(tr & te)
    mark = "OK" if overlap == 0 else f"!! {overlap}종 겹침"
    print(f"    분할        무작위 {[len(v) for v in ds.split_random().values()]} | "
          f"미지N-1 {[len(v) for v in sp.values()]} ({mark})")

    return {
        "case": ds.case, "seed": ds.seed,
        "n_samples": ds.n_samples, "n_attempted": ds.n_attempted,
        "n_diverged": ds.n_diverged, "divergence_pct": round(div_rate, 4),
        "n1_samples": n1, "n1_kinds": kinds,
        "vm_min_mean": round(float(vm_lo.mean()), 5),
        "vm_min_worst": round(float(vm_lo.min()), 5),
        "va_spread_mean_deg": round(float(spread.mean()), 3),
        "max_pf_residual_pu": float(worst),
        "gen_seconds": round(ds.gen_seconds, 2),
        "unseen_n1_overlap": overlap,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="조류계산 대체모델 학습 데이터 생성")
    # 기본값은 P2(PI-GAT)가 쓴 계통 그대로. 결과를 논문 수치와 직접 비교할 수 있다.
    # case300 은 기저 케이스부터 이미 병적(전압 0.889 pu, 26개 모선이 하한 미달)이고
    # 급전가능 용량도 빠듯해 기본에서 뺐다. 필요하면 --case case300 으로 명시할 것.
    ap.add_argument("--case", nargs="+", default=["case30", "case118"])
    ap.add_argument("-n", "--n-samples", nargs="+", type=int, default=[6000, 20000],
                    help="계통별 표본 수. 하나만 주면 모든 계통에 같은 값을 쓴다.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="data", help="저장 디렉터리")
    ap.add_argument("--no-save", action="store_true", help="생성만 하고 저장하지 않음")
    ap.add_argument("--n1-ratio", type=float, default=0.25)
    args = ap.parse_args()

    counts = args.n_samples
    if len(counts) == 1:
        counts = counts * len(args.case)
    if len(counts) != len(args.case):
        ap.error(f"--case {len(args.case)}개에 -n {len(counts)}개가 왔습니다.")

    out_dir = pathlib.Path(args.out)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ScenarioConfig(n1_ratio=args.n1_ratio)
    print("=" * 72)
    print(" nnopf 3단계 : 학습 데이터 생성 (설정 근거 = P2 PI-GAT §4.1)")
    print("=" * 72)

    manifest = []
    for case, n in zip(args.case, counts):
        for seed in args.seeds:
            print(f"\n--- {case} / seed {seed} / {n:,}표본 ---")
            ds = generate_dataset(case, n_samples=n, config=cfg, seed=seed,
                                  workers=args.workers, verbose=False)
            info = report(ds)
            if not args.no_save:
                path = out_dir / f"{case}_seed{seed}_n{ds.n_samples}.npz"
                ds.save(str(path))
                size_mb = path.stat().st_size / 1e6
                info["path"] = str(path)
                info["size_mb"] = round(size_mb, 1)
                print(f"    저장        {path} ({size_mb:.1f} MB)")
            manifest.append(info)

    if not args.no_save:
        mpath = out_dir / "manifest.json"
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n요약 저장: {mpath}")

    print("\n" + "=" * 72)
    bad = [m for m in manifest if m["unseen_n1_overlap"] != 0]
    if bad:
        print(f" !! 미지 N-1 분할에 겹침이 있는 데이터셋 {len(bad)}개")
        return 1
    print(f" 완료: 데이터셋 {len(manifest)}개")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
