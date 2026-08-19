#!/usr/bin/env python
"""2단계 실행 스크립트: pandapower 대조 정합성 검증.

사용법::

    .venv/bin/python scripts/s01_validate_vs_pandapower.py
    .venv/bin/python scripts/s01_validate_vs_pandapower.py --cases case9 case30 --no-opf
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nnopf.compare import DEFAULT_CASES, compare_opf, compare_power_flow


def main() -> int:
    ap = argparse.ArgumentParser(description="직접 구현 PF/OPF 를 pandapower 와 비교")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--no-opf", action="store_true", help="조류계산만 검증")
    ap.add_argument("--line-limits", action="store_true", help="선로 조류 한계 적용")
    args = ap.parse_args()

    print("=" * 72)
    print(" nnopf 2단계 정합성 검증 : 직접 구현 vs pandapower")
    print("=" * 72)

    reports = []
    for case in args.cases:
        reports.append(compare_power_flow(case))
        print(reports[-1].format())
        if not args.no_opf:
            reports.append(compare_opf(case, enforce_line_limits=args.line_limits))
            print(reports[-1].format())

    n_fail = sum(1 for r in reports if not r.ok)
    print("=" * 72)
    print(f" 결과: {len(reports) - n_fail}/{len(reports)} 통과")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
