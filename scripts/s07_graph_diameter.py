r"""계통 그래프의 지름을 재고, 층 수가 몇이어야 하는지 정한다.

M4 에서 GAT 가 MLP 보다 나빴던 이유가 여기 있었다 (06 문서 §7.7).

메시지 전달 신경망은 **L 층이면 L-hop 까지만** 본다. 그런데 조류방정식은
:math:`V = Y_{bus}^{-1} S/V^*` 라, 한 모선의 전압이 **계통 전체**에 걸린다.
층 수가 그래프 지름보다 작으면, 모델은 물리적으로 볼 수 없는 것을 예측하라는
요구를 받는 셈이다. 학습으로 메울 수 있는 종류의 부족이 아니다.

그래서 층 수는 취향이 아니라 **계통이 정해 주는 값**이다.

    python scripts/s07_graph_diameter.py
    python scripts/s07_graph_diameter.py --case case118 --hops 4 8 12 16

선로 하나가 끊기면(N-1) 우회 경로가 길어져 지름이 늘어난다. ``--n1`` 을 주면
상정사고 각각에 대해 다시 재서 **최악의 지름**까지 본다. 미지 N-1 분할에서
필요한 층 수는 정상 지름이 아니라 이쪽이다.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnopf.case import load_case  # noqa: E402


def _adjacency(f_bus, t_bus, drop=None):
    """무향 인접 리스트. ``drop`` 은 빼고 볼 선로 인덱스."""
    n = int(max(f_bus.max(), t_bus.max())) + 1
    adj = [[] for _ in range(n)]
    for k, (a, b) in enumerate(zip(f_bus, t_bus)):
        if k == drop:
            continue
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _bfs(adj, s):
    d = [-1] * len(adj)
    d[s] = 0
    q = collections.deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d[v] < 0:
                d[v] = d[u] + 1
                q.append(v)
    return d


def diameter(adj):
    """(지름, 반지름, 평균 최단거리). 그래프가 갈라져 있으면 지름은 ``inf``."""
    n = len(adj)
    ecc, total, pairs = [], 0, 0
    for s in range(n):
        d = _bfs(adj, s)
        if min(d) < 0:                      # 도달 못 하는 모선이 있다
            return float("inf"), float("inf"), float("inf")
        ecc.append(max(d))
        total += sum(d)
        pairs += n
    return max(ecc), min(ecc), total / pairs


def hop_coverage(adj, k):
    """k-hop 안에 들어오는 모선 수의 평균 (자기 자신 포함)."""
    return float(np.mean([sum(1 for x in _bfs(adj, s) if 0 <= x <= k)
                          for s in range(len(adj))]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", nargs="+", default=["case30", "case118"])
    p.add_argument("--hops", type=int, nargs="+", default=[3, 4, 8, 12, 16])
    p.add_argument("--n1", action="store_true",
                   help="선로를 하나씩 끊어 보며 최악의 지름까지 잰다 (느리다)")
    a = p.parse_args()

    for name in a.case:
        sysm = load_case(name)
        n = sysm.nb
        adj = _adjacency(sysm.f_bus, sysm.t_bus)
        diam, rad, mean = diameter(adj)

        print(f"\n{name}: 모선 {n} · 선로 {len(sysm.f_bus)}")
        print(f"  지름 {diam} hop · 반지름 {rad} · 평균 최단거리 {mean:.2f}")
        print(f"  {'층 수':>6}  {'보이는 모선':>12}  {'비율':>7}")
        for k in a.hops:
            r = hop_coverage(adj, k)
            print(f"  {k:>6}  {r:>9.1f}/{n:<3}  {100 * r / n:>6.1f}%")
        print(f"  → 전 계통을 보려면 **{diam} 층 이상** 필요하다.")

        if not a.n1:
            continue

        worst, worst_line, split = diam, None, 0
        for k in range(len(sysm.f_bus)):
            d, _, _ = diameter(_adjacency(sysm.f_bus, sysm.t_bus, drop=k))
            if d == float("inf"):
                split += 1                  # 끊으면 계통이 갈라지는 선로
            elif d > worst:
                worst, worst_line = d, k
        print(f"  N-1 최악 지름 {worst} hop (선로 {worst_line} 고장 시), "
              f"계통이 갈라지는 선로 {split}개")
        print(f"  → 미지 N-1 까지 보려면 **{worst} 층 이상** 필요하다.")


if __name__ == "__main__":
    main()
