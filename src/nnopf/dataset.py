"""3단계: 조류계산 대체모델 학습용 데이터 생성기.

무엇을 만드는가
---------------
부하/발전 시나리오를 샘플링하고, **직접 구현한 뉴턴-랩슨**(2단계에서 pandapower와
기계정밀도 일치를 검증함)으로 정답 전압을 계산해 (입력, 라벨) 쌍을 만든다.

설계 근거
---------
샘플링 설정과 특징 구성은 배경 논문 P2 (Wen et al., *Energies* 2026, PI-GAT)
§3.1, §4.1 을 그대로 따른다. P2 가 우리와 **동일하게 pandapower `case30`/`case118`**
을 쓰므로, 같은 설정으로 만들면 우리 결과를 논문 수치와 직접 비교할 수 있다.
자세한 근거는 ``docs/04_paper_review.md`` §2.2, §2.5 참조.

재현성
------
* 표본 ``i`` 의 난수는 ``default_rng([seed, i])`` 로 **독립적으로** 만든다.
  따라서 워커 수나 청크 크기를 바꿔도 표본 ``i`` 는 항상 동일하다.
* 재생에너지 모선 지정, 상정사고 선택도 모두 시드에서 결정된다.
* 그래서 **데이터 파일을 옮길 필요가 없다.** 코드와 시드만 있으면 같은 데이터가
  몇 분 만에 재생성된다.

저장 형식
---------
특징 대부분은 시나리오마다 바뀌지 않는다(전압 설정값, 모선 종류, 선로 임피던스).
바뀌는 것만 표본별로 저장하고 나머지는 한 번만 저장한다.
:meth:`PowerFlowDataset.node_features` / :meth:`~PowerFlowDataset.edge_attr`
가 학습 시점에 P2 형식의 텐서로 조립한다.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from nnopf.case import PQ, PV, SLACK, PowerSystem, load_case
from nnopf.powerflow import solve_power_flow
from nnopf.ybus import make_ybus

__all__ = [
    "ScenarioConfig",
    "allocate_dispatch",
    "check_capacity_feasible",
    "PowerFlowDataset",
    "generate_dataset",
    "physics_residual",
    "renewable_generators",
    "valid_contingencies",
]

# 시드 파생용 도메인 상수 (표본 인덱스와 충돌하지 않도록 큰 값을 쓴다)
_SEED_RENEWABLE = 1_000_000_001
_SEED_CONTINGENCY = 1_000_000_002


# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioConfig:
    """시나리오 샘플링 설정. 기본값은 P2 §4.1 을 그대로 옮긴 것이다."""

    # --- 부하 배율 (모선별 독립) ---
    p_load_lo: float = 0.80
    p_load_hi: float = 1.20
    q_load_lo: float = 0.85
    q_load_hi: float = 1.15

    # --- 재생에너지 ---
    ren_bus_frac: float = 0.60
    """슬랙을 뺀 PV 모선 중 재생에너지로 지정할 비율."""
    ren_capacity_frac: float = 0.60
    """재생에너지 정격 합 / 기저 총부하."""
    ren_lo: float = 0.65
    ren_hi: float = 1.35
    """재생에너지 출력 변동 배율. 부하보다 훨씬 넓다."""
    ren_share_cap: float = 0.95
    """재생에너지가 현재 총부하의 이 비율을 넘지 않도록 출력제한(curtailment)."""

    # --- 기존 급전가능 발전 ---
    conv_jitter_lo: float = 0.95
    conv_jitter_hi: float = 1.05
    """잔여 부하를 비례 배분한 뒤 얹는 급전 변동."""

    # --- 상정사고 ---
    n1_ratio: float = 0.25
    """전체 표본 중 N-1 상정사고 비율. 섬이 생기는 고장은 후보에서 제외된다."""

    # --- 조류계산 ---
    pf_tol: float = 1e-8
    pf_max_iter: int = 30

    def validate(self) -> None:
        """설정값이 물리적으로 말이 되는지 확인한다."""
        if not 0.0 <= self.n1_ratio <= 1.0:
            raise ValueError(f"n1_ratio 는 [0,1] 이어야 합니다: {self.n1_ratio}")
        if self.p_load_lo > self.p_load_hi or self.q_load_lo > self.q_load_hi:
            raise ValueError("부하 배율의 하한이 상한보다 큽니다.")
        if self.ren_lo > self.ren_hi:
            raise ValueError("재생에너지 배율의 하한이 상한보다 큽니다.")
        if not 0.0 <= self.ren_bus_frac <= 1.0:
            raise ValueError(f"ren_bus_frac 는 [0,1] 이어야 합니다: {self.ren_bus_frac}")


# --------------------------------------------------------------------------
# 계통 구조에서 파생되는 것들 (시드만 같으면 항상 동일)
# --------------------------------------------------------------------------


def renewable_generators(sys: PowerSystem, cfg: ScenarioConfig, seed: int) -> np.ndarray:
    """재생에너지로 지정할 **발전기 인덱스**를 고른다.

    P2 는 "슬랙을 제외한 PV 발전 모선의 60%" 를 재생에너지로 지정하고, 그 선택을
    데이터 생성 내내 고정한다. 여기서도 시드에서만 결정되므로 항상 재현된다.
    """
    on = sys.gen_status.astype(bool)
    pv_set = set(sys.pv.tolist())
    eligible = np.array(
        [g for g in range(sys.ng) if on[g] and int(sys.gen_bus[g]) in pv_set], dtype=int
    )
    n_ren = int(round(cfg.ren_bus_frac * len(eligible)))
    if n_ren == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng([seed, _SEED_RENEWABLE])
    return np.sort(rng.choice(eligible, size=n_ren, replace=False))


def valid_contingencies(sys: PowerSystem) -> np.ndarray:
    """N-1 후보 브랜치 인덱스. **섬(island)을 만드는 고장은 제외**한다.

    한 선로를 빼도 계통이 하나로 연결되어 있어야 조류계산이 성립한다.
    연결 성분 개수로 판정한다.
    """
    base_on = sys.br_status.astype(bool)
    ok = []
    for line in np.flatnonzero(base_on):
        mask = base_on.copy()
        mask[line] = False
        adj = sp.coo_matrix(
            (np.ones(int(mask.sum())), (sys.f_bus[mask], sys.t_bus[mask])),
            shape=(sys.nb, sys.nb),
        )
        n_comp, _ = connected_components(adj, directed=False)
        if n_comp == 1:
            ok.append(int(line))
    return np.array(ok, dtype=int)


def allocate_dispatch(
    target: float,
    pmin: np.ndarray,
    pmax: np.ndarray,
    weight: np.ndarray,
    jitter: np.ndarray,
    max_iter: int = 30,
) -> np.ndarray:
    r"""목표 발전량 ``target`` 을 출력 한계를 지키며 발전기에 배분한다.

    단순히 비례 배분 후 :func:`numpy.clip` 만 하면 **상·하한에 걸린 만큼이 전부
    슬랙 한 모선으로 떨어진다.** 부족분이 수천 MW 에 이르면 슬랙 모선 하나가
    그걸 전부 공급해야 하므로 조류계산이 발산하거나, 수렴해도 물리적으로
    말이 안 되는 운전점이 된다. (case300 에서 실제로 100% 발산을 관측)

    그래서 잘린 뒤 남은 부족/과잉을 **여유(headroom)에 비례해 재배분**한다.
    실제 급전이 하는 일과 같고, 몇 번만 돌면 수렴한다.

    Returns
    -------
    ndarray
        발전기별 출력. 총합은 ``target`` 에 최대한 가깝고, 용량이 모자라면
        전 발전기가 상한에 붙은 상태로 끝난다(이때 잔여는 슬랙이 흡수한다).
    """
    if pmax.size == 0:
        return np.array([])
    w = weight / weight.sum() if weight.sum() > 0 else np.full(pmax.size, 1.0 / pmax.size)
    alloc = np.clip(target * w * jitter, pmin, pmax)

    for _ in range(max_iter):
        gap = target - float(alloc.sum())
        if abs(gap) < 1e-10:
            break
        headroom = (pmax - alloc) if gap > 0 else (alloc - pmin)
        total = float(headroom.sum())
        if total < 1e-12:
            break  # 더 줄 곳도 뺄 곳도 없다 -> 잔여는 슬랙이 받는다
        alloc = np.clip(alloc + gap * headroom / total, pmin, pmax)
    return alloc


# --------------------------------------------------------------------------
# 시나리오 한 건
# --------------------------------------------------------------------------


@dataclass
class Scenario:
    """샘플링된 운전 시나리오 한 건 (조류계산 입력)."""

    Pd: np.ndarray       # (nb,) [pu]
    Qd: np.ndarray       # (nb,)
    p_ren: np.ndarray    # (nb,) 모선별 재생에너지 발전 [pu]
    p_gen: np.ndarray    # (nb,) 모선별 기존 발전 [pu] (슬랙은 0)
    Pg: np.ndarray       # (ng,) 발전기별 지령 [pu]
    outage: int          # 고장 브랜치 인덱스, 정상이면 -1


def sample_scenario(
    sys: PowerSystem,
    cfg: ScenarioConfig,
    index: int,
    seed: int,
    ren_gen: np.ndarray,
    ren_capacity: np.ndarray,
    contingencies: np.ndarray,
) -> Scenario:
    """표본 ``index`` 의 시나리오를 만든다.

    난수를 ``[seed, index]`` 에서 파생하므로 **워커 수·청크 크기와 무관하게**
    같은 index 는 항상 같은 시나리오가 된다.
    """
    rng = np.random.default_rng([seed, index])
    on = sys.gen_status.astype(bool)

    # ---- 부하 ---------------------------------------------------------
    Pd = sys.Pd * rng.uniform(cfg.p_load_lo, cfg.p_load_hi, sys.nb)
    Qd = sys.Qd * rng.uniform(cfg.q_load_lo, cfg.q_load_hi, sys.nb)
    total_load = float(Pd.sum())

    # ---- 재생에너지 ----------------------------------------------------
    Pg = np.zeros(sys.ng)
    if ren_gen.size:
        p_ren = ren_capacity * rng.uniform(cfg.ren_lo, cfg.ren_hi, ren_gen.size)
        # 출력제한: 재생에너지가 현재 총부하를 압도하지 않도록
        cap = cfg.ren_share_cap * total_load
        if p_ren.sum() > cap > 0:
            p_ren *= cap / p_ren.sum()
        Pg[ren_gen] = np.clip(p_ren, sys.Pmin[ren_gen], sys.Pmax[ren_gen])

    # ---- 기존 급전가능 발전 (슬랙 제외) --------------------------------
    slack = sys.slack
    conv = np.array(
        [
            g
            for g in range(sys.ng)
            if on[g] and g not in set(ren_gen.tolist()) and int(sys.gen_bus[g]) != slack
        ],
        dtype=int,
    )
    if conv.size:
        remaining = max(total_load - float(Pg[ren_gen].sum()), 0.0)
        Pg[conv] = allocate_dispatch(
            remaining,
            sys.Pmin[conv],
            sys.Pmax[conv],
            weight=sys.Pmax[conv],
            jitter=rng.uniform(cfg.conv_jitter_lo, cfg.conv_jitter_hi, conv.size),
        )
    # 슬랙 발전기는 손실과 남은 잔여분을 흡수하므로 지령을 주지 않는다.
    # 위의 재배분 덕분에 잔여는 보통 손실 수준(총부하의 수 %)에 그친다.

    # ---- 상정사고 ------------------------------------------------------
    outage = -1
    if contingencies.size and cfg.n1_ratio > 0:
        crng = np.random.default_rng([seed, _SEED_CONTINGENCY, index])
        if crng.random() < cfg.n1_ratio:
            outage = int(crng.choice(contingencies))

    # ---- 모선별 집계 (노드 특징용) --------------------------------------
    p_ren_bus = np.zeros(sys.nb)
    p_gen_bus = np.zeros(sys.nb)
    if ren_gen.size:
        np.add.at(p_ren_bus, sys.gen_bus[ren_gen], Pg[ren_gen])
    if conv.size:
        np.add.at(p_gen_bus, sys.gen_bus[conv], Pg[conv])

    # 입력을 float32 로 **저장하기 전에** float32 로 내림한 뒤 그 값으로 푼다.
    # 그러지 않으면 "저장된 입력"과 "라벨을 만든 입력"이 미세하게 달라져,
    # 나중에 물리 잔차를 재면 0 이 아니라 float32 정밀도만큼 남는다.
    # 그 오차는 Ybus 를 거치며 max|Y| 배로 증폭되어 case300 에서 2.5e-4 pu 까지
    # 커진다 (측정하려는 신경망 잔차와 같은 자릿수라 무시할 수 없다).
    f32 = lambda a: a.astype(np.float32).astype(np.float64)  # noqa: E731
    return Scenario(
        Pd=f32(Pd), Qd=f32(Qd), p_ren=f32(p_ren_bus), p_gen=f32(p_gen_bus),
        Pg=f32(Pg), outage=outage,
    )


def solve_scenario(
    sys: PowerSystem, cfg: ScenarioConfig, scn: Scenario, ybus_cache: dict
) -> tuple[np.ndarray, np.ndarray] | None:
    """시나리오를 뉴턴-랩슨으로 풀어 ``(Vm, Va)`` 를 돌려준다. 발산하면 ``None``.

    ``ybus_cache`` 는 상정사고별 :math:`Y_{bus}` 를 재사용하기 위한 사전이다.
    토폴로지가 같으면 부하가 바뀌어도 :math:`Y_{bus}` 는 그대로이므로,
    건당 약 10% 를 아낀다.
    """
    if scn.outage >= 0:
        status = sys.br_status.copy()
        status[scn.outage] = 0
        sys_scn = dataclasses.replace(sys, Pd=scn.Pd, Qd=scn.Qd, br_status=status)
    else:
        sys_scn = dataclasses.replace(sys, Pd=scn.Pd, Qd=scn.Qd)

    if scn.outage not in ybus_cache:
        ybus_cache[scn.outage] = make_ybus(sys_scn)

    try:
        res = solve_power_flow(
            sys_scn,
            Pg=scn.Pg,
            tol=cfg.pf_tol,
            max_iter=cfg.pf_max_iter,
            Ybus=ybus_cache[scn.outage],
        )
    except Exception:
        # 특이 야코비안 등으로 선형해가 실패하는 경우가 드물게 있다.
        return None
    if not res.converged or not np.all(np.isfinite(res.Vm)):
        return None
    return res.Vm, res.Va


# --------------------------------------------------------------------------
# 데이터셋 컨테이너
# --------------------------------------------------------------------------


@dataclass
class PowerFlowDataset:
    """생성된 데이터셋.

    표본마다 바뀌는 것만 배열로 들고 있고, 정적인 것(전압 설정값, 모선 종류,
    선로 파라미터)은 한 번만 저장한다. case118 · 20,000 표본 기준 약 56 MB.
    """

    case: str
    seed: int
    config: ScenarioConfig

    # --- 표본별 (N, nb) ---
    # 입력은 float32, 라벨(Vm/Va)만 float64 다. 전압 오차는 Ybus 를 거치며
    # max|Y| 배(case300 기준 2400배)로 증폭되므로 라벨에서 정밀도를 아끼면
    # 물리 잔차의 측정 바닥이 신경망 오차와 같은 자릿수까지 올라간다.
    Pd: np.ndarray                # float32
    Qd: np.ndarray                # float32
    p_ren: np.ndarray             # float32
    p_gen: np.ndarray             # float32
    Vm: np.ndarray                # float64 (라벨)
    Va: np.ndarray                # float64 (라벨)
    outage: np.ndarray            # (N,) int, -1 = 정상

    # --- 정적 ---
    v_set: np.ndarray             # (nb,) 모선별 전압 설정값 (발전기 없으면 1.0)
    bus_type: np.ndarray          # (nb,) SLACK/PV/PQ
    edge_index: np.ndarray        # (2, 2*nl) 양방향 메시지 패싱 엣지
    edge_attr_base: np.ndarray    # (2*nl, 7) r, x, sh, tap, status, g, b
    edge_line: np.ndarray         # (2*nl,) 각 방향 엣지가 어느 브랜치인지

    # --- 통계 ---
    n_attempted: int = 0
    n_diverged: int = 0
    gen_seconds: float = 0.0

    # ------------------------------------------------------------------ 크기
    @property
    def n_samples(self) -> int:
        return self.Vm.shape[0]

    @property
    def n_bus(self) -> int:
        return self.Vm.shape[1]

    # ------------------------------------------------------- 학습용 텐서 조립
    def node_features(self, idx: np.ndarray | slice | None = None) -> np.ndarray:
        """P2 §3.1 의 노드 특징 8개를 조립한다 ``(N, nb, 8)``.

        순서: ``P_load, Q_load, P_ren, P_gen, V_set, isPQ, isPV, isref``

        .. note::
           슬랙 모선의 ``P_gen`` 은 **0** 이다. 슬랙 출력은 조류방정식이 결정하는
           종속변수라 추론 시점에 알 수 없기 때문이다. 모델은 ``isref`` 원-핫으로
           그 모선이 슬랙임을 알고 스스로 추론해야 한다.
        """
        sl = slice(None) if idx is None else idx
        pd_, qd_ = self.Pd[sl], self.Qd[sl]
        n = pd_.shape[0]

        static = np.stack(
            [
                self.v_set,
                (self.bus_type == PQ).astype(np.float32),
                (self.bus_type == PV).astype(np.float32),
                (self.bus_type == SLACK).astype(np.float32),
            ],
            axis=-1,
        )  # (nb, 4)

        return np.concatenate(
            [
                np.stack([pd_, qd_, self.p_ren[sl], self.p_gen[sl]], axis=-1),
                np.broadcast_to(static, (n, self.n_bus, 4)),
            ],
            axis=-1,
        ).astype(np.float32)

    def edge_attr(self, idx: np.ndarray | slice | None = None) -> np.ndarray:
        """엣지 특징 7개를 조립한다 ``(N, E, 7)``.

        정적인 ``edge_attr_base`` 를 복제하고 **``status`` 열만** 상정사고에 맞춰
        0 으로 바꾼다. P2 처럼 고장 선로를 그래프에서 제거하지 않고 특징으로만
        표시하므로, 그래프 구조가 고정되어 배치 추론이 가능하다.
        """
        sl = slice(None) if idx is None else idx
        outage = np.atleast_1d(self.outage[sl])
        out = np.broadcast_to(
            self.edge_attr_base, (len(outage), *self.edge_attr_base.shape)
        ).copy()
        hit = np.flatnonzero(outage >= 0)
        for k in hit:
            out[k, self.edge_line == outage[k], 4] = 0.0
        return out.astype(np.float32)

    def targets(self, idx: np.ndarray | slice | None = None) -> np.ndarray:
        """라벨 ``(N, nb, 2)`` = ``[Vm, Va]``. 위상은 라디안."""
        sl = slice(None) if idx is None else idx
        return np.stack([self.Vm[sl], self.Va[sl]], axis=-1).astype(np.float32)

    # ------------------------------------------------------------------ 분할
    def split_random(
        self, ratios: tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = 0
    ) -> dict[str, np.ndarray]:
        """무작위 분할 (P2 의 표준 분할)."""
        rng = np.random.default_rng([seed, 7])
        perm = rng.permutation(self.n_samples)
        n_tr = int(ratios[0] * self.n_samples)
        n_va = int(ratios[1] * self.n_samples)
        return {
            "train": perm[:n_tr],
            "val": perm[n_tr : n_tr + n_va],
            "test": perm[n_tr + n_va :],
        }

    def split_unseen_n1(
        self, ratios: tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = 0
    ) -> dict[str, np.ndarray]:
        """**미지 N-1 분할** (P2 §4.7).

        고장 *유형* 을 먼저 나누고, 테스트용 고장은 학습·검증에 **단 한 번도**
        등장하지 않게 한다. 무작위 분할보다 훨씬 엄격한 일반화 시험이며,
        "본 적 없는 토폴로지에서도 되는가" 를 측정하는 유일한 방법이다.
        정상 표본은 같은 비율로 무작위 분할한다.
        """
        rng = np.random.default_rng([seed, 11])
        kinds = np.unique(self.outage[self.outage >= 0])
        rng.shuffle(kinds)
        n_tr = int(ratios[0] * len(kinds))
        n_va = int(ratios[1] * len(kinds))
        kind_split = {
            "train": set(kinds[:n_tr].tolist()),
            "val": set(kinds[n_tr : n_tr + n_va].tolist()),
            "test": set(kinds[n_tr + n_va :].tolist()),
        }

        normal = np.flatnonzero(self.outage < 0)
        perm = rng.permutation(normal)
        m_tr = int(ratios[0] * len(normal))
        m_va = int(ratios[1] * len(normal))
        normal_split = {
            "train": perm[:m_tr],
            "val": perm[m_tr : m_tr + m_va],
            "test": perm[m_tr + m_va :],
        }

        out = {}
        for name in ("train", "val", "test"):
            faulted = np.flatnonzero(
                (self.outage >= 0) & np.isin(self.outage, list(kind_split[name]))
            )
            out[name] = np.sort(np.concatenate([normal_split[name], faulted]))
        return out

    # ------------------------------------------------------------ 저장/불러오기
    def save(self, path: str) -> None:
        """압축 ``.npz`` 로 저장한다."""
        np.savez_compressed(
            path,
            case=self.case,
            seed=self.seed,
            config=np.array([dataclasses.asdict(self.config)], dtype=object),
            Pd=self.Pd, Qd=self.Qd, p_ren=self.p_ren, p_gen=self.p_gen,
            Vm=self.Vm, Va=self.Va, outage=self.outage,
            v_set=self.v_set, bus_type=self.bus_type,
            edge_index=self.edge_index, edge_attr_base=self.edge_attr_base,
            edge_line=self.edge_line,
            stats=np.array([self.n_attempted, self.n_diverged, self.gen_seconds]),
        )

    @classmethod
    def load(cls, path: str) -> "PowerFlowDataset":
        z = np.load(path, allow_pickle=True)
        stats = z["stats"]
        return cls(
            case=str(z["case"]),
            seed=int(z["seed"]),
            config=ScenarioConfig(**z["config"][0]),
            Pd=z["Pd"], Qd=z["Qd"], p_ren=z["p_ren"], p_gen=z["p_gen"],
            Vm=z["Vm"], Va=z["Va"], outage=z["outage"],
            v_set=z["v_set"], bus_type=z["bus_type"],
            edge_index=z["edge_index"], edge_attr_base=z["edge_attr_base"],
            edge_line=z["edge_line"],
            n_attempted=int(stats[0]), n_diverged=int(stats[1]), gen_seconds=float(stats[2]),
        )

    def summary(self) -> str:
        n1 = int((self.outage >= 0).sum())
        kinds = len(np.unique(self.outage[self.outage >= 0]))
        rate = 100.0 * self.n_diverged / max(self.n_attempted, 1)
        return (
            f"{self.case}: {self.n_samples}표본 x {self.n_bus}모선 | "
            f"N-1 {n1}건 ({100 * n1 / max(self.n_samples, 1):.0f}%, 고장유형 {kinds}종) | "
            f"발산 {self.n_diverged}/{self.n_attempted} ({rate:.2f}%) | "
            f"생성 {self.gen_seconds:.1f}s"
        )


# --------------------------------------------------------------------------
# 엣지 구성
# --------------------------------------------------------------------------


def build_edges(sys: PowerSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """양방향 메시지 패싱 엣지를 만든다 ``(edge_index, edge_attr_base, edge_line)``.

    물리 선로는 무향이지만 메시지 패싱은 방향이 있어야 하므로 브랜치마다
    ``f->t`` 와 ``t->f`` 두 엣지를 둔다.

    엣지 특징 7개 (P2 §3.1): ``r, x, sh, tap, status, g, b``.
    임피던스 ``r,x`` 와 그로부터 유도되는 어드미턴스 ``g,b`` 를 **둘 다** 넣는다.
    비선형 변환을 모델이 따로 학습하지 않아도 되게 하려는 것이다.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 1.0 / (sys.br_r + 1j * sys.br_x)
    g, b = np.nan_to_num(np.real(y)), np.nan_to_num(np.imag(y))

    attr = np.stack(
        [sys.br_r, sys.br_x, sys.br_b, sys.br_tap, sys.br_status.astype(float), g, b],
        axis=-1,
    )  # (nl, 7)

    edge_index = np.concatenate(
        [np.stack([sys.f_bus, sys.t_bus]), np.stack([sys.t_bus, sys.f_bus])], axis=1
    )
    edge_attr_base = np.concatenate([attr, attr], axis=0)
    edge_line = np.concatenate([np.arange(sys.nl), np.arange(sys.nl)])
    return edge_index.astype(np.int64), edge_attr_base.astype(np.float32), edge_line


def physics_residual(
    sys: PowerSystem,
    Pd: np.ndarray,
    Qd: np.ndarray,
    p_ren: np.ndarray,
    p_gen: np.ndarray,
    Vm: np.ndarray,
    Va: np.ndarray,
    outage: int = -1,
    Ybus=None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""조류방정식 잔차 :math:`(\Delta P, \Delta Q)` 를 **모선 종류를 구분해** 계산한다.

    이것이 곧 P2 §3.3 의 물리정보 손실 항이다. 여기서는 정답 라벨을 넣어
    "잔차가 0 인가" 를 확인하는 데 쓰지만, 4단계에서 ``Vm, Va`` 자리에 신경망
    예측을 넣으면 그대로 학습 손실이 된다.

    핵심은 **지정값이 있는 곳에만 잔차를 건다**는 것이다.

    =======  =========================  ==================================
    모선     잔차에 포함                 이유
    =======  =========================  ==================================
    PQ       :math:`\Delta P, \Delta Q`  P, Q 모두 지정값
    PV       :math:`\Delta P` 만         Q 는 조류방정식이 정하는 종속변수
    슬랙     없음                        P, Q 모두 종속변수
    =======  =========================  ==================================

    이 구분을 놓치고 모든 모선에 잔차를 걸면 학습이 잘못된 방향으로 간다.

    Returns
    -------
    (dP, dQ)
        각각 ``(nb,)``. 잔차를 걸지 않는 자리는 ``0`` 이다.

    Notes
    -----
    고정 병렬 소자(커패시터/리액터)는 지정 주입 :math:`Q^{sp}` 에 넣지 **않는다**.
    :math:`Y_{bus}` 대각에 이미 들어 있어 계산된 :math:`\hat Q` 쪽에 반영되기
    때문이며, 양쪽에 넣으면 이중계산이 된다. (2단계 검증에서 겪은 그 이슈)
    """
    if outage >= 0:
        status = sys.br_status.copy()
        status[outage] = 0
        sys = dataclasses.replace(sys, br_status=status)
    if Ybus is None:
        Ybus = make_ybus(sys)

    V = np.asarray(Vm, float) * np.exp(1j * np.asarray(Va, float))
    S_calc = V * np.conj(Ybus @ V)

    P_sp = np.asarray(p_gen, float) + np.asarray(p_ren, float) - np.asarray(Pd, float)
    Q_sp = -np.asarray(Qd, float)

    dP = np.zeros(sys.nb)
    dQ = np.zeros(sys.nb)
    non_slack = sys.bus_type != SLACK
    pq = sys.bus_type == PQ

    dP[non_slack] = P_sp[non_slack] - np.real(S_calc)[non_slack]
    dQ[pq] = Q_sp[pq] - np.imag(S_calc)[pq]
    return dP, dQ


def bus_voltage_setpoints(sys: PowerSystem) -> np.ndarray:
    """모선별 전압 설정값 ``(nb,)``. 발전기가 없는 모선은 1.0."""
    v = np.ones(sys.nb)
    on = sys.gen_status.astype(bool)
    v[sys.gen_bus[on]] = sys.Vg[on]
    v[sys.slack] = sys.Vm0[sys.slack]
    return v


# --------------------------------------------------------------------------
# 병렬 생성
# --------------------------------------------------------------------------

_W: dict = {}


def _worker_init(case: str, cfg: ScenarioConfig, seed: int) -> None:
    """워커 프로세스마다 계통을 한 번만 읽는다."""
    sysm = load_case(case)
    ren_gen = renewable_generators(sysm, cfg, seed)
    ren_capacity = _renewable_capacity(sysm, cfg, ren_gen)
    _W.update(
        sys=sysm,
        cfg=cfg,
        seed=seed,
        ren_gen=ren_gen,
        ren_capacity=ren_capacity,
        contingencies=valid_contingencies(sysm) if cfg.n1_ratio > 0 else np.array([], int),
        ybus_cache={},
    )


def check_capacity_feasible(
    sys: PowerSystem, cfg: ScenarioConfig, seed: int
) -> None:
    """설정이 **발전 용량 측면에서 성립하는지** 미리 확인한다.

    발전기를 재생에너지로 지정하면 그 발전기의 급전가능 용량이 확률적 재생에너지
    출력으로 **대체**된다. 따라서 ``ren_bus_frac`` 은 크게 두고 ``ren_capacity_frac``
    만 낮추면, 지정된 만큼의 급전 용량이 계통에서 사라진다.

    부족분은 전부 슬랙 한 모선으로 몰리는데, 슬랙 용량을 넘어서면 조류계산이
    통째로 발산한다. 실제로 case300 에서 ``ren_capacity_frac=0.15`` 로 두면
    기존 발전기 27대가 전부 상한에 붙고 슬랙이 5,395 MW 를 떠안아야 하는데
    슬랙 상한은 2,399 MW 라 **표본 100% 가 발산**한다.

    조용히 발산 데이터를 쏟아내는 대신 여기서 원인과 함께 멈춘다.

    Raises
    ------
    ValueError
        최악 부하에서 발전 용량이 모자랄 때. 어떤 설정을 바꿔야 하는지 알려 준다.
    """
    on = sys.gen_status.astype(bool)
    slack = sys.slack
    ren = renewable_generators(sys, cfg, seed)
    ren_set = set(ren.tolist())
    conv = np.array(
        [g for g in range(sys.ng) if on[g] and g not in ren_set and int(sys.gen_bus[g]) != slack],
        dtype=int,
    )
    slack_gen = np.array(
        [g for g in range(sys.ng) if on[g] and int(sys.gen_bus[g]) == slack], dtype=int
    )

    import warnings

    base = sys.base_mva
    ren_cap = float(_renewable_capacity(sys, cfg, ren).sum())
    dispatchable = float(sys.Pmax[conv].sum()) + float(sys.Pmax[slack_gen].sum())
    nominal_load = float(sys.Pd.sum())

    def mw(x: float) -> str:
        return f"{x * base:,.0f} MW"

    detail = (
        f"  재생E {ren.size}대 정격 {mw(ren_cap)} + 기존 {conv.size}대 상한 "
        f"{mw(float(sys.Pmax[conv].sum()))} + 슬랙 상한 {mw(float(sys.Pmax[slack_gen].sum()))}\n"
        f"  -> ren_capacity_frac({cfg.ren_capacity_frac})을 올리거나, "
        f"ren_bus_frac({cfg.ren_bus_frac})을 낮춰 급전가능 발전기를 남기거나, "
        f"p_load_hi({cfg.p_load_hi})를 낮추세요."
    )

    # (1) 확실히 못 쓰는 설정 -- 기저 부하·평균 재생출력에서도 용량이 모자란다.
    #     이러면 사실상 전 표본이 발산하므로 데이터를 만들 이유가 없다.
    ren_mean = ren_cap * 0.5 * (cfg.ren_lo + cfg.ren_hi)
    if ren_mean + dispatchable < nominal_load:
        raise ValueError(
            f"{sys.name}: 발전 용량이 부족해 사실상 전 표본이 발산합니다 "
            f"(기저 부하 {mw(nominal_load)} 대비 "
            f"{mw(nominal_load - ren_mean - dispatchable)} 모자람).\n" + detail
        )

    # (2) 빠듯한 설정 -- 최악 조합(최대 부하 x 최소 재생출력)에서만 모자란다.
    #     대부분 수렴하지만 일부 표본이 발산하므로 알려만 준다.
    peak_load = nominal_load * cfg.p_load_hi
    ren_min = ren_cap * cfg.ren_lo
    if ren_min + dispatchable < peak_load:
        warnings.warn(
            f"{sys.name}: 최악 조합(최대 부하 {mw(peak_load)} x 최소 재생출력)에서 "
            f"{mw(peak_load - ren_min - dispatchable)} 모자랍니다. "
            f"일부 표본이 발산할 수 있습니다.\n" + detail,
            stacklevel=2,
        )


def _renewable_capacity(
    sys: PowerSystem, cfg: ScenarioConfig, ren_gen: np.ndarray
) -> np.ndarray:
    """재생에너지 발전기별 정격 ``(len(ren_gen),)`` [pu].

    P2: 재생에너지 정격 **합** 을 기저 총부하의 60% 로 두고, 발전기 용량에
    비례해 나눠 준다.
    """
    if ren_gen.size == 0:
        return np.array([])
    total = cfg.ren_capacity_frac * float(sys.Pd.sum())
    w = sys.Pmax[ren_gen]
    w = w / w.sum() if w.sum() > 0 else np.full(ren_gen.size, 1.0 / ren_gen.size)
    return total * w


def _worker_run(indices: np.ndarray) -> tuple:
    """표본 인덱스 묶음을 처리한다. 발산한 표본은 버리고 개수만 센다."""
    sysm, cfg = _W["sys"], _W["cfg"]
    rows = []
    diverged = 0
    for i in indices:
        scn = sample_scenario(
            sysm, cfg, int(i), _W["seed"], _W["ren_gen"], _W["ren_capacity"],
            _W["contingencies"],
        )
        sol = solve_scenario(sysm, cfg, scn, _W["ybus_cache"])
        if sol is None:
            diverged += 1
            continue
        Vm, Va = sol
        rows.append((scn.Pd, scn.Qd, scn.p_ren, scn.p_gen, Vm, Va, scn.outage))
    if not rows:
        return None, len(indices), diverged
    # 입력은 float32 (이미 float32 로 내림된 값이라 손실 없음).
    # 라벨(전압)만 float64 로 둔다 — Ybus 를 거치면 오차가 max|Y| 배로 증폭되므로
    # 여기서 정밀도를 아끼면 물리 잔차의 측정 바닥이 올라간다.
    inputs = tuple(np.stack([r[k] for r in rows]).astype(np.float32) for k in range(4))
    labels = tuple(np.stack([r[k] for r in rows]).astype(np.float64) for k in (4, 5))
    outage = np.array([r[6] for r in rows], dtype=np.int32)
    return (*inputs, *labels, outage), len(indices), diverged


def generate_dataset(
    case: str = "case118",
    n_samples: int = 20_000,
    config: ScenarioConfig | None = None,
    seed: int = 2026,
    workers: int | None = None,
    chunk: int = 250,
    verbose: bool = True,
) -> PowerFlowDataset:
    """데이터셋을 만든다.

    Parameters
    ----------
    case
        pandapower 시험계통 이름 (``case30``, ``case118``, ``case300`` 등).
    n_samples
        **시도할** 시나리오 수. 발산분이 빠지므로 결과는 이보다 적을 수 있다.
    seed
        모든 난수의 출발점. 같은 시드면 같은 데이터가 나온다.
    workers
        병렬 프로세스 수. ``None`` 이면 CPU 수. 조류계산은 순수 CPU 작업이라
        GPU 는 쓰이지 않는다.

    Returns
    -------
    PowerFlowDataset
    """
    from multiprocessing import Pool, cpu_count

    cfg = config or ScenarioConfig()
    cfg.validate()
    workers = workers or cpu_count()

    sysm = load_case(case)
    # 용량이 모자란 설정이면 여기서 원인과 함께 멈춘다.
    # (그러지 않으면 표본 100% 발산 데이터를 조용히 만들어 낸다)
    check_capacity_feasible(sysm, cfg, seed)
    edge_index, edge_attr_base, edge_line = build_edges(sysm)
    ren_gen = renewable_generators(sysm, cfg, seed)

    if verbose:
        cont = valid_contingencies(sysm) if cfg.n1_ratio > 0 else np.array([], int)
        print(f"[{case}] {sysm.summary()}")
        print(
            f"  재생에너지 발전기 {ren_gen.size}/{sysm.ng} "
            f"(정격 합 {cfg.ren_capacity_frac * sysm.Pd.sum() * sysm.base_mva:.0f} MW) | "
            f"N-1 후보 {cont.size}/{sysm.nl} (섬 유발 {sysm.nl - cont.size}개 제외)"
        )
        print(f"  {n_samples}표본 생성, 워커 {workers}개 ...")

    chunks = [np.arange(i, min(i + chunk, n_samples)) for i in range(0, n_samples, chunk)]
    t0 = time.perf_counter()
    with Pool(workers, initializer=_worker_init, initargs=(case, cfg, seed)) as pool:
        results = pool.map(_worker_run, chunks)
    dt = time.perf_counter() - t0

    parts = [r[0] for r in results if r[0] is not None]
    if not parts:
        raise RuntimeError(f"{case}: 수렴한 표본이 하나도 없습니다. 설정을 확인하세요.")
    attempted = sum(r[1] for r in results)
    diverged = sum(r[2] for r in results)

    def cat(k: int) -> np.ndarray:
        return np.concatenate([p[k] for p in parts], axis=0)

    ds = PowerFlowDataset(
        case=case,
        seed=seed,
        config=cfg,
        Pd=cat(0), Qd=cat(1), p_ren=cat(2), p_gen=cat(3), Vm=cat(4), Va=cat(5),
        outage=np.concatenate([p[6] for p in parts], axis=0),
        v_set=bus_voltage_setpoints(sysm).astype(np.float32),
        bus_type=sysm.bus_type.astype(np.int32),
        edge_index=edge_index,
        edge_attr_base=edge_attr_base,
        edge_line=edge_line,
        n_attempted=attempted,
        n_diverged=diverged,
        gen_seconds=dt,
    )
    if verbose:
        print(f"  {ds.summary()}")
    return ds
