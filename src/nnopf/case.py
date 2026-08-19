"""계통 케이스 데이터 컨테이너와 로더.

내부 표준 형식은 MATPOWER/PYPOWER의 ``ppc`` 를 그대로 따르되,
연구 코드에서 쓰기 편하도록 numpy 배열 필드로 풀어 놓은 :class:`PowerSystem`
데이터클래스를 사용한다.

단위 규약
---------
* 전압은 모두 per-unit (pu)
* 전력은 모두 per-unit (pu, base = ``base_mva``)
* 각도는 모두 라디안 (rad)
* 발전 비용 계수만 예외적으로 MW 단위 (c2*P_MW^2 + c1*P_MW + c0)
  — 문헌/툴 관행이 MW 기준이라 그대로 유지하고, 필요할 때 내부에서 환산한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# MATPOWER 열 인덱스 (0-based). 원문 규격: https://matpower.org 의 caseformat
# --------------------------------------------------------------------------

# bus 행렬
BUS_I, BUS_TYPE, PD, QD, GS, BS = 0, 1, 2, 3, 4, 5
BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN = 6, 7, 8, 9, 10, 11, 12

# branch 행렬
F_BUS, T_BUS, BR_R, BR_X, BR_B = 0, 1, 2, 3, 4
RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS = 5, 6, 7, 8, 9, 10

# pandapower 확장 열 (MATPOWER 원본에는 없음).
# 변압기 철손(BR_G)과 비대칭 파라미터를 표현하기 위해 pandapower 가 추가한 것으로,
# 이를 무시하면 변압기가 있는 계통에서 Ybus 가 미세하게 어긋난다.
BR_R_ASYM, BR_X_ASYM, BR_G, BR_G_ASYM, BR_B_ASYM = 21, 22, 23, 24, 25

# gen 행렬
GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN = range(10)

# 모선 종류
PQ, PV, SLACK = 1, 2, 3

__all__ = ["PowerSystem", "load_case", "from_ppc", "from_pandapower", "PQ", "PV", "SLACK"]


@dataclass
class PowerSystem:
    """한 시점의 계통 상태를 담는 불변(관례상) 데이터 컨테이너."""

    name: str
    base_mva: float

    # --- 모선 (nb,) ---
    bus_type: np.ndarray
    Pd: np.ndarray       # 유효전력 부하 [pu]
    Qd: np.ndarray       # 무효전력 부하 [pu]
    Gs: np.ndarray       # 모선 병렬 컨덕턴스 [pu]
    Bs: np.ndarray       # 모선 병렬 서셉턴스 [pu]
    Vmax: np.ndarray
    Vmin: np.ndarray
    Vm0: np.ndarray      # 초기치/기준 전압 크기
    Va0: np.ndarray      # 초기치/기준 전압 위상 [rad]
    base_kv: np.ndarray

    # --- 선로/변압기 (nl,) ---
    f_bus: np.ndarray    # 송단 모선 인덱스 (0-based)
    t_bus: np.ndarray    # 수단 모선 인덱스 (0-based)
    br_r: np.ndarray     # 저항 [pu]
    br_x: np.ndarray     # 리액턴스 [pu]
    br_b: np.ndarray     # 총 충전 서셉턴스 [pu]
    br_tap: np.ndarray   # 변압기 탭비 (선로는 1.0)
    br_shift: np.ndarray  # 위상 이동 [rad]
    br_status: np.ndarray
    rate_a: np.ndarray   # 정상 조류 한계 [pu]  (0 이면 무제한)


    # --- 발전기 (ng,) ---
    gen_bus: np.ndarray
    Pg0: np.ndarray
    Qg0: np.ndarray
    Pmax: np.ndarray
    Pmin: np.ndarray
    Qmax: np.ndarray
    Qmin: np.ndarray
    Vg: np.ndarray
    gen_status: np.ndarray

    # --- 비용 (ng,) : c2*P_MW^2 + c1*P_MW + c0 ---
    cost_c2: np.ndarray
    cost_c1: np.ndarray
    cost_c0: np.ndarray

    # pandapower 확장: 변압기 철손 및 비대칭 파라미터 (없으면 모두 0)
    br_g: np.ndarray = None      # type: ignore[assignment]
    br_r_asym: np.ndarray = None  # type: ignore[assignment]
    br_x_asym: np.ndarray = None  # type: ignore[assignment]
    br_g_asym: np.ndarray = None  # type: ignore[assignment]
    br_b_asym: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self):
        """선택 필드가 비어 있으면 0 배열로 채운다."""
        for f in ("br_g", "br_r_asym", "br_x_asym", "br_g_asym", "br_b_asym"):
            if getattr(self, f) is None:
                setattr(self, f, np.zeros(len(self.f_bus)))

    # ---------------------------------------------------------------- 크기
    @property
    def nb(self) -> int:
        """모선 수."""
        return len(self.bus_type)

    @property
    def nl(self) -> int:
        """선로(브랜치) 수."""
        return len(self.f_bus)

    @property
    def ng(self) -> int:
        """발전기 수."""
        return len(self.gen_bus)

    # ------------------------------------------------------------ 모선 분류
    @property
    def slack(self) -> int:
        """슬랙 모선 인덱스. 여러 개면 첫 번째를 쓴다."""
        idx = np.flatnonzero(self.bus_type == SLACK)
        if idx.size == 0:
            raise ValueError(f"{self.name}: 슬랙 모선이 없습니다.")
        return int(idx[0])

    @property
    def pv(self) -> np.ndarray:
        """PV 모선 인덱스 배열."""
        return np.flatnonzero(self.bus_type == PV)

    @property
    def pq(self) -> np.ndarray:
        """PQ 모선 인덱스 배열."""
        return np.flatnonzero(self.bus_type == PQ)

    @property
    def pvpq(self) -> np.ndarray:
        """PV + PQ 모선 인덱스 (= 슬랙을 뺀 전부)."""
        return np.sort(np.concatenate([self.pv, self.pq]))

    # ---------------------------------------------------------------- 행렬
    @property
    def Cg(self) -> np.ndarray:
        """발전기 -> 모선 결합 행렬 ``(nb, ng)``.

        같은 모선에 발전기가 여러 대 붙을 수 있으므로 단순 인덱싱 대신
        결합 행렬을 쓴다. ``Cg @ Pg`` 가 모선별 총 발전량이 된다.
        """
        C = np.zeros((self.nb, self.ng))
        C[self.gen_bus, np.arange(self.ng)] = 1.0
        return C

    def ybus(self):
        """계통 어드미턴스 행렬 ``Ybus`` (희소, ``(nb, nb)`` 복소)."""
        from nnopf.ybus import make_ybus

        return make_ybus(self)

    # ------------------------------------------------------------ 편의 함수
    def sbus_demand(self) -> np.ndarray:
        """모선별 부하 복소전력 ``(nb,)`` [pu]."""
        return self.Pd + 1j * self.Qd

    def flat_start(self) -> tuple[np.ndarray, np.ndarray]:
        """평기동(flat start) 초기치 ``(Vm, Va)``.

        슬랙/PV 모선은 지정 전압을, 나머지는 1.0 pu / 0 rad 를 쓴다.
        """
        Vm = np.ones(self.nb)
        Va = np.zeros(self.nb)
        on = self.gen_status.astype(bool)
        Vm[self.gen_bus[on]] = self.Vg[on]
        Vm[self.slack] = self.Vm0[self.slack]
        Va[self.slack] = self.Va0[self.slack]
        return Vm, Va

    def summary(self) -> str:
        """사람이 읽기 위한 한 줄 요약."""
        return (
            f"{self.name}: bus={self.nb} (slack=1, PV={len(self.pv)}, PQ={len(self.pq)}), "
            f"branch={self.nl}, gen={self.ng}, "
            f"load={self.Pd.sum() * self.base_mva:.1f} MW / "
            f"{self.Qd.sum() * self.base_mva:.1f} Mvar"
        )


# --------------------------------------------------------------------------
# 로더
# --------------------------------------------------------------------------


def from_ppc(ppc: dict[str, Any], name: str = "case") -> PowerSystem:
    """MATPOWER/PYPOWER ``ppc`` 딕셔너리를 :class:`PowerSystem` 으로 변환."""
    base_mva = float(ppc["baseMVA"])
    bus = np.real(np.asarray(ppc["bus"], dtype=complex))
    branch = np.asarray(ppc["branch"], dtype=complex)
    gen = np.real(np.asarray(ppc["gen"], dtype=complex))

    # ppc 의 모선 번호가 0..nb-1 로 연속이 아닐 수 있으므로 내부 인덱스로 재사상
    bus_ids = bus[:, BUS_I].astype(int)
    id2idx = {int(b): i for i, b in enumerate(bus_ids)}

    # 탭비는 복소수(위상 이동 포함)로 들어오는 구현이 있어 크기/각도로 분리한다.
    tap_raw = branch[:, TAP]
    tap = np.abs(tap_raw)
    tap[tap == 0.0] = 1.0          # MATPOWER 규약: 0 은 "변압기 아님" = 1.0
    shift = np.deg2rad(np.real(branch[:, SHIFT])) + np.angle(tap_raw)

    ng = gen.shape[0]
    gencost = ppc.get("gencost")
    c2 = np.zeros(ng)
    c1 = np.zeros(ng)
    c0 = np.zeros(ng)
    if gencost is not None:
        gc = np.real(np.asarray(gencost, dtype=complex))
        if gc.ndim == 1:
            gc = gc.reshape(1, -1)
        for i in range(min(ng, gc.shape[0])):
            model, ncost = int(gc[i, 0]), int(gc[i, 3])
            if model != 2:
                raise NotImplementedError(
                    "구간선형(piecewise) 비용 모델은 아직 지원하지 않습니다 "
                    f"(gencost[{i}] model={model})."
                )
            coeffs = gc[i, 4 : 4 + ncost]      # 고차항부터 나열됨
            padded = np.zeros(3)
            padded[3 - ncost :] = coeffs       # [c2, c1, c0] 로 정렬
            c2[i], c1[i], c0[i] = padded

    # ---- pandapower 확장 파라미터 --------------------------------------
    # ppc 의 branch 행렬이 26열이면(= pandapower 내부 ppc) 위치로 읽고,
    # 22열이면(= to_ppc 결과) pandapower 가 따로 빼 둔 키에서 읽는다.
    # to_ppc 는 확장 열이 전부 0 이면 키 자체를 만들지 않으므로 기본값은 0 이다.
    nl_ = branch.shape[0]
    _KEY_BY_COL = {
        BR_R_ASYM: "branch_r_asym",
        BR_X_ASYM: "branch_x_asym",
        BR_G: "branch_g",
        BR_G_ASYM: "branch_g_asym",
        BR_B_ASYM: "branch_b_asym",
    }

    def _col(idx: int) -> np.ndarray:
        if branch.shape[1] >= 26:
            return np.real(branch[:, idx])
        val = ppc.get(_KEY_BY_COL[idx])
        if val is None:
            return np.zeros(nl_)
        return np.real(np.asarray(val, dtype=complex)).ravel()

    return PowerSystem(
        name=name,
        base_mva=base_mva,
        bus_type=bus[:, BUS_TYPE].astype(int),
        Pd=bus[:, PD] / base_mva,
        Qd=bus[:, QD] / base_mva,
        Gs=bus[:, GS] / base_mva,
        Bs=bus[:, BS] / base_mva,
        Vmax=bus[:, VMAX].copy(),
        Vmin=bus[:, VMIN].copy(),
        Vm0=bus[:, VM].copy(),
        Va0=np.deg2rad(bus[:, VA]),
        base_kv=bus[:, BASE_KV].copy(),
        f_bus=np.array([id2idx[int(b)] for b in np.real(branch[:, F_BUS])]),
        t_bus=np.array([id2idx[int(b)] for b in np.real(branch[:, T_BUS])]),
        br_r=np.real(branch[:, BR_R]),
        br_x=np.real(branch[:, BR_X]),
        br_b=np.real(branch[:, BR_B]),
        br_tap=tap,
        br_shift=shift,
        br_status=np.real(branch[:, BR_STATUS]).astype(int),
        rate_a=np.real(branch[:, RATE_A]) / base_mva,
        gen_bus=np.array([id2idx[int(b)] for b in gen[:, GEN_BUS]]),
        Pg0=gen[:, PG] / base_mva,
        Qg0=gen[:, QG] / base_mva,
        Pmax=gen[:, PMAX] / base_mva,
        Pmin=gen[:, PMIN] / base_mva,
        Qmax=gen[:, QMAX] / base_mva,
        Qmin=gen[:, QMIN] / base_mva,
        Vg=np.nan_to_num(gen[:, VG], nan=1.0),
        gen_status=gen[:, GEN_STATUS].astype(int),
        cost_c2=c2,
        cost_c1=c1,
        cost_c0=c0,
        br_g=_col(BR_G),
        br_r_asym=_col(BR_R_ASYM),
        br_x_asym=_col(BR_X_ASYM),
        br_g_asym=_col(BR_G_ASYM),
        br_b_asym=_col(BR_B_ASYM),
    )


def load_case(name: str = "case9") -> PowerSystem:
    """pandapower 표준 시험계통을 :class:`PowerSystem` 으로 읽어온다.

    pandapower 는 여기서 "케이스 파일 리더" 역할만 한다. Ybus 구성, 조류계산,
    OPF 는 모두 :mod:`nnopf` 안에서 직접 구현하며, pandapower 결과는 2단계
    정합성 검증에서 정답지로만 쓴다.

    Parameters
    ----------
    name
        ``pandapower.networks`` 의 함수명. 예: ``"case9"``, ``"case14"``,
        ``"case30"``, ``"case57"``, ``"case118"``.
    """
    import pandapower.networks as pn
    from pandapower.converter.pypower import to_ppc

    if not hasattr(pn, name):
        raise ValueError(f"pandapower.networks 에 '{name}' 이(가) 없습니다.")
    net = getattr(pn, name)()
    return from_pandapower(net, name=name)


def from_pandapower(net, name: str = "case") -> PowerSystem:
    """pandapower ``net`` 을 :class:`PowerSystem` 으로 변환한다.

    ``to_ppc`` 는 branch 행렬을 22열로 줄이면서 변압기 철손 ``BR_G`` 등 확장
    파라미터를 ``ppc["branch_g"]`` 같은 별도 키로 옮긴다. :func:`from_ppc` 가
    그 키까지 읽으므로 pandapower 와 동일한 :math:`Y_{bus}` 가 나온다.
    """
    from pandapower.converter.pypower import to_ppc

    ppc = to_ppc(net, init="flat", mode="opf")
    return from_ppc(ppc, name=name)


def load_pandapower_net(name: str = "case9"):
    """검증용 pandapower ``net`` 객체를 그대로 돌려준다."""
    import pandapower.networks as pn

    return getattr(pn, name)()
