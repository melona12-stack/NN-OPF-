r"""계통 어드미턴스 행렬(:math:`Y_{bus}`) 구성.

이론
----
각 브랜치(선로 또는 변압기)를 표준 :math:`\pi` 등가회로로 모델링한다.

.. code::

        f ──[ 1 : tap ]──┬── y_s ──┬── t
                         │         │
                       jb/2      jb/2

* 직렬 어드미턴스 :math:`y_s = 1/(r + jx)`
* 총 충전 서셉턴스 :math:`b` 를 양단에 :math:`b/2` 씩 나눠 붙임
* 이상변압기 탭비 :math:`\tau = a e^{j\theta_{shift}}` 는 송단(f)에 위치

브랜치 한 개의 어드미턴스 행렬은

.. math::

    \begin{bmatrix} I_f \\ I_t \end{bmatrix} =
    \begin{bmatrix}
        (y_s + jb/2)/|\tau|^2 & -y_s/\bar{\tau} \\
        -y_s/\tau             & y_s + jb/2
    \end{bmatrix}
    \begin{bmatrix} V_f \\ V_t \end{bmatrix}

전체 :math:`Y_{bus}` 는 이 4개 성분을 모선 결합 행렬로 조립하고,
모선 병렬 소자 :math:`y_{sh} = G_s + jB_s` 를 대각에 더해 얻는다.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["make_ybus", "make_branch_admittance"]


def make_branch_admittance(sys) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """브랜치별 4개 어드미턴스 성분 ``(Yff, Yft, Ytf, Ytt)`` 을 계산한다.

    각 배열의 길이는 브랜치 수 ``nl`` 이다. 개방(status=0)된 브랜치는 0 이 된다.
    """
    status = sys.br_status.astype(float)

    # 직렬 어드미턴스. 개방 브랜치는 0 으로 만들어 조립에서 자동 제외된다.
    # 송단/수단 비대칭 파라미터(br_*_asym)를 지원하므로 두 방향을 따로 계산한다.
    ys_f = status / (sys.br_r + 1j * sys.br_x)
    ys_t = status / (
        (sys.br_r + sys.br_r_asym) + 1j * (sys.br_x + sys.br_x_asym)
    )

    # 병렬(충전) 어드미턴스. 실수부 br_g 는 변압기 철손을 나타낸다.
    ysh_f = status * (sys.br_g + 1j * sys.br_b)
    ysh_t = status * (
        (sys.br_g + sys.br_g_asym) + 1j * (sys.br_b + sys.br_b_asym)
    )

    tau = sys.br_tap * np.exp(1j * sys.br_shift)

    Ytt = ys_t + ysh_t / 2.0
    Yff = (ys_f + ysh_f / 2.0) / (tau * np.conj(tau))
    Yft = -ys_f / np.conj(tau)
    Ytf = -ys_t / tau
    return Yff, Yft, Ytf, Ytt


def make_connection_matrices(sys) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """송단/수단 결합 행렬 ``(Cf, Ct)``, 각각 ``(nl, nb)``.

    ``Cf[l, f_bus[l]] = 1`` 이므로 ``Cf @ V`` 가 브랜치별 송단 전압이 된다.
    """
    nl, nb = sys.nl, sys.nb
    rows = np.arange(nl)
    ones = np.ones(nl)
    Cf = sp.csr_matrix((ones, (rows, sys.f_bus)), shape=(nl, nb))
    Ct = sp.csr_matrix((ones, (rows, sys.t_bus)), shape=(nl, nb))
    return Cf, Ct


def make_ybus(sys, return_branch: bool = False):
    """계통 어드미턴스 행렬 :math:`Y_{bus}` 를 만든다.

    Parameters
    ----------
    sys
        :class:`nnopf.case.PowerSystem`
    return_branch
        True 면 조류 계산용 ``(Ybus, Yf, Yt)`` 를 함께 돌려준다.
        ``If = Yf @ V`` 가 브랜치 송단 전류가 된다.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(nb, nb)`` 복소 희소행렬.
    """
    nl, nb = sys.nl, sys.nb
    Yff, Yft, Ytf, Ytt = make_branch_admittance(sys)
    Cf, Ct = make_connection_matrices(sys)

    rows = np.arange(nl)
    Yf = sp.csr_matrix((Yff, (rows, sys.f_bus)), shape=(nl, nb)) + sp.csr_matrix(
        (Yft, (rows, sys.t_bus)), shape=(nl, nb)
    )
    Yt = sp.csr_matrix((Ytf, (rows, sys.f_bus)), shape=(nl, nb)) + sp.csr_matrix(
        (Ytt, (rows, sys.t_bus)), shape=(nl, nb)
    )

    # 모선 병렬 소자 (커패시터 뱅크, 리액터 등)
    Ysh = sp.diags(sys.Gs + 1j * sys.Bs, format="csr")

    Ybus = (Cf.T @ Yf + Ct.T @ Yt + Ysh).tocsr()

    if return_branch:
        return Ybus, Yf, Yt
    return Ybus


def branch_flows(sys, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """전압 벡터로부터 브랜치 송단/수단 복소전력 ``(Sf, St)`` [pu] 계산.

    부호 규약: 모선에서 브랜치로 **흘러 들어가는** 방향이 양(+).
    """
    _, Yf, Yt = make_ybus(sys, return_branch=True)
    Vf = V[sys.f_bus]
    Vt = V[sys.t_bus]
    Sf = Vf * np.conj(Yf @ V)
    St = Vt * np.conj(Yt @ V)
    return Sf, St
