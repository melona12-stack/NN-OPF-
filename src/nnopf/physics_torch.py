r"""조류방정식 잔차의 PyTorch 판 — 4단계 물리정보 손실.

``dataset.physics_residual`` (NumPy) 과 **같은 식**을 미분 가능하게 옮긴 것이다.
정답 라벨 대신 신경망 예측 :math:`(\hat V_m, \hat\theta)` 를 넣으면 그대로
P2 §3.3 의 물리 손실 항이 된다.

두 가지를 신경 썼다.

**① 복소수 대신 실수 산술.**
:math:`V = e + jf` (``e = Vm cos θ``, ``f = Vm sin θ``) 로 두면

.. math::

    Y V = (Ge - Bf) + j(Gf + Be), \qquad
    P = e(Ge-Bf) + f(Gf+Be), \qquad
    Q = f(Ge-Bf) - e(Gf+Be)

복소 autograd 의 미묘한 문제를 피하고 속도도 더 빠르다.

**② N-1 을 행렬 스택이 아니라 4개 원소 보정으로.**
상정사고마다 :math:`Y_{bus}` 를 통째로 들고 있으면 (case118 기준 178개 ×
118² × 4바이트 = 20 MB) 배치마다 그걸 gather 해야 한다. 대신 **기저
:math:`Y_{bus}` 하나만** 두고, 고장 선로가 기여하던 4개 성분
(:math:`Y_{ff}, Y_{ft}, Y_{tf}, Y_{tt}`) 만 빼 준다. 정확히 같은 값이고
표본당 O(1) 이다.
"""

from __future__ import annotations

import numpy as np
import torch

from nnopf.case import PQ, SLACK, PowerSystem
from nnopf.ybus import make_branch_admittance, make_ybus

__all__ = ["ACPhysics"]


class ACPhysics:
    r"""배치 조류방정식 잔차 계산기.

    Parameters
    ----------
    sys
        계통. 기저 :math:`Y_{bus}` 와 브랜치별 기여분을 여기서 뽑는다.
    dtype
        ``torch.float32`` (학습용) 또는 ``torch.float64`` (최종 평가용).
        전압 오차는 :math:`Y_{bus}` 를 거치며 :math:`\max|Y|` 배로 증폭되므로,
        **최종 잔차를 보고할 때는 float64 를 쓴다** (05 문서 §6.1 과 같은 이유).
    """

    def __init__(self, sys: PowerSystem, dtype: torch.dtype = torch.float32) -> None:
        Ybus = np.asarray(make_ybus(sys).todense())
        Yff, Yft, Ytf, Ytt = make_branch_admittance(sys)
        st = sys.br_status.astype(float)

        def t(x: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype)

        self.dtype = dtype
        self.nb = sys.nb
        self.G = t(Ybus.real)                      # (nb, nb)
        self.B = t(Ybus.imag)
        # 고장 시 빼 줄 브랜치 기여분 (이미 out 인 선로는 0)
        self.yff = (t((Yff * st).real), t((Yff * st).imag))
        self.yft = (t((Yft * st).real), t((Yft * st).imag))
        self.ytf = (t((Ytf * st).real), t((Ytf * st).imag))
        self.ytt = (t((Ytt * st).real), t((Ytt * st).imag))
        self.f_bus = torch.as_tensor(sys.f_bus, dtype=torch.long)
        self.t_bus = torch.as_tensor(sys.t_bus, dtype=torch.long)

        # 잔차를 거는 자리 — 지정값이 있는 모선만 (05 문서 §7)
        self.mask_p = torch.as_tensor(sys.bus_type != SLACK, dtype=dtype)  # (nb,)
        self.mask_q = torch.as_tensor(sys.bus_type == PQ, dtype=dtype)
        self.p_scale = None   # set_scale() 로 등록하면 물리 손실이 무차원화된다
        self.q_scale = None

    # ------------------------------------------------------------------ 내부
    def _yv(
        self, e: torch.Tensor, f: torch.Tensor, outage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:math:`YV` 의 실수부·허수부 ``(B, nb)``. 상정사고를 반영한다."""
        yv_re = e @ self.G.T - f @ self.B.T
        yv_im = f @ self.G.T + e @ self.B.T

        hit = torch.nonzero(outage >= 0, as_tuple=False).squeeze(-1)
        if hit.numel() == 0:
            return yv_re, yv_im

        ln = outage[hit]
        p, q = self.f_bus[ln], self.t_bus[ln]
        ep, fp = e[hit, p], f[hit, p]
        eq, fq = e[hit, q], f[hit, q]

        def cmul(ar, ai, br, bi):
            return ar * br - ai * bi, ar * bi + ai * br

        # 고장 선로가 기여하던 몫을 뺀다: 행 f 는 Yff·V_f + Yft·V_t
        fr1, fi1 = cmul(self.yff[0][ln], self.yff[1][ln], ep, fp)
        fr2, fi2 = cmul(self.yft[0][ln], self.yft[1][ln], eq, fq)
        tr1, ti1 = cmul(self.ytf[0][ln], self.ytf[1][ln], ep, fp)
        tr2, ti2 = cmul(self.ytt[0][ln], self.ytt[1][ln], eq, fq)

        zero = torch.zeros_like(yv_re)
        corr_re = zero.index_put((hit, p), -(fr1 + fr2), accumulate=True)
        corr_re = corr_re.index_put((hit, q), -(tr1 + tr2), accumulate=True)
        corr_im = zero.index_put((hit, p), -(fi1 + fi2), accumulate=True)
        corr_im = corr_im.index_put((hit, q), -(ti1 + ti2), accumulate=True)
        return yv_re + corr_re, yv_im + corr_im

    # ------------------------------------------------------------------ 공개
    def injection(
        self, Vm: torch.Tensor, Va: torch.Tensor, outage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """전압에서 모선 주입전력 :math:`(P, Q)` 를 **닫힌 식으로** 계산 ``(B, nb)``.

        학습 대상이 아니다 — 01 문서 §6.3 의 "닫힌 식으로 구할 수 있는 건
        학습시키지 않는다" 원칙 그대로다.
        """
        e = Vm * torch.cos(Va)
        f = Vm * torch.sin(Va)
        yv_re, yv_im = self._yv(e, f, outage)
        return e * yv_re + f * yv_im, f * yv_re - e * yv_im

    def residual(
        self,
        Vm: torch.Tensor,
        Va: torch.Tensor,
        p_spec: torch.Tensor,
        q_spec: torch.Tensor,
        outage: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""잔차 :math:`(\Delta P, \Delta Q)` ``(B, nb)``. 잔차를 안 거는 자리는 0.

        ``p_spec = p_gen + p_ren - Pd``, ``q_spec = -Qd`` (병렬 소자는 넣지 않는다 —
        :math:`Y_{bus}` 대각에 이미 있어 이중계산이 된다).
        """
        P, Q = self.injection(Vm, Va, outage)
        return (p_spec - P) * self.mask_p, (q_spec - Q) * self.mask_q

    def set_scale(self, p_scale: torch.Tensor, q_scale: torch.Tensor) -> None:
        r"""물리 손실을 무차원화할 기준 크기를 등록한다 ``(nb,)``.

        .. important::
           **λ 값은 손실 정규화 방식이 같아야만 논문 사이에 옮길 수 있다.**
           우리 지도 손실은 표준화 공간이라 :math:`O(1)` 인데, 물리 잔차를
           원단위 :math:`\mathrm{pu}^2` 로 두면 같은 숫자 λ 가 전혀 다른 상대
           가중치를 뜻하게 된다. 실제로 P2 의 λ=3e-3 을 그대로 썼더니 우리
           실험에서 가장 나쁜 설정이 됐다 (전압·물리가 같이 나빠지고 조기 종료).

           그래서 잔차를 모선별 지정주입 크기로 나눠 무차원화한다. 그러면 λ 가
           "지도 항 대비 몇 배" 라는 해석 가능한 값이 된다.
        """
        self.p_scale = p_scale.to(self.dtype).clamp(min=1e-3)
        self.q_scale = q_scale.to(self.dtype).clamp(min=1e-3)

    def loss(
        self,
        Vm: torch.Tensor,
        Va: torch.Tensor,
        p_spec: torch.Tensor,
        q_spec: torch.Tensor,
        outage: torch.Tensor,
    ) -> torch.Tensor:
        r"""P2 §3.3 의 물리 손실 (스칼라).

        .. math::

            \mathcal{L}_{phys} =
            \frac{1}{|\mathcal{N}_{PQ}|}\sum_{PQ}(\Delta P^2 + \Delta Q^2)
            + \frac{1}{|\mathcal{N}_{PV}|}\sum_{PV}\Delta P^2

        모선 종류별로 나눠 평균 내는 게 핵심이다. 전체를 한 번에 평균 내면
        모선 수 비율에 따라 PV 항의 비중이 계통마다 달라진다.
        """
        dP, dQ = self.residual(Vm, Va, p_spec, q_spec, outage)
        if getattr(self, "p_scale", None) is not None:
            dP, dQ = dP / self.p_scale, dQ / self.q_scale
        n_pq = self.mask_q.sum().clamp(min=1.0)
        n_pv = (self.mask_p - self.mask_q).sum().clamp(min=1.0)
        pq_term = ((dP * self.mask_q) ** 2 + dQ**2).sum(-1) / n_pq
        pv_term = ((dP * (self.mask_p - self.mask_q)) ** 2).sum(-1) / n_pv
        return (pq_term + pv_term).mean()

    def to(self, dtype: torch.dtype) -> "ACPhysics":
        """dtype 을 바꾼 사본. 최종 평가를 float64 로 돌릴 때 쓴다."""
        import copy

        out = copy.copy(self)
        out.dtype = dtype
        for name in ("G", "B", "mask_p", "mask_q", "p_scale", "q_scale"):
            v = getattr(self, name)
            setattr(out, name, v if v is None else v.to(dtype))
        for name in ("yff", "yft", "ytf", "ytt"):
            re, im = getattr(self, name)
            setattr(out, name, (re.to(dtype), im.to(dtype)))
        return out
