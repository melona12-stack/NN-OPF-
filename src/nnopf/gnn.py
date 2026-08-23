r"""그래프 어텐션 대체모델 — M4 의 본체.

MLP 가 case118 에서 선형 기준선을 못 이겼다 (06 문서 §5.3: 1.14 % vs 0.61 %).
약점은 정확히 하나로 특정돼 있다 — **토폴로지 외삽** (06 문서 §4.1).

    정상 토폴로지   선형 0.84 %   MLP  1.26 %
    미지 N-1        선형 2.62 %   MLP 16.38 %   ← 13배 나빠진다

이유는 구조에 있다. MLP 에게 선로 상태는 입력 벡터 끝에 붙은 **186 개의 0/1
비트**일 뿐이다. 그 비트가 계통 그래프의 어느 간선인지, 끊기면 조류가 어디로
우회하는지 — 모델에 그 정보가 없다. 학습에서 본 고장은 통째로 외우고, 못 본
고장은 손쓸 방법이 없다.

그래서 **선로를 지우는 것이 곧 메시지 경로를 지우는 것**이 되게 만든다.
그러면 못 본 고장이어도 구조가 자동으로 반영된다.

설계에서 지킨 것 넷
-------------------

**① 입력 계약을 바꾸지 않는다.** ``PowerFlowMLP`` 와 **똑같은 평탄 입력**
``X (B, 4·nb + nl)`` 을 받아 모델 안에서 그래프로 되풀어쓴다. 그래야
``Bundle`` · ``prepare`` · ``evaluate`` · ``train`` · 선형 비교군이 한 줄도
안 바뀌고, 분할·정규화·측정이 완전히 같아 **사과 대 사과 비교**가 된다.

**② 어텐션에 선로 파라미터를 넣는다** (P2 §3.2). 어느 이웃을 얼마나 볼지는
그 선로의 임피던스에 달렸다. 노드 임베딩만으로는 그걸 알 수 없다.

**③ 끊긴 선로는 어텐션을 막는다.** 이것이 이 모델의 존재 이유다. P2 는 고장
선로를 ``status=0`` 특징으로만 표시하는데, 그러면 "이 값이 0 이면 무시하라"
는 것까지 학습으로 배워야 한다. 우리는 소프트맥스 전에 로짓을 직접
:math:`-\infty` 로 눌러 **구조로 보장한다.** 절제 실험은
``--no-gate`` 로 켜고 끌 수 있다.

**④ 출력 계약도 그대로.** PQ 모선에만 스케일링 인자 헤드, 슬랙 위상은
계통 기준값(06 문서 §7.4 재발 방지), 선형 지름길 유지(§6 에서 물리 잔차를
1.8 배 준 장치).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnopf.case import PQ, PV, SLACK
from nnopf.models import IOLayout

__all__ = ["GATSpec", "PowerFlowGAT"]

NEG = -1e9        # 소프트맥스 전에 로짓을 누르는 값. -inf 는 NaN 을 만든다.


@dataclass
class GATSpec:
    """그래프 어텐션 대체모델의 형태.

    ``hidden`` 은 **헤드 전체 합**이다 (heads 로 나누어 떨어져야 한다).
    """

    hidden: int = 128
    layers: int = 4
    heads: int = 4
    dropout: float = 0.0
    vm_head: str = "scaled"      # PowerFlowMLP 와 같은 뜻
    vm_margin: float = 0.05
    residual: bool = True        # 선형 지름길
    skip_init: str = "zero"      # "zero" | "lstsq" — models.SurrogateSpec 와 같은 뜻
    skip_freeze: bool = False    # 지름길을 얼려 둔다
    gate: bool = True            # 끊긴 선로의 어텐션을 막는다 (③)
    edge_dim: int = 7            # r, x, sh, tap, status, g, b
    node_id: int = 16            # 모선별 학습 임베딩 차원 (0 이면 끔)
    agg: str = "softmax"         # "softmax" = GAT · "sum" = 어드미턴스 가중 합

    # 층마다 활성값을 들고 있지 않고 역전파 때 다시 계산한다.
    #
    # 어텐션 한 층이 붙들고 있는 간선 텐서 ``(B, E, H, D)`` 는 case118 에서
    # 한 개에 128 MB(B=256, E=490, H=4, D=64)이고, 층 하나가 그런 걸 대여섯 개
    # 만든다. 18층이면 활성값만 **11.6 GB** — RTX 5060 의 8 GB 에 안 들어간다.
    #
    # 켜면 층의 입력 ``(B, N, dim)`` 만 남기므로 18층이 557 MB 로 내려간다.
    # 대신 역전파에서 순전파를 한 번 더 돌아 계산이 약 1/3 늘어난다.
    # 그 대가가 남는 이유는, 지금 느린 원인이 계산이 아니라 **할당기가
    # 한계선에서 캐시를 비웠다 다시 잡기를 반복하는 것**이기 때문이다
    # (06 문서 §8.2 — 연산량으로는 epoch 당 40초인데 실측이 189초였다).
    checkpoint: bool = False


class EdgeGAT(nn.Module):
    r"""엣지 특징을 쓰는 어텐션 한 층.

    .. math::

        \alpha_{ij} = \mathrm{softmax}_j\!\left(
            a^\top \mathrm{LeakyReLU}(W_s h_i + W_d h_j + W_e e_{ij})\right)

    소프트맥스는 **받는 노드별**로 정규화한다. 끊긴 선로는 그 전에 로짓을
    눌러 :math:`\alpha \to 0` 이 되게 한다.
    """

    def __init__(self, dim: int, edge_dim: int, heads: int, gate: bool,
                 agg: str = "softmax") -> None:
        super().__init__()
        assert dim % heads == 0, "hidden 은 heads 로 나누어 떨어져야 한다"
        self.h, self.d = heads, dim // heads
        self.gate, self.agg = gate, agg
        self.src = nn.Linear(dim, dim, bias=False)
        self.dst = nn.Linear(dim, dim, bias=False)
        self.edge = nn.Linear(edge_dim, dim, bias=False)
        self.val = nn.Linear(dim, dim, bias=False)
        self.val_e = nn.Linear(edge_dim, dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, self.d))
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.att.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,        # (B, N, dim)
        ei: torch.Tensor,       # (2, E) 정적 — 배치 전체가 공유
        ea: torch.Tensor,       # (B, E, edge_dim)
        alive: torch.Tensor,    # (B, E) 1 = 살아 있는 선로
    ) -> torch.Tensor:
        B, N, _ = x.shape
        s, d = ei[0], ei[1]
        H, D = self.h, self.d

        hs = self.src(x)[:, s].view(B, -1, H, D)     # (B, E, H, D)
        hd = self.dst(x)[:, d].view(B, -1, H, D)
        he = self.edge(ea).view(B, -1, H, D)

        logit = (F.leaky_relu(hs + hd + he, 0.2) * self.att).sum(-1)   # (B, E, H)
        if self.gate:
            # ③ 끊긴 선로는 여기서 막는다. 학습에 맡기지 않고 구조로 보장.
            logit = logit + (1.0 - alive).unsqueeze(-1) * NEG

        if self.agg == "sum":
            # 소프트맥스를 쓰지 않는다. 조류방정식은 이웃 기여의 **합**이지
            # 평균이 아니다 — :math:`YV` 를 보면 정규화가 없다. 소프트맥스는
            # :math:`\sum\alpha = 1` 을 강제해서, 선로 5개가 붙은 모선과
            # 1개가 붙은 모선이 같은 크기의 메시지를 받게 만든다.
            # 대신 게이트를 0~1 로만 눌러 두고 크기는 살린다.
            alpha = torch.sigmoid(logit) * alive.unsqueeze(-1)
        else:
            # 받는 노드별 소프트맥스. scatter 로 직접 짠다 — 노드 수가 작아
            # 외부 그래프 라이브러리를 들이는 것보다 이쪽이 가볍다.
            idx = d.view(1, -1, 1).expand(B, -1, H)
            big = torch.full((B, N, H), NEG, device=x.device, dtype=logit.dtype)
            big = big.scatter_reduce(1, idx, logit, "amax", include_self=True)
            ex = (logit - big.gather(1, idx)).exp()
            den = torch.zeros(B, N, H, device=x.device, dtype=ex.dtype).scatter_add(
                1, idx, ex
            )
            alpha = ex / den.gather(1, idx).clamp(min=1e-16)

        msg = (self.val(x)[:, s].view(B, -1, H, D)
               + self.val_e(ea).view(B, -1, H, D)) * alpha.unsqueeze(-1)
        agg = torch.zeros(B, N, H, D, device=x.device, dtype=msg.dtype).scatter_add(
            1, d.view(1, -1, 1, 1).expand(B, -1, H, D), msg
        )
        return self.norm(x + self.out(agg.reshape(B, N, H * D)))


class PowerFlowGAT(nn.Module):
    """그래프 어텐션 대체모델. ``PowerFlowMLP`` 와 입출력이 완전히 같다.

    ``forward(X) -> (Vm, Va)`` 이므로 ``train`` · ``evaluate`` · ``s04_plot``
    이 그대로 돌아간다.
    """

    def __init__(
        self,
        layout: IOLayout,
        spec: GATSpec,
        *,
        sysm,
        edge_index: np.ndarray,
        edge_attr_base: np.ndarray,
        edge_line: np.ndarray,
        in_mean: np.ndarray,
        in_std: np.ndarray,
        v_set: np.ndarray,
        vm_lo: np.ndarray,
        vm_hi: np.ndarray,
        va_mean: np.ndarray,
        va_std: np.ndarray,
        vm_w: np.ndarray,
        va_w: np.ndarray,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.nb, self.nl = layout.nb, layout.nl
        self.n_vm, self.n_va = len(layout.pq), len(layout.nonslack)

        def buf(name: str, x, dtype=torch.float32) -> None:
            self.register_buffer(name, torch.as_tensor(np.asarray(x), dtype=dtype))

        buf("in_mean", in_mean); buf("in_std", in_std)
        buf("v_set", v_set); buf("va_ref", layout.va_ref)
        buf("vm_lo", vm_lo); buf("vm_hi", vm_hi)
        buf("va_mean", va_mean); buf("va_std", va_std)
        buf("vm_w", vm_w); buf("va_w", va_w)
        buf("pq_idx", layout.pq, torch.long)
        buf("va_idx", layout.nonslack, torch.long)

        # --- 그래프. 자기 자신으로 가는 엣지를 항상 하나씩 붙인다 ---------
        # 어떤 모선의 인접 선로가 전부 끊기면 소프트맥스 분모가 0 이 된다.
        # 자기 엣지는 절대 끊기지 않으므로 그 경우에도 값이 살아 있다.
        ei = np.asarray(edge_index)
        ea = np.asarray(edge_attr_base)
        el = np.asarray(edge_line)
        loop = np.stack([np.arange(self.nb), np.arange(self.nb)])
        loop_attr = np.zeros((self.nb, ea.shape[1]), ea.dtype)
        loop_attr[:, 4] = 1.0                        # status = 1 (항상 살아 있다)
        buf("edge_index", np.concatenate([ei, loop], 1), torch.long)
        buf("edge_attr_base", np.concatenate([ea, loop_attr], 0))
        # 자기 엣지는 어느 브랜치에도 속하지 않는다 → -1
        buf("edge_line", np.concatenate([el, np.full(self.nb, -1)]), torch.long)

        # 엣지 특징 정규화 — r,x,g,b 는 자릿수가 크게 벌어진다. 부호를 보존하는
        # log 변환 후 표준화하고, status 열(4)은 건드리지 않는다 (§7.1 교훈).
        base = np.concatenate([ea, loop_attr], 0).astype(np.float64)
        t = np.sign(base) * np.log1p(np.abs(base))
        mu, sd = t.mean(0), t.std(0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        mu[4], sd[4] = 0.0, 1.0
        buf("e_mean", mu); buf("e_std", sd)

        # --- 정적 노드 특징: 설정전압 + 모선종류 원-핫 -----------------------
        static = np.stack([
            np.asarray(v_set, np.float32),
            (sysm.bus_type == SLACK).astype(np.float32),
            (sysm.bus_type == PV).astype(np.float32),
            (sysm.bus_type == PQ).astype(np.float32),
        ], axis=-1)                                   # (nb, 4)
        buf("static_node", static)

        dim = spec.hidden
        # 모선마다 학습되는 고유 벡터. GNN 은 정의상 노드를 구분하지 않는데
        # (어느 계통에도 쓰려고), 우리는 **고정된 하나의 계통**을 다룬다.
        # 모선 5 와 27 은 물리적으로 다른 자리이므로 그걸 알려 줘야 한다.
        self.node_id = (
            nn.Parameter(torch.randn(self.nb, spec.node_id) * 0.02)
            if spec.node_id > 0 else None
        )
        n_in = 8 + (spec.node_id if spec.node_id > 0 else 0)
        self.enc = nn.Sequential(nn.Linear(n_in, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            EdgeGAT(dim, spec.edge_dim, spec.heads, spec.gate, spec.agg)
            for _ in range(spec.layers)
        )
        self.drop = nn.Dropout(spec.dropout) if spec.dropout > 0 else nn.Identity()
        # 출력도 모선별 가중치. 공유 헤드는 "모든 모선이 같은 함수" 를
        # 강제하는데, 전압은 모선마다 다른 사상이다.
        self.head_w = nn.Parameter(torch.randn(self.nb, dim, 2) * (dim ** -0.5))
        self.head_b = nn.Parameter(torch.zeros(self.nb, 2))

        # ④ 선형 지름길 — MLP 와 같은 이유로 남긴다 (06 문서 §6)
        self.skip = None
        if spec.residual:
            self.skip = nn.Linear(layout.in_dim, self.n_vm + self.n_va)
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    # ------------------------------------------------------------------ 내부
    def _unfold(self, x: torch.Tensor):
        """평탄 입력을 그래프로 되풀어쓴다. **여기가 계약의 접합부다.**"""
        B, nb = x.shape[0], self.nb
        phys = x[:, : 4 * nb].view(B, 4, nb).transpose(1, 2)        # (B, nb, 4)
        node = torch.cat([phys, self.static_node.expand(B, nb, 4)], -1)
        if self.node_id is not None:
            node = torch.cat([node, self.node_id.expand(B, nb, -1)], -1)

        status = x[:, 4 * nb :]                                     # (B, nl)
        ones = torch.ones(B, self.nb, device=x.device, dtype=x.dtype)
        line = self.edge_line
        alive = torch.where(
            (line >= 0).unsqueeze(0),
            status[:, line.clamp(min=0)],
            ones[:, :1].expand(B, len(line)),
        )                                                            # (B, E)

        ea = self.edge_attr_base.unsqueeze(0).expand(B, -1, -1).clone()
        ea[:, :, 4] = alive                       # 상정사고를 status 열에 반영
        ea = (torch.sign(ea) * torch.log1p(ea.abs()) - self.e_mean) / self.e_std
        return node, ea, alive

    # ------------------------------------------------------------------ 공개
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        z = (x - self.in_mean) / self.in_std
        node, ea, alive = self._unfold(z)

        h = self.enc(node)
        ckpt = self.spec.checkpoint and self.training and torch.is_grad_enabled()
        for blk in self.blocks:
            if ckpt:
                # use_reentrant=False 라야 LayerNorm·dropout 이 있는 블록에서
                # 안전하다. 재진입 방식은 입력에 requires_grad 가 없으면
                # 기울기를 조용히 끊어 먹는다.
                h = torch.utils.checkpoint.checkpoint(
                    blk, h, self.edge_index, ea, alive, use_reentrant=False
                )
            else:
                h = blk(h, self.edge_index, ea, alive)
            h = self.drop(h)

        out = torch.einsum("bnd,ndk->bnk", h, self.head_w) + self.head_b
        vm_raw = out[:, self.pq_idx, 0]
        va_raw = out[:, self.va_idx, 1]
        if self.skip is not None:
            add = self.skip(z)
            vm_raw = vm_raw + add[:, : self.n_vm]
            va_raw = va_raw + add[:, self.n_vm :]

        if self.spec.vm_head == "scaled":
            vm = self.vm_lo + torch.sigmoid(vm_raw) * (self.vm_hi - self.vm_lo)
        else:
            vm = self.vm_lo + vm_raw * self.vm_hi
        va = self.va_mean + va_raw * self.va_std

        Vm = self.v_set.expand(B, self.nb).index_copy(1, self.pq_idx, vm)
        Va = self.va_ref.expand(B, self.nb).index_copy(1, self.va_idx, va)
        return Vm, Va

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
