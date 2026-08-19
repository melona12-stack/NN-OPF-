"""nnopf — 신경망 기반 최적조류계산(NN-OPF) 연구 패키지.

단계별 구성
-----------
1단계  OPF 기본 개념 학습 + 직접 구현      -> ``ybus``, ``powerflow``, ``opf``
2단계  pandapower 대조 정합성 검증          -> ``compare``
3단계  NN 스터디 + 학습 데이터 생성         -> ``dataset``
4단계  조류계산 대체(surrogate) 학습        -> (예정) ``models``, ``train``
5단계  대체모델을 포함한 OPF                -> (예정) ``opf_nn``
"""

__version__ = "0.2.0"

from nnopf.case import PowerSystem, load_case, PQ, PV, SLACK
from nnopf.ybus import make_ybus
from nnopf.powerflow import solve_power_flow, PowerFlowResult
from nnopf.opf import solve_acopf, ACOPFResult
from nnopf.dataset import (
    PowerFlowDataset,
    ScenarioConfig,
    generate_dataset,
    physics_residual,
)

__all__ = [
    # 계통 데이터
    "PowerSystem",
    "load_case",
    "PQ",
    "PV",
    "SLACK",
    # 물리 계산
    "make_ybus",
    "solve_power_flow",
    "PowerFlowResult",
    "solve_acopf",
    "ACOPFResult",
    # 학습 데이터 (3단계)
    "ScenarioConfig",
    "PowerFlowDataset",
    "generate_dataset",
    "physics_residual",
]
