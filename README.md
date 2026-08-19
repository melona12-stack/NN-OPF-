# NN-OPF — 신경망 기반 최적조류계산 연구

> 최적조류계산(OPF) 안에서 가장 비싼 부분인 **조류방정식 풀이를 신경망 대체모델로
> 대체**하고, 그 해가 실제 물리계통에서도 실행가능함을 보이는 것을 목표로 하는
> 12개월 연구 프로젝트입니다.

## 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | OPF 기본 개념 학습 + 직접 구현 | ✅ 완료 |
| 2 | pandapower 대조 정합성 검증 | ✅ 완료 — 22/22 테스트 통과 |
| 3 | NN 관련 스터디 | 📋 계획 수립 완료 |
| 4 | NN 으로 조류계산 대체(학습) | ⬜ |
| 5 | 대체모델을 포함한 OPF | ⬜ |

### 1~2단계 검증 결과 요약

| 항목 | 결과 |
|---|---|
| $Y_{bus}$ (case9/14/30/57/118/300) | pandapower 와 오차 **정확히 0.0** |
| 조류계산 전압 (6개 계통) | 최대 오차 **1e-9 ~ 1e-16 pu**, 4~5회 반복 수렴 |
| AC-OPF 비용 (case9/14/30/118) | 상대오차 **1.8e-9 ~ 1.3e-7** |

상세: [`docs/04_validation_report.md`](docs/04_validation_report.md)

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/00_roadmap.md`](docs/00_roadmap.md) | 12개월 마일스톤, 논문 방향, 위험 요소 |
| [`docs/01_opf_basics.md`](docs/01_opf_basics.md) | OPF 이론 (per-unit부터 LMP까지) |
| [`docs/02_reference_uot_toolkit.md`](docs/02_reference_uot_toolkit.md) | Stanford ASL 참고자료 정리 |
| [`docs/03_nn_surrogate_plan.md`](docs/03_nn_surrogate_plan.md) | NN 스터디 계획 + 대체모델 설계 |
| [`docs/04_validation_report.md`](docs/04_validation_report.md) | 2단계 검증 리포트 |

## 코드 구조

```
src/nnopf/
├── case.py        전력계통 데이터 컨테이너 + pandapower 로더
├── ybus.py        어드미턴스 행렬 Ybus 직접 구성
├── powerflow.py   뉴턴-랩슨 조류계산 (해석적 야코비안)
├── opf.py         AC-OPF 비선형계획 (해석적 그래디언트)
└── compare.py     pandapower 대조 검증

scripts/
└── s01_validate_vs_pandapower.py

tests/
└── test_nnopf.py  회귀 테스트 22개
```

### 설계 원칙

1. **pandapower 는 케이스 리더와 정답지로만 쓴다.** 물리 계산은 전부 직접 구현.
2. **조류방정식을 독립 함수로 분리한다** (`opf.py::power_flow_residual`).
   4단계에서 이 함수를 신경망으로 교체하는 것이 프로젝트의 핵심.
3. **모든 주장은 테스트로 고정한다.** 물리 구현을 건드릴 때의 안전망.

## 시작하기

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 정합성 검증 실행
.venv/bin/python scripts/s01_validate_vs_pandapower.py

# 회귀 테스트
.venv/bin/python -m pytest tests/ -q
```

### 사용 예

```python
import sys; sys.path.insert(0, "src")
from nnopf import load_case, solve_power_flow, solve_acopf

sys_ = load_case("case30")
print(sys_.summary())

pf = solve_power_flow(sys_)
print(f"수렴: {pf.converged}, 반복: {pf.iterations}, 전압범위: "
      f"{pf.Vm.min():.4f} ~ {pf.Vm.max():.4f} pu")

opf = solve_acopf(sys_)
print(f"최적 발전비용: {opf.cost:.2f}, 등식잔차: {opf.max_eq_violation:.2e}")
```

## 환경

Python 3.11 / numpy / scipy / pandapower 3.5 / PyTorch 2.x (3단계부터)
