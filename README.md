# NN-OPF — 신경망 기반 최적조류계산 연구

> 최적조류계산(OPF) 안에서 가장 비싼 부분인 **조류방정식 풀이를 신경망 대체모델로
> 대체**하고, 그 해가 실제 물리계통에서도 실행가능함을 보이는 것을 목표로 하는
> 12개월 연구 프로젝트입니다.

## 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | OPF 기본 개념 학습 + 직접 구현 | ✅ 완료 |
| 2 | pandapower 대조 정합성 검증 | ✅ 완료 — 기계정밀도 일치 |
| 3 | NN 관련 스터디 + 학습 데이터 생성 | ✅ 완료 — 논문 4편 정독 ✅, 데이터 생성기 ✅ |
| 4 | NN 으로 조류계산 대체(학습) | 🔄 진행 중 — **MLP 기준선 ✅**, GNN ⬜ |
| 5 | 대체모델을 포함한 OPF | ⬜ |

**개발 환경**: 현재 CPU 4코어 / 15GB RAM.
**RTX 5060**(Blackwell, 8GB GDDR7, sm_120) 도입 예정 —
PyTorch는 반드시 `cu128` 이상 빌드 필요 ([근거](docs/00_overview.md#8-계산-자원--rtx-5060-도입)).

### 1~2단계 검증 결과 요약

| 항목 | 결과 |
|---|---|
| $Y_{bus}$ (case9/14/30/57/118/300) | pandapower 와 오차 **정확히 0.0** |
| 조류계산 전압 (6개 계통) | 최대 오차 **1e-9 ~ 1e-16 pu**, 4~5회 반복 수렴 |
| AC-OPF 비용 (case9/14/30/118) | 상대오차 **1.8e-9 ~ 1.3e-7** |

상세: [`docs/02_validation.md`](docs/02_validation.md)

## 문서 — 번호 순서대로 읽으면 됩니다

전기공학·신경망 배경이 없어도 읽을 수 있게 썼습니다. 각 문서 맨 위에 선수 지식이
적혀 있고, 맨 아래에 다음에 읽을 문서가 연결되어 있습니다.

| 문서 | 내용 | 선수 지식 |
|---|---|---|
| [`00_overview.md`](docs/00_overview.md) | **여기서 시작** — 무엇을 왜 하는가, 12개월 계획, 계산자원 | 없음 |
| [`01_power_flow_and_opf.md`](docs/01_power_flow_and_opf.md) | 전력계통 기초 → 조류방정식 → 뉴턴-랩슨 → OPF → LMP | 00 |
| [`02_validation.md`](docs/02_validation.md) | 우리 구현이 맞는지 3층으로 검증한 기록 + 잡은 버그 2개 | 01 |
| [`03_neural_networks.md`](docs/03_neural_networks.md) | 신경망 기초(뉴런부터) + 6주 커리큘럼 + 대체모델 설계 | 01 |
| [`04_paper_review.md`](docs/04_paper_review.md) | 배경 논문 4편 정독 + **연구 갭 3개 도출** | 01, 03 |
| [`05_dataset_generator.md`](docs/05_dataset_generator.md) | 학습 데이터 생성기 (설계·검증·잡은 버그 2개) | 01, 03, 04 |
| [`06_surrogate_training.md`](docs/06_surrogate_training.md) | **4단계 대체모델 학습 — 기준선 결과와 잡은 버그 3개** | 03, 05 |
| [`07_appendix_reference.md`](docs/07_appendix_reference.md) | 부록 — Stanford ASL 참고자료 분석 (5단계에서 다시 봄) | 01 |

같은 내용이 Notion에도 정리되어 있습니다 (`tools/md_to_notion.py` 로 변환).

### 배경 논문 4편

| # | 논문 | 역할 | 단계 |
|---|---|---|---|
| P1 | Surrogate Modeling for Solving OPF: A Review (*Sustainability* 2024) | 분야 지형도 | 0 |
| P2 | Power Flow Surrogate via Physics-Informed Graph Attention Network (*Energies* 2026) | 조류계산 대체 | **4** |
| P3 | ICNN-Assisted OPF in Distribution Networks (arXiv 2024) | 볼록 대체모델 삽입 | **5** |
| P4 | Enhanced OPF Using a Trained NN Surrogate (arXiv 2026) | MILP 정확 인코딩 | **5** |

네 편이 하나의 이야기로 이어지고 그 끝에 빈칸이 셋 있습니다 —
**"그래프 + 물리정보 + 볼록성"을 동시에 만족하는 대체모델이 아직 없습니다.**
상세는 [`docs/04_paper_review.md`](docs/04_paper_review.md) §5.

## 코드 구조

```
src/nnopf/
├── case.py        전력계통 데이터 컨테이너 + pandapower 로더
├── ybus.py        어드미턴스 행렬 Ybus 직접 구성
├── powerflow.py   뉴턴-랩슨 조류계산 (해석적 야코비안)
├── opf.py         AC-OPF 비선형계획 (해석적 그래디언트)
├── compare.py     pandapower 대조 검증
├── dataset.py     학습 데이터 생성 (샘플링 + 라벨 + 물리 잔차)
├── physics_torch.py  조류방정식 잔차의 PyTorch 판 (물리손실용)
├── models.py      MLP 대체모델 (스케일링 인자 헤드 + 선형 지름길)
├── train.py       학습 루프 · 평가 지표 · λ 스케줄
├── baselines.py   선형 최소제곱 비교군
└── viz.py         그림 팔레트·스타일 (색각이상 검증 통과)

scripts/
├── s01_validate_vs_pandapower.py   2단계 정합성 검증
├── s02_generate_dataset.py         3단계 데이터 생성
├── s03_train_surrogate.py          4단계 대체모델 학습
├── s03b_topology_breakdown.py      오차를 토폴로지로 분해
├── s04_plot.py                     결과를 그림으로 (figures/)
└── s00_check_env.py                환경 점검 (GPU 실연산까지)

tests/
├── test_nnopf.py      1~2단계 회귀 테스트 22개
├── test_dataset.py    3단계 회귀 테스트 22개
└── test_surrogate.py  4단계 회귀 테스트 22개
```

### 학습 데이터 (3단계)

```bash
.venv/bin/python scripts/s02_generate_dataset.py       # case30 + case118
```

| 계통 | 표본 | 발산 | N-1 | 생성 시간 (4코어) | 조류방정식 잔차 |
|---|---|---|---|---|---|
| case30 | 6,000 | 0% | 25.5% (38종) | 13초 | 9.8e-09 pu |
| case118 | 20,000 | 0% | 25.5% (177종) | 66초 | 2.2e-09 pu |

설정은 P2(PI-GAT) §4.1 을 그대로 복제했습니다 — **P2도 같은 pandapower
`case30`/`case118`을 쓰므로 결과를 논문 수치와 직접 비교할 수 있습니다.**

**데이터 파일은 커밋하지 않습니다.** 시드가 고정되어 있어 어느 컴퓨터에서든
몇 분이면 같은 데이터가 재생성됩니다 (`data/manifest.json` 에 생성 기록만 남김).

### 설계 원칙

1. **pandapower 는 케이스 리더와 정답지로만 쓴다.** 물리 계산은 전부 직접 구현.
2. **조류방정식을 독립 함수로 분리한다** (`opf.py::power_flow_residual`).
   4단계에서 이 함수를 신경망으로 교체하는 것이 프로젝트의 핵심.
3. **모든 주장은 테스트로 고정한다.** 물리 구현을 건드릴 때의 안전망.

## 시작하기

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 환경 점검 — 새 컴퓨터에서 제일 먼저
.venv/bin/python scripts/s00_check_env.py

# 정합성 검증 실행
.venv/bin/python scripts/s01_validate_vs_pandapower.py

# 회귀 테스트 (66개)
.venv/bin/python -m pytest tests/ -q
```

### 윈도우에서

가상환경 실행 파일의 위치가 다릅니다. 문서와 아래 예시의 `.venv/bin/python` 을
전부 `.venv\Scripts\python.exe` 로 바꿔 읽으시면 됩니다.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/s00_check_env.py
```

매번 바꿔 쓰기 번거로우면 가상환경을 활성화해 두면 그냥 `python` 으로 쓸 수 있습니다.

```powershell
.venv\Scripts\Activate.ps1     # 실행정책 오류가 나면:
                                # Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python scripts/s00_check_env.py
```

### GPU 실험 — 클라우드에서 개발, 로컬 GPU 에서 실행

코드는 클라우드 세션에서 쓰고 CPU 로 검증한 뒤, 무거운 학습만 GPU 가 달린
컴퓨터에서 돌리는 방식이다. 결과를 화면에서 복사하지 않고 **저장소로 돌려보낸다** —
로그를 긁으면 숫자가 잘리거나 섞이는데, JSON 을 그대로 받으면 그럴 일이 없다.

```powershell
.\scripts\gpu_run.ps1 s03_train_surrogate.py --case case118
```

`git pull` → 실험 → `results/` 만 커밋 → `git push` 를 한 번에 한다.
체크포인트(`*.pt`)는 용량 때문에 `.gitignore` 에 있으므로 로컬에만 남고,
지표가 담긴 `*.json` 만 돌아온다.

### GPU 설치 (선택)

NVIDIA GPU 가 있으면 4단계 학습이 수십 배 빨라집니다. **계산 능력(compute
capability)에 맞는 PyTorch 빌드**를 깔아야 합니다.

```powershell
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
```

`compute_cap` 이 **12.0** 이면 Blackwell(RTX 50 시리즈)이고, **cu128 이상**
빌드가 필요합니다. 그 이전 빌드로 깔면 `torch.cuda.is_available()` 은 `True` 인데
실제 연산에서 `no kernel image is available` 로 터집니다 — `s00_check_env.py` 가
실제 행렬곱을 한 번 시켜 보고 이 경우를 잡아냅니다.

```powershell
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
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
