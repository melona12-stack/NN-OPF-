# 06 · 부록 — Stanford ASL *Unbalanced OPF Toolkit* 분석

> **선수 지식**: [01](01_power_flow_and_opf.md)(모선 분류·OPF 정식화).
> **읽고 나면**: "OPF 안의 조류계산을 갈아끼운다"는 말이 코드 수준에서 정확히
> 무엇을 뜻하는지, 그리고 우리 대체모델이 지켜야 할 인터페이스가 무엇인지 알게 됩니다.
> **언제 다시 볼까**: 5단계(M5)에서 신경망을 OPF에 삽입하는 설계를 할 때.
>
> 원문: <https://github.com/StanfordASL/unbalanced-opf-toolkit/blob/master/docs/tutorial/new_power_flow_surrogate/index.rst>

---

## 1. 먼저 짚을 것 — 이 툴킷에는 신경망이 없습니다

프로젝트 참고자료로 주신 사이트인데, 열어 보고 가장 먼저 확인해야 할 사실입니다.

| 항목 | 내용 |
|---|---|
| 언어 | **MATLAB** (Python 아님) |
| 최적화 백엔드 | YALMIP (Gurobi, MOSEK, SeDuMi 등 교체 가능) |
| 계통 모델 | **불평형 3상 배전계통** (GridLAB-D 모델 임포트) |
| 대체모델 종류 | **선형화/SDP 완화** — Bolognani2015 LP, Bernstein2017 LP, Gan2014 LP/SDP |
| 신경망 | **없음** |

그럼 왜 이 자료가 중요할까요?

> [!IMPORTANT] 이 튜토리얼은 "갈아끼우는 자리"를 정의한 문서입니다
> 우리 4~5단계 목표는 **"OPF 내의 조류계산 부분을 신경망으로 대체한다"** 입니다.
> 이 툴킷은 같은 자리에 **선형 근사식**을 끼워 넣습니다.
>
> 즉 **끼워 넣는 대상만 다르고, 끼워 넣는 자리(= 추상 인터페이스)는 완전히
> 동일합니다.** 그래서 이 튜토리얼은 우리에게 **"신경망 대체모델이 지켜야 할
> 계약서"** 역할을 합니다. 아키텍처는 그대로 베끼고 구현체만 신경망으로 바꾸면 됩니다.

---

## 2. 핵심 추상화 — 직접변수 / 간접변수

튜토리얼 첫 문단이 대체모델의 정의를 규정합니다.

> A power flow surrogate relates **"direct variables"** (voltage at the point of
> common coupling [PCC] and power injections at PQ buses) to **"indirect
> variables"** (power injection at PCC and voltage at PQ buses).

| 구분 | 변수 | 의미 |
|---|---|---|
| **직접변수 (direct)** | PCC 전압 $u_{pcc}, \theta_{pcc}$, PQ 모선 주입 $p_i, q_i$ | **입력** — 우리가 지정하는 값 |
| **간접변수 (indirect)** | PCC 주입 $p_{pcc}, q_{pcc}$, PQ 모선 전압 $U_i, T_i$ | **출력** — 조류방정식이 결정하는 값 |

$$
f_{\theta} : \underbrace{(u_{pcc},\ \theta_{pcc},\ p_{PQ},\ q_{PQ})}_{\text{직접변수 = 신경망 입력}}
\;\longmapsto\;
\underbrace{(U_{PQ},\ T_{PQ},\ p_{pcc},\ q_{pcc})}_{\text{간접변수 = 신경망 출력}}
$$

> [!NOTE] 용어 번역
> 배전계통 용어인 **PCC(Point of Common Coupling, 공통연결점)** 는 송전계통 용어로는
> **슬랙 모선**에 해당합니다. 우리 코드에서는 `slack` 입니다.
>
> 그리고 이 직접/간접 구분이 [01 문서](01_power_flow_and_opf.md) §7의
> 슬랙/PV/PQ 분류와 정확히 같은 이야기입니다 — **무엇이 주어지고 무엇이
> 결정되는가.** 같은 개념을 두 분야가 다른 이름으로 부르는 것뿐입니다.

### 2.1 왜 이 방향인가

반대 방향(전압 → 주입)은 $S = V\overline{YV}$ 로 **닫힌 형태의 명시적 식**이라
학습할 필요가 없습니다. 어려운 쪽은 **주입 → 전압**, 즉 비선형 연립방정식을
푸는 방향입니다.

**대체모델이 대체하는 것은 뉴턴-랩슨 반복 그 자체입니다.**
([01 문서](01_power_flow_and_opf.md) §6.3에서 이미 본 비대칭입니다.)

---

## 3. 클래스 구조 — 우리가 그대로 가져올 골격

```
AbstractPowerFlowSurrogateSpec        (사양: "어떤 대체모델을 쓸지" 선언)
  ├─ PowerFlowSurrogateSpec_Bolognani2015_LP
  ├─ PowerFlowSurrogateSpec_Bernstein2017_LP
  ├─ PowerFlowSurrogateSpec_Gan2014_LP / _SDP
  └─ (우리가 추가) PowerFlowSurrogateSpec_NN          ★

AbstractPowerFlowSurrogate            (구현체: 실제 제약식/해를 생성)
  ├─ PowerFlowSurrogate_Bolognani2015_LP
  ├─ PowerFlowSurrogate_Bernstein2017_LP
  ├─ PowerFlowSurrogate_Gan2014_LP / _SDP
  └─ (우리가 추가) PowerFlowSurrogate_NN             ★

OPFproblem                            (최적화 문제 전체를 조립)
ControllableLoad                      (제어 가능 부하 = 결정변수)
```

`Spec` 과 `Surrogate` 를 분리한 이유는 **"설정"과 "인스턴스"의 분리**입니다.
`Spec` 은 가볍고 직렬화 가능한 설정 객체이고, `Create()` 를 호출하면 실제
최적화 변수와 제약을 들고 있는 무거운 `Surrogate` 객체가 만들어집니다.

> 우리 Python 코드에서는 **dataclass 설정 + 팩토리 함수**로 옮기면 됩니다.
> `ScenarioConfig` 가 이미 그런 구조입니다 ([05 문서](05_dataset_generator.md)).

### 3.1 추상 클래스가 강제하는 4개 메서드 — 계약서 본문

| 메서드 | 역할 |
|---|---|
| `SolveApproxPowerFlow` | 주어진 부하와 PCC 전압에서 **근사 조류해**를 낸다 |
| `GetConstraintArray` | 대체모델이 OPF에 기여할 **제약식 배열**을 만든다 (전압 한계, PCC 전압 지정, PCC 주입전력) |
| `AssignBaseCaseSolution` | 모든 결정변수를 **기저 케이스 값**으로 세팅한다 (디버깅 기준점) |
| `ComputeVoltageEstimate` | 대체모델이 추정하는 **전압 크기/위상** `(U, T)` 를 계산한다 |

보유 속성은 `opf_problem` 하나뿐입니다 — 대체모델은 자기가 속한 OPF 문제를
역참조하며, 이를 통해 주변 제약과 결합됩니다.

### 3.2 결정적인 결합 지점

튜토리얼이 "Key Integration Point"로 강조하는 한 줄입니다.

```matlab
[P_inj_array, Q_inj_array] = obj.opf_problem.ComputeNodalPowerInjection();
```

> [!WARNING] 이걸 빠뜨리면 겉보기엔 풀리지만 의미 없는 해가 나옵니다
> 대체모델은 **자기 혼자 노는 게 아니라** OPF 문제가 계산한 모선별 주입전력을
> 받아서 그것을 입력으로 써야 합니다. 이 연결이 없으면 제어가능부하가 계통
> 제약과 묶이지 않아, 최적화는 수렴하는데 물리적으로 무의미한 답이 나옵니다.

**우리 코드에서의 대응**: `src/nnopf/opf.py` 의 `power_flow_residual()` 이
정확히 같은 역할을 합니다. `sys.Cg @ Pg - sys.Pd` 로 모선 주입을 만들고
그것을 전압과 묶는 함수입니다. 4단계에서 이 함수를 신경망으로 교체합니다.

---

## 4. 튜토리얼의 9단계 개발 절차 = 우리 5단계 작업 순서표

| # | 원문 단계 | 우리 프로젝트에서의 대응 |
|---|---|---|
| 1 | `Spec` 클래스 생성 | `SurrogateConfig` dataclass + `build_surrogate()` |
| 2 | 뼈대 `Surrogate` 클래스 (메서드 stub) | `NNPowerFlowSurrogate` 골격 |
| 3 | `SolveApproxPowerFlowAlt()` 로 **논문 결과 재현** | 학습된 신경망의 예측 정확도를 뉴턴-랩슨 해와 비교 |
| 4 | 초기 테스트 케이스 작성 | `tests/test_surrogate.py` |
| 5 | `GetConstraintArrayHelper()` — **핵심 제약 3종** | 신경망을 미분가능 제약으로 OPF에 삽입 |
| 6 | `AssignBaseCaseSolution()` | 기저 케이스 웜스타트 |
| 7 | 테스트 강화 — `AssertConstraintSatisfaction()` | **물리 재검증**: 신경망 OPF 해를 진짜 조류계산에 재투입 |
| 8 | `SolveApproxPowerFlow()` 최종 구현 | 추론 API 확정 |
| 9 | Sphinx 문서화 | 논문 초안 + API 문서 |

### 4.1 3단계의 정량 기준 — 우리가 넘어야 할 선

튜토리얼은 Bernstein(2017) 논문의 Figure 3, 5를 재현하며 다음 기준을 제시합니다.

> targeting **voltage errors under 0.2%** and **power errors under 1.5%**

> [!IMPORTANT] 이 숫자가 우리 최소 기준선입니다
> **선형 근사의 오차 수준이 전압 0.2%, 전력 1.5%** 입니다.
> 신경망이 이보다 못하면 굳이 신경망을 쓸 이유가 없습니다.
> 논문에 반드시 들어가야 할 비교표입니다.

| 기법 | 전압 오차 | 전력 오차 | 비고 |
|---|---|---|---|
| DC 조류계산 | (측정 예정) | (측정 예정) | 가장 단순한 기준선 |
| Bolognani 2015 LP | (측정 예정) | (측정 예정) | 고정점 선형화 |
| **Bernstein 2017 LP** | **< 0.2%** | **< 1.5%** | 튜토리얼 명시 기준 |
| **우리 신경망** | **목표: < 0.1%** | **목표: < 0.5%** | ★ 논문 기여점 |

### 4.2 7단계의 경고 — 이 프로젝트에서 가장 중요한 문장

> **Important Note:** Decision variables satisfying OPF constraints doesn't
> guarantee constraint satisfaction when solving actual power flow equations
> with those load values. The approximation introduces this discrepancy.

번역하면:

> [!CAUTION] 근사모델 OPF의 본질적 위험
> **근사모델을 쓴 OPF가 "제약을 만족하는 해"를 내놓아도, 그 해의 부하값으로
> 진짜 조류방정식을 풀면 제약을 위반할 수 있다.**
>
> 이것이 대체모델 기반 OPF의 본질적 위험이고, 동시에 **우리 논문의 핵심 평가
> 지표**가 되어야 합니다. 평가는 반드시 2단으로 해야 합니다.
>
> 1. **내부 정합성** — 신경망 OPF가 자기 근사모델 기준으로 실행가능한가?
> 2. **물리적 정합성** — 그 해를 **진짜** 뉴턴-랩슨에 재투입했을 때도 전압/조류
>    한계를 지키는가? ← **이쪽이 진짜 지표**

우리 저장소에는 이미 이 검사가 들어가 있습니다:
`tests/test_nnopf.py::test_opf_solution_is_a_true_power_flow_solution`
([02 문서](02_validation.md) §5.3).

> [!NOTE] 배경 논문 중 하나만 이 검증을 합니다
> [04 문서](04_paper_review.md) §5.2의 GAP 3이 정확히 이 얘기입니다 —
> P4만 사후 AC 조류계산 재검증을 하고, P2는 물리 잔차, P3는 위반 여부만 봅니다.
> **여기가 우리가 기여할 수 있는 자리입니다.**

---

## 5. 툴킷이 구현한 4가지 대체모델 = 우리의 비교군

| 클래스 | 원논문 | 방식 | 성격 |
|---|---|---|---|
| `PowerFlowSurrogate_Bolognani2015_LP` | Bolognani & Zampieri (2015) | 고정점 근처 1차 선형화 | LP |
| `PowerFlowSurrogate_Bernstein2017_LP` | Bernstein et al. (2017) | 명시적 선형근사 (튜토리얼 주제) | LP |
| `PowerFlowSurrogate_Gan2014_LP` | Gan et al. (2014) | 방사형 배전망 선형화 | LP |
| `PowerFlowSurrogate_Gan2014_SDP` | Gan et al. (2014) | 반정부호 완화 | SDP |

이 넷은 우리 논문에서 **비교군(baseline)** 으로 삼기 좋은 후보입니다.
다만 **모두 MATLAB 구현**이므로, Python으로 다시 구현하거나 DC-OPF / 선형화 OPF로
대체하는 판단이 필요합니다 (→ [00 문서](00_overview.md) M4).

---

## 6. 이 툴킷을 그대로 쓰지 않는 이유

프로젝트 초기 의사결정 기록으로 남길 가치가 있어 정직하게 정리합니다.

| 이유 | 설명 |
|---|---|
| **언어 불일치** | MATLAB. 우리는 PyTorch로 학습해야 하므로 Python이 필수. MATLAB↔Python 브리지는 비용 대비 이득이 없음 |
| **문제 범위 불일치** | 불평형 3상 **배전**계통 전용. 우리는 평형 단상등가 **송전**계통(IEEE 9/14/30/118) |
| **입력 데이터 의존** | GridLAB-D 모델 임포터 기반. 우리는 pandapower/MATPOWER 케이스 |
| **유지보수 상태** | 2019년 논문 부속 코드로, 활발히 관리되지 않음 |

> **결론: 코드는 쓰지 않되, 아키텍처와 평가 기준은 전부 가져온다.**

---

## 7. 이 자료에서 확정한 설계 결정 5가지

1. **대체할 대상은 "주입 → 전압" 방향이다.** 반대 방향은 닫힌 식이라 학습 불필요.
2. **대체모델은 4개 메서드 계약을 만족해야 한다.**
   근사조류해 / 제약생성 / 기저해할당 / 전압추정.
3. **기준선은 선형근사다.** 전압 0.2%, 전력 1.5%를 넘어서야 의미가 있다.
4. **평가는 2단이다.** 내부 정합성 + 진짜 조류방정식 재투입 검증.
5. **PCC(슬랙) 결합을 빠뜨리면 안 된다.** 대체모델은 OPF의 모선 주입을 받아야 한다.

---

## 8. 더 읽을 것

- Estandia, A. et al. (2019) — 툴킷 인용 문헌
- Bernstein, A. et al. (2017), *Linear power-flow models in multiphase
  distribution networks* — 튜토리얼의 주제
- Bolognani, S. & Zampieri, S. (2015), *On the existence and linear
  approximation of the power flow solution in power distribution networks*
- Gan, L. et al. (2014) — 방사형 배전망 볼록완화

---

## 9. 처음으로 돌아가기

이것으로 전체 문서를 한 바퀴 돌았습니다.

**→ [00 · 프로젝트 전체 그림](00_overview.md)** 으로 돌아가 현재 진행 상황을 확인하세요.

지금 위치는 **M3 5~6주차 완료 / 7~8주차 대기**이고, 다음 작업은
**MLP 기준선 학습**입니다.
