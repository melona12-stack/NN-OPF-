# 1단계 — OPF 기본 개념 (완전 기초부터)

> 이 문서는 `src/nnopf/` 코드와 1:1 대응됩니다. 개념을 읽고 바로 해당 파일을
> 열어 보세요.

## 0. 전력계통을 수식으로 보는 법

발전기와 부하가 선로로 연결된 망이 있습니다. 알고 싶은 건 **각 모선(bus)의
전압**과 **각 선로에 흐르는 전력**입니다.

### 0.1 왜 per-unit(pu) 인가

실제 계통은 345 kV, 154 kV, 22.9 kV 가 변압기로 섞여 있습니다. 그대로 계산하면
숫자 크기가 제각각이라 수치적으로 불안정합니다. 그래서 기준값으로 나눠 정규화합니다.

$$
V_{pu} = \frac{V_{\text{실제}}}{V_{\text{base}}}, \qquad
S_{pu} = \frac{S_{\text{실제}}}{S_{\text{base}}}, \qquad
Z_{base} = \frac{V_{base}^2}{S_{base}}
$$

보통 $S_{base} = 100$ MVA 를 씁니다. 이러면 정상 전압이 모두 **1.0 근처**가 되어
"0.95 ~ 1.05 사이면 정상" 같은 판단이 계통 전체에서 통일됩니다.

> 코드: `PowerSystem` 의 모든 전력/전압 필드는 pu 입니다 (`case.py` 상단 주석).

### 0.2 어드미턴스 행렬 $Y_{bus}$

선로를 $\pi$ 등가회로로 바꾸고, 키르히호프 전류법칙을 모든 모선에 적용하면

$$
\mathbf{I} = Y_{bus}\,\mathbf{V}
$$

라는 선형관계가 나옵니다. $Y_{bus}$ 의 구성 규칙은 단순합니다.

* **대각 원소** $Y_{ii}$ = 모선 $i$ 에 붙은 **모든** 어드미턴스의 합
* **비대각 원소** $Y_{ij}$ = 모선 $i$–$j$ 를 잇는 어드미턴스의 **음수**

브랜치 하나의 기여는 다음 4개 성분입니다 (탭비 $\tau = a e^{j\theta_{shift}}$):

$$
Y_{tt} = y_s + \frac{y_{sh}}{2}, \quad
Y_{ff} = \frac{y_s + y_{sh}/2}{|\tau|^2}, \quad
Y_{ft} = \frac{-y_s}{\bar\tau}, \quad
Y_{tf} = \frac{-y_s}{\tau}
$$

> 코드: `ybus.py::make_branch_admittance`, `make_ybus`
>
> **실전 함정 하나** — pandapower 는 변압기 철손을 표현하려고 브랜치 병렬
> **컨덕턴스** `BR_G` 를 추가로 씁니다. MATPOWER 원본 규격에는 없는 열이라
> 이걸 빠뜨리면 case118 같은 계통에서 $Y_{bus}$ 가 $10^{-4}$ 수준으로 어긋나고,
> 조류해가 미세하게 달라집니다. 실제로 이 프로젝트에서 처음 겪은 불일치가
> 바로 이것이었습니다. (`case.py` 의 `BR_G` 주석 참조)

---

## 1. 조류계산 (Power Flow) — "지금 계통이 어떤 상태인가"

### 1.1 조류방정식

모선 $i$ 의 주입 복소전력은

$$
S_i = P_i + jQ_i = V_i \overline{\left(\sum_{k} Y_{ik} V_k\right)}
$$

$V_i = |V_i|e^{j\theta_i}$ 로 풀어쓰면

$$
P_i = |V_i| \sum_k |V_k| \left( G_{ik}\cos\theta_{ik} + B_{ik}\sin\theta_{ik} \right) \\
Q_i = |V_i| \sum_k |V_k| \left( G_{ik}\sin\theta_{ik} - B_{ik}\cos\theta_{ik} \right)
$$

여기서 $\theta_{ik} = \theta_i - \theta_k$. **삼각함수 곱이 들어간 비선형 연립방정식**
이고, 이게 이 분야 모든 어려움의 근원입니다. 닫힌 해가 없어서 반복법으로 풀어야 합니다.

### 1.2 모선 분류 — 미지수 세기

방정식 수와 미지수 수를 맞추려면 모선마다 무엇을 알고 무엇을 모르는지 정해야 합니다.

| 종류 | 물리적 대응 | 주어진 값 | 구할 값 | 개수 |
|---|---|---|---|---|
| **슬랙 (Slack)** | 기준 발전기 / 계통연계점 | $\|V\|,\ \theta$ | $P,\ Q$ | 정확히 1개 |
| **PV** | 전압제어 발전기 | $P,\ \|V\|$ | $Q,\ \theta$ | 발전기 수 |
| **PQ** | 부하 모선 | $P,\ Q$ | $\|V\|,\ \theta$ | 나머지 전부 |

**왜 슬랙이 필요한가**: 선로 손실을 미리 알 수 없기 때문입니다. 총 발전량 =
총 부하 + 손실인데, 손실은 조류해가 나와야 알 수 있으므로 닭-달걀 문제입니다.
그래서 한 모선을 "잔여분을 다 떠안는 모선"으로 지정합니다.

> 코드: `case.py::PowerSystem.slack / .pv / .pq / .pvpq`

### 1.3 뉴턴-랩슨법

미지수 $x = [\theta_{PV,PQ};\ |V|_{PQ}]$ 에 대해 불일치 $f(x)=0$ 을 풉니다.

$$
x^{(k+1)} = x^{(k)} + J^{-1} f(x^{(k)}), \qquad
J = \begin{bmatrix}
\partial P/\partial\theta & \partial P/\partial|V| \\
\partial Q/\partial\theta & \partial Q/\partial|V|
\end{bmatrix}
$$

야코비안은 복소수로 한 번에 유도하면 깔끔합니다.

$$
\frac{\partial S}{\partial \theta} = j\,\mathrm{diag}(V)\,
\overline{\mathrm{diag}(YV) - Y\,\mathrm{diag}(V)}, \qquad
\frac{\partial S}{\partial |V|} = \mathrm{diag}(V)\,\overline{Y\,\mathrm{diag}(V/|V|)}
+ \overline{\mathrm{diag}(YV)}\,\mathrm{diag}(V/|V|)
$$

실수부를 취하면 $\partial P/\partial\cdot$, 허수부를 취하면 $\partial Q/\partial\cdot$.

**수렴 특성**: 뉴턴법은 2차 수렴합니다. 실제로 우리 구현에서 오차가
$10^{0} \to 10^{-1} \to 10^{-3} \to 10^{-7} \to 10^{-14}$ 로 줄어듭니다.
**보통 4~5회면 끝납니다.**

> 코드: `powerflow.py::dSbus_dV`, `solve_power_flow`
> 검증: `tests/test_nnopf.py::test_analytic_jacobian_matches_finite_difference`

### 1.4 4단계에서 대체할 것이 바로 이것

**신경망이 대체하는 대상은 이 반복 루프 전체**입니다.

$$
\underbrace{(P^{spec}, Q^{spec}, |V|^{set})}_{\text{입력}}
\;\xrightarrow[\text{4~5회 반복}]{\text{뉴턴-랩슨}}\;
\underbrace{(|V|, \theta)}_{\text{출력}}
\qquad\Longrightarrow\qquad
f_\theta(\cdot) \text{ 한 번의 순전파}
$$

반대 방향 $(|V|,\theta) \to (P,Q)$ 는 **닫힌 식** $S = V\overline{YV}$ 이므로
학습할 필요가 없습니다. 이 비대칭이 대체모델 설계의 출발점입니다.

---

## 2. 최적조류계산 (OPF) — "어떻게 운전해야 가장 싼가"

조류계산은 "지금 상태"를 구하는 것이고, OPF 는 "가장 좋은 상태"를 **고르는** 것입니다.

### 2.1 정식화

$$
\begin{aligned}
\min_{\theta,\,|V|,\,P_g,\,Q_g} \quad & \sum_{i=1}^{ng} \left( c_{2,i}P_{g,i}^2 + c_{1,i}P_{g,i} + c_{0,i} \right) \\
\text{s.t.} \quad
& P_i(\theta,|V|) - \left(\textstyle\sum_{g\in i} P_g - P^d_i\right) = 0 && \forall i \quad \text{(유효전력 균형)} \\
& Q_i(\theta,|V|) - \left(\textstyle\sum_{g\in i} Q_g - Q^d_i\right) = 0 && \forall i \quad \text{(무효전력 균형)} \\
& V^{min}_i \le |V|_i \le V^{max}_i && \forall i \\
& P^{min}_g \le P_g \le P^{max}_g,\quad Q^{min}_g \le Q_g \le Q^{max}_g && \forall g \\
& |S_f| \le S^{max},\quad |S_t| \le S^{max} && \text{(선로 조류 한계)} \\
& \theta_{slack} = 0
\end{aligned}
$$

**핵심**: 등식제약이 정확히 §1.1 의 조류방정식입니다. **OPF = 조류계산 + 최적화**.

> 코드: `opf.py`. 특히 `power_flow_residual()` 이 등식제약을 만드는 함수이고,
> **4단계에서 신경망으로 교체될 지점**이라 일부러 독립 함수로 분리해 두었습니다.

### 2.2 왜 어려운가

| 성질 | 결과 |
|---|---|
| 등식제약이 **비볼록** | 전역최적해 보장 불가, 국소해에 빠질 수 있음 |
| 변수 수 $2n_b + 2n_g$ | case118 → 344개 변수, 236개 등식제약 |
| 제약이 조밀하게 결합 | 한 모선 전압이 바뀌면 이웃 전부 영향 |

그래서 **매 시각 실시간으로 푸는 게 부담**이고, 이것이 대체모델 연구의 동기입니다.

### 2.3 라그랑주 승수 = 한계가격 (LMP)

유효전력 균형 제약의 라그랑주 승수 $\lambda_i^P$ 는 경제적으로 의미가 있습니다:

> **모선 $i$ 에서 부하가 1 MW 늘어날 때 총 발전비용이 얼마나 증가하는가**

이것이 전력시장의 **한계가격(Locational Marginal Price)** 입니다.
pandapower 결과의 `res_bus.lam_p` 가 바로 이 값입니다.
손실과 혼잡이 없으면 모든 모선에서 같고, 선로가 혼잡하면 갈라집니다.

### 2.4 우리 구현의 선택

| 항목 | 선택 | 이유 |
|---|---|---|
| 솔버 | `scipy.optimize` SLSQP (기본) / trust-constr | 외부 의존 없음, 중소 계통에서 충분 |
| 그래디언트 | **해석적** (목적함수 + 등식제약 + 선로제약) | 수치미분 대비 훨씬 빠르고 정확 |
| 초기치 | **조류계산 웜스타트** | 평기동보다 반복 수가 크게 줄어듦 |
| 목적함수 스케일 | $O(1)$ 로 정규화 | SLSQP 의 `ftol` 은 절대 기준이라, 비용이 $10^5$ 규모면 사실상 $10^{-14}$ 상대정밀도를 요구하게 됨 |

**한계**: case118(344변수)에서 ~50초. 더 큰 계통은 IPOPT 같은
내점법 솔버가 필요합니다 (로드맵 §6 위험요소 참조).

---

## 3. 자주 헷갈리는 것들

### 부호 규약
* **주입(injection) 기준**: 계통으로 들어가는 방향이 양(+). 발전기 $P_g > 0$, 부하 $P_d > 0$ 이지만 주입은 $-P_d$.
* pandapower 의 `res_bus.p_mw` 는 **소비 기준**입니다. 우리 주입 기준과 부호가 반대이고, 게다가 **병렬 소자(shunt) 소비분이 포함**됩니다. 비교할 때 반드시 맞춰야 합니다 (`compare.py` 주석 참조).

### 병렬 소자(shunt)
모선에 붙은 커패시터/리액터는 $Y_{bus}$ **대각에 포함**됩니다. 따라서
$S = V\overline{YV}$ 는 "shunt 를 제외한 외부 소스의 순주입"입니다.

### $Q$ 한계 처리
PV 모선의 무효출력이 한계를 넘으면 전압을 유지할 수 없으므로 **PQ 모선으로 전환**하고
다시 풉니다. `solve_power_flow(enforce_q_limits=True)`.
단, OPF 는 $Q_g$ 를 결정변수로 두고 부등식으로 처리하므로 이 전환이 필요 없습니다.

---

## 4. 실행해 보기

```bash
# 환경 준비
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 조류계산 + OPF 를 pandapower 와 대조
.venv/bin/python scripts/s01_validate_vs_pandapower.py

# 회귀 테스트
.venv/bin/python -m pytest tests/ -q
```

## 5. 확인 문제 (스스로 점검)

1. 슬랙 모선이 두 개면 무슨 일이 생기나? 왜 안 되나?
2. $Y_{bus}$ 는 대칭인가? 위상이동 변압기가 있으면?
3. 뉴턴법 야코비안이 특이(singular)해지는 물리적 상황은?
4. OPF 등식제약을 DC 근사로 바꾸면 문제는 무엇이 되나? (힌트: 볼록)
5. 어떤 모선의 LMP 가 다른 모선보다 높다면 계통에서 무슨 일이 벌어지고 있나?
