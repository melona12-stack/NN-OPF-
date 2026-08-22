# 02 · 우리가 만든 OPF가 맞는지 검증하기

> **선수 지식**: [01 문서](01_power_flow_and_opf.md) — 조류계산과 OPF가 무엇인지.
> **읽고 나면**: 왜 검증이 이 프로젝트에서 가장 중요한 단계인지, 어떻게 3층으로
> 쌓아 검증했는지, 그 과정에서 잡은 버그 두 개와 거기서 배운 것을 알게 됩니다.

---

## 1. 왜 이 단계가 제일 중요한가

### 1.1 우리는 정답을 스스로 만듭니다

일반적인 기계학습 프로젝트는 데이터를 어디선가 받아옵니다. 우리는 다릅니다.

```
우리 뉴턴-랩슨 코드  ──►  (문제, 정답) 쌍 수만 개  ──►  신경망 학습
      └─ 이게 틀리면                                        └─ 전부 무의미
```

**정답 생성기가 틀리면 신경망은 틀린 것을 열심히 학습합니다.** 그리고 그걸
알아챌 방법이 없습니다 — 신경망 출력과 "정답"이 잘 맞으니까요. 학습곡선도
예쁘게 내려가고, 테스트 오차도 작게 나옵니다. 논문을 다 쓰고 나서야 발견하게 됩니다.

> [!IMPORTANT] 그래서 순서가 이렇습니다
> **"먼저 정답 생성기를 기계정밀도로 검증하고, 그다음에 신경망을 붙인다."**
>
> 이 순서를 지키면 이후 모든 실험에서 "혹시 데이터가 틀린 건 아닐까?"라는
> 의심을 하지 않아도 됩니다. 디버깅 대상이 절반으로 줄어듭니다.

### 1.2 무엇과 비교했나 — pandapower

**pandapower**는 독일 프라운호퍼 연구소에서 만든 오픈소스 전력계통 해석 라이브러리입니다.
널리 검증되어 학계·산업계에서 표준처럼 쓰입니다.

여기서 중요한 것은 **우리가 pandapower를 어떻게 썼는가**입니다.

| pandapower의 역할 | 썼나? |
|---|---|
| IEEE 표준 계통 데이터 제공 (케이스 파일 리더) | ✅ 씀 |
| 정답 제공 (`runpp`, `runopp` 결과를 대조군으로) | ✅ 씀 |
| $Y_{bus}$ 구성 코드 | ❌ **직접 구현** |
| 뉴턴-랩슨 코드 | ❌ **직접 구현** |
| AC-OPF 코드 | ❌ **직접 구현** |

> [!NOTE] 왜 굳이 직접 구현했나
> pandapower를 그냥 쓰면 되는 것 아니냐고 할 수 있습니다. 안 됩니다.
> 우리 목표는 **OPF 안의 조류계산 부분을 신경망으로 갈아끼우는 것**입니다.
> 남의 라이브러리는 그 내부를 열어 부품을 바꿔 끼울 수 없습니다.
> 갈아끼우려면 내가 만든 것이어야 합니다.

---

## 2. 검증을 3층으로 쌓았습니다

한 번에 최종 결과만 비교하면 안 됩니다. 틀렸을 때 **어디가 틀렸는지 알 수 없기
때문**입니다. 그래서 아래층부터 하나씩 확인했습니다.

| 층 | 검증 대상 | 판정 기준 | 여기가 틀리면 |
|---|---|---|---|
| **1층** | $Y_{bus}$ 행렬 | pandapower 내부 값과 **완전 일치** | 위층은 볼 필요도 없음 |
| **2층** | 조류계산 해 | 전압 크기·위상, 모선 주입전력 | 데이터 라벨이 전부 틀림 |
| **3층** | OPF 해 | 목적함수 값 + 실행가능성 | 5단계 비교 기준이 사라짐 |

> [!TIP] 이 방식이 실제로 효과가 있었습니다
> 1층에서 버그를 하나 잡았습니다. 만약 3층(OPF 비용)만 봤다면 그 오차가
> 소수점 아래 어딘가에 묻혀 발견하지 못했을 것입니다.
> **검증은 최종 결과가 아니라 중간 산출물 수준에서 해야 합니다.**

---

## 3. 1층 — $Y_{bus}$ 행렬

### 3.0 검증 코드

pandapower 는 `runpp` 를 돌리고 나면 **자기가 실제로 쓴 행렬**을
`net._ppc["internal"]["Ybus"]` 에 남깁니다. 그걸 그대로 꺼내 비교합니다.
아래가 테스트 전체입니다 — 잘라낸 부분이 없습니다.

```python
# tests/test_nnopf.py
PF_CASES = ["case9", "case14", "case30", "case57", "case118"]

@pytest.mark.parametrize("name", PF_CASES)     # 계통 5개를 각각 한 번씩 돌린다
def test_ybus_matches_pandapower(name):
    """Ybus 가 pandapower 내부 Ybus 와 정확히 일치해야 한다."""
    import pandapower as pp
    import pandapower.networks as pn

    sysm = load_case(name)        # 우리 구현으로 계통을 읽는다
    net = getattr(pn, name)()     # 같은 계통을 pandapower 로도 읽는다
    pp.runpp(net, numba=False)    # 이걸 돌려야 net._ppc 에 내부 Ybus 가 채워진다

    ref = net._ppc["internal"]["Ybus"].toarray()          # 저쪽이 실제로 쓴 행렬
    assert np.max(np.abs(sysm.ybus().toarray() - ref)) == 0.0
```

**마지막 줄 하나만 보시면 됩니다.** 두 행렬의 원소별 차이 중 최댓값이 `0.0` 이어야
한다 — **허용오차가 없습니다.**

같은 입력에 같은 공식을 쓰면 부동소수점 연산 순서까지 같아서 비트 단위로 일치해야
하고, 실제로 그렇습니다. `< 1e-12` 같은 여유를 뒀다면 §3.2 의 변압기 철손 버그
(오차 2.1e-4)를 그냥 통과시켰을 것입니다.

### 3.1 결과

| 계통 | 모선 | 브랜치 | 발전기 | 최대 오차 |
|---|---|---|---|---|
| case9   | 9   | 9   | 3  | **0.0** |
| case14  | 14  | 20  | 5  | **0.0** |
| case30  | 30  | 41  | 6  | **0.0** |
| case57  | 57  | 80  | 7  | **0.0** |
| case118 | 118 | 186 | 54 | **0.0** |
| case300 | 300 | 411 | 69 | **0.0** |

**정확히 0.0입니다** — 반올림한 결과가 아니라 비트 단위로 같습니다.

### 3.2 여기서 잡은 진짜 버그

처음에는 이렇지 않았습니다. **case118에서만** 오차 $2.1\times10^{-4}$ 가 남았습니다.
다른 다섯 계통은 전부 0.0인데 하나만 어긋났습니다.

추적한 결과는 이랬습니다.

1. pandapower는 **변압기 철손**(변압기 코어에서 열로 빠지는 손실)을 표현하려고
   브랜치 병렬 컨덕턴스 `BR_G` 라는 열을 씁니다. MATPOWER 원본 규격에는 없는
   **pandapower 확장 열**입니다.
2. 그런데 `to_ppc()` 함수가 브랜치 행렬을 22열로 줄이면서, 이 열을
   `ppc["branch_g"]` 라는 **별도 키로 옮겨** 버립니다.
3. 우리 코드는 열 위치로만 읽고 있었으니 이 정보를 통째로 놓쳤습니다.
4. 그런데 **case9/14/30/57은 변압기 철손이 0**이라 이 버그가 드러나지 않았습니다.
   case118에만 철손이 있는 변압기가 있었던 것입니다.

**수정**: `case.py::from_ppc()` 가 26열 원본과 22열 축약본을 **둘 다** 읽도록 고쳤습니다.

```python
_KEY_BY_COL = {BR_R_ASYM: "branch_r_asym", BR_X_ASYM: "branch_x_asym",
               BR_G: "branch_g", BR_G_ASYM: "branch_g_asym",
               BR_B_ASYM: "branch_b_asym"}

def _col(idx):
    if branch.shape[1] >= 26:          # 26열 원본이면 위치로 읽고
        return np.real(branch[:, idx])
    val = ppc.get(_KEY_BY_COL[idx])    # 22열이면 별도 키에서 읽는다
    return np.zeros(nl_) if val is None else np.real(np.asarray(val, complex)).ravel()
```

> [!CAUTION] 여기서 얻은 교훈 두 가지
> **① 작은 계통만 보고 판단하면 안 됩니다.** case9~57만 돌렸으면 "완벽하다"고
> 결론 냈을 것입니다. 검증은 **여러 크기의 계통**에서 해야 합니다.
>
> **② 라이브러리의 데이터 포맷을 그대로 믿으면 안 됩니다.** 문서에 없는
> 확장 열이 있었고, 변환 함수가 그걸 다른 곳으로 옮겼습니다.
> 중간 산출물을 직접 눈으로 대조하지 않으면 절대 못 찾습니다.

---

## 4. 2층 — 뉴턴-랩슨 조류계산

### 4.0 검증 코드

`compare_power_flow(case)` 는 계통 이름 하나를 받아 **비교 리포트**를 돌려줍니다.
전압으로 한 번, **모선 주입전력으로 또 한 번** 대조합니다 — 전압만 보면 발전기
매핑이 어긋난 경우를 놓치기 때문입니다.

먼저 위상각 빼기에 쓰는 작은 도우미 하나입니다.

```python
# src/nnopf/compare.py
def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """위상각 차이를 (-pi, pi] 로 감아서 계산한다."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi
```

위상은 $2\pi$ 주기라 $-\pi$ 와 $+\pi$ 가 같은 각인데, 그냥 빼면 $2\pi$ 차이로
보입니다. 그래서 감아서 빼야 합니다.

본체입니다. 이것도 함수 전체입니다.

```python
# src/nnopf/compare.py
def compare_power_flow(case: str = "case9", tol: float = 1e-6) -> ComparisonReport:
    """뉴턴-랩슨 조류계산을 pandapower ``runpp`` 와 비교한다."""
    import pandapower as pp

    sysm = load_case(case)                   # 우리 계통 데이터
    net = load_pandapower_net(case)          # 같은 계통, pandapower 쪽
    pp.runpp(net, numba=False, tolerance_mva=1e-10)   # 저쪽을 우리보다 조인다

    mine = solve_power_flow(sysm, tol=1e-11)          # 우리 뉴턴-랩슨

    # --- (1) 전압으로 비교 ------------------------------------------------
    ref_vm = net.res_bus.vm_pu.to_numpy()
    ref_va = np.deg2rad(net.res_bus.va_degree.to_numpy())   # deg -> rad

    d_vm = float(np.max(np.abs(mine.Vm - ref_vm)))
    d_va = float(np.max(np.abs(_angle_diff(mine.Va, ref_va))))

    # --- (2) 주입전력으로 한 번 더 ----------------------------------------
    # 축을 먼저 맞춰야 한다 (§4.3 에서 겪은 함정).
    #   우리 S = V * conj(Ybus V)  -> 병렬 소자는 Ybus 안에 있으므로 제외된다
    #   저쪽 res_bus.p_mw          -> 그 모선에 붙은 모든 요소의 합, 부호도 반대
    ref_p = -net.res_bus.p_mw.to_numpy() / sysm.base_mva    # 소비(+) -> 주입(+)
    ref_q = -net.res_bus.q_mvar.to_numpy() / sysm.base_mva

    Ybus  = sysm.ybus()
    S     = mine.V * np.conj(Ybus @ mine.V)                  # 총 주입
    S_sh  = np.abs(mine.V) ** 2 * np.conj(sysm.Gs + 1j * sysm.Bs)   # 병렬 소자 몫
    S_src = S - S_sh                                         # 외부 소스의 순주입

    d_p = float(np.max(np.abs(np.real(S_src) - ref_p)))
    d_q = float(np.max(np.abs(np.imag(S_src) - ref_q)))

    # --- (3) 판정 ---------------------------------------------------------
    metrics = {
        "max |dVm| [pu]":   d_vm,
        "max |dVa| [rad]":  d_va,
        "max |dP_bus| [pu]": d_p,
        "max |dQ_bus| [pu]": d_q,
        "my mismatch [pu]": mine.max_mismatch,
        "iterations":       float(mine.iterations),
    }
    notes = []
    if not mine.converged:
        notes.append("직접 구현 조류계산이 수렴하지 않았습니다.")

    ok = mine.converged and max(d_vm, d_va, d_p, d_q) < tol
    return ComparisonReport(case, "pf", ok, metrics, notes)
```

**세 곳만 보시면 됩니다.**

| 줄 | 무엇이 중요한가 |
|---|---|
| `tolerance_mva=1e-10` | 기본값으로 두면 pandapower 쪽이 덜 수렴한 상태라, 남는 차이가 **우리 오차인지 저쪽 오차인지 구분되지 않습니다** |
| `ref_p = -net.res_bus.p_mw...` | 저쪽은 **소비(+)** 기준, 우리는 **주입(+)** 기준이라 부호를 뒤집습니다 |
| `S_src = S - S_sh` | 병렬 소자 몫을 빼서 **같은 것끼리** 비교합니다. 이걸 안 했을 때 case14 에서 0.21 pu 가 남았고, 그 값이 정확히 그 계통의 병렬 커패시터 용량이었습니다 |

### 4.1 결과

| 계통 | 전압크기 오차 [pu] | 위상 오차 [rad] | 반복 | 최종 불일치 [pu] | 시간 [ms] |
|---|---|---|---|---|---|
| case9   | 3.6e-11 | 2.1e-11 | 4 | 5.5e-14 | 15.8 |
| case14  | 6.2e-12 | 6.7e-12 | 4 | 5.4e-15 | 14.8 |
| case30  | 8.9e-12 | 1.1e-11 | 4 | 1.2e-14 | 15.4 |
| case57  | 3.1e-09 | 3.1e-09 | 5 | 1.2e-14 | 20.5 |
| case118 | 9.5e-11 | 1.9e-10 | 5 | 7.7e-14 | 24.2 |
| case300 | 5.3e-11 | 1.0e-10 | 5 | 1.9e-12 | 27.9 |

**전 계통에서 4~5회 반복으로 수렴**하고, pandapower 해와의 차이는 $10^{-9}$ pu
이하입니다. 이건 사실상 **pandapower 쪽의 수렴 허용오차 수준**입니다.
(pandapower의 허용오차를 `tolerance_mva=1e-10` 으로 조이면 우리와의 차이가
$10^{-15}$ 까지 내려갑니다. 즉 남은 차이는 우리 오차가 아니라 저쪽이 덜 조인 것입니다.)

> [!NOTE] "시간 [ms]" 열을 기억해 두세요
> case118 조류계산 1건에 24 ms입니다. 4단계에서 신경망이 이 시간을 얼마나
> 줄이는지가 성능 지표가 됩니다. 참고로 배경 논문 P2는 같은 계통에서
> 뉴턴-랩슨 12 ms, PI-GAT CPU 단건 1.4 ms(8.7배), GPU 배치 0.048 ms(257배)를
> 보고했습니다. 우리 24 ms가 같은 자릿수이므로 구현이 정상 범위입니다.

### 4.2 2차 수렴이 그대로 재현됩니다

case9의 실제 반복 이력입니다.

```
iter 0   max|불일치| = 1.630e+00
iter 1   max|불일치| = 1.671e-01     (10배)
iter 2   max|불일치| = 1.887e-03     (100배)
iter 3   max|불일치| = 5.714e-07     (3,000배)
iter 4   max|불일치| = 5.529e-14     (1,000만배)
```

오차의 자릿수가 매 반복마다 대략 **2배씩** 늘어납니다. 교과서에 나오는 뉴턴법의
2차 수렴 그대로입니다.

> [!TIP] 이 패턴이 해석적 야코비안이 맞다는 강력한 증거입니다
> 야코비안 유도에 실수가 있으면 이 패턴이 무너집니다. 수치미분을 쓰면
> 자릿수 손실 때문에 마지막 몇 자리에서 정체됩니다. **수렴 이력을 찍어 보는 것
> 자체가 무료 검증 도구입니다.**

### 4.3 두 번째 함정 — 버그가 아니라 회계 규약 차이

처음에 case14/case30에서 무효전력 주입 오차가 각각 **0.21 pu, 1.8e-3 pu**로
나왔습니다. 전압은 $10^{-12}$ 로 완벽한데 전력만 어긋나니 이상했습니다.

원인은 코드 버그가 아니라 **비교 축이 달랐던 것**입니다.

| | 우리 $S = V\overline{YV}$ | pandapower `res_bus.q_mvar` |
|---|---|---|
| 부호 | 주입(+) 기준 | **소비(+) 기준** |
| 병렬 소자(shunt) | $Y_{bus}$ 안에 있음 → **제외됨** | 모선 요소로 **포함됨** |

병렬 소자 몫 $S_{sh} = |V|^2\overline{(G_s + jB_s)}$ 를 빼서 축을 맞추자
$10^{-11}$ 로 떨어졌습니다. 그리고 case14의 차이 0.21 pu = 21 Mvar는
**정확히 그 계통의 병렬 커패시터 용량**이었습니다.

```python
S_sh = np.abs(mine.V) ** 2 * np.conj(sysm.Gs + 1j * sysm.Bs)
S_src = S - S_sh          # 축을 맞춘 뒤 비교
```

> [!IMPORTANT] 이 교훈이 앞으로도 계속 쓰입니다
> **불일치가 나오면 먼저 "같은 것을 비교하고 있는가"를 의심하세요.**
> 부호 규약과 요소 포함 범위가 다른 경우가 실제 버그보다 훨씬 흔합니다.
> 차이값이 계통의 어떤 물리량과 정확히 일치하는지 확인해 보면 대개 답이 나옵니다.
>
> 참고로 배경 논문 P2도 물리정보 손실을 만들 때 **똑같은 이슈**를 다룹니다 —
> 고정 병렬 소자를 지정값 쪽에 넣으면 이중계산이 됩니다
> (→ [04 문서](04_paper_review.md) §2.3).

---

## 5. 3층 — AC-OPF

### 5.0 검증 코드

`compare_opf(case)` 도 구조는 같습니다. 다만 **OPF 는 해가 여러 개일 수 있어서**
판정 기준을 둘로 나눕니다 — 실행가능성과 목적함수 값만 불합격 사유로 삼고,
전압 프로파일은 경고로만 남깁니다.

```python
# src/nnopf/compare.py
def compare_opf(case: str = "case9", tol_cost_rel: float = 1e-6,
                tol_vm: float = 1e-4, enforce_line_limits: bool = False,
                method: str = "SLSQP") -> ComparisonReport:
    """AC-OPF 를 pandapower ``runopp`` 와 비교한다."""
    import pandapower as pp

    sysm = load_case(case)
    net = load_pandapower_net(case)

    if not enforce_line_limits:
        # 양쪽 모두 선로 한계를 걸지 않도록 맞춘다 — 설정이 다르면 비교가 무의미
        for tbl in ("line", "trafo", "trafo3w"):
            if tbl in net and "max_loading_percent" in net[tbl]:
                net[tbl]["max_loading_percent"] = np.nan

    # --- (1) 기준값 얻기 --------------------------------------------------
    # pandapower 내부 IPM 은 초기치에 민감해서 한 번에 실패하는 경우가 있다.
    # 초기치를 바꿔 가며 세 번 시도한다.
    notes: list[str] = []
    ref_cost = ref_vm = None
    for kwargs in ({}, {"init": "flat"}, {"init": "pf"}):
        try:
            pp.runopp(net, numba=False, **kwargs)
            ref_cost = float(net.res_cost)
            ref_vm = net.res_bus.vm_pu.to_numpy()
            break
        except Exception:
            net = load_pandapower_net(case)      # 실패하면 깨끗한 상태로 되돌린다

    if ref_cost is None:
        return ComparisonReport(case, "opf", False, {},
                                ["pandapower runopp 실패(모든 초기치)"])

    # --- (2) 우리 해와 비교 -----------------------------------------------
    mine = solve_acopf(sysm, enforce_line_limits=enforce_line_limits, method=method)

    d_cost   = abs(mine.cost - ref_cost)
    rel_cost = d_cost / max(abs(ref_cost), 1e-9)
    d_vm     = float(np.max(np.abs(mine.Vm - ref_vm)))

    metrics = {
        "my cost":          mine.cost,
        "pandapower cost":  ref_cost,
        "rel cost diff":    rel_cost,
        "max |dVm| [pu]":   d_vm,
        "eq residual [pu]": mine.max_eq_violation,
        "solve time [s]":   mine.solve_time,
    }

    # --- (3) 판정 ---------------------------------------------------------
    feasible = mine.max_eq_violation < 1e-6     # 조류방정식을 만족하는가

    if rel_cost >= tol_cost_rel:
        notes.append("비용이 다릅니다 — 제약 설정이 어긋났을 가능성이 큽니다.")
    elif d_vm >= tol_vm:
        # 비용은 같은데 전압만 다르다 -> 불합격이 아니라 경고
        notes.append(f"비용은 같은데 전압이 {d_vm:.2e} pu 다릅니다 "
                     "— 최적해가 평평할 수 있습니다.")

    ok = feasible and rel_cost < tol_cost_rel   # 합격은 이 둘로만 판정한다
    return ComparisonReport(case, "opf", ok, metrics, notes)
```

**마지막 줄이 핵심입니다.** `ok = feasible and rel_cost < tol_cost_rel` —
전압 차이 `d_vm` 은 `ok` 계산에 **들어가지 않습니다.** 왜 그런지는 §5.2 에 있습니다.

### 5.1 결과

| 계통 | 우리 비용 | pandapower 비용 | 상대오차 | 전압 차이 [pu] | 등식잔차 [pu] | 시간 [s] |
|---|---|---|---|---|---|---|
| case9   | 5,311.91  | 5,311.91  | **1.8e-09** | 9.0e-07 | 9.5e-14 | 0.15 |
| case14  | 8,081.53  | 8,081.53  | **4.0e-08** | 1.4e-06 | 2.7e-11 | 0.09 |
| case30  | 575.40    | 575.40    | **1.3e-07** | 2.2e-05 | 1.1e-10 | 0.13 |
| case57  | — | **pandapower 미수렴** | — | — | — | — |
| case118 | 129,704.74 | 129,704.74 | **1.5e-08** | 1.1e-04 | 4.8e-10 | 43.7 |

### 5.2 합격 기준을 왜 그렇게 정했나

- **주 기준**: 목적함수 값의 상대오차 < 1e-6, 그리고 등식제약 잔차 < 1e-6
- **부 기준(경고만)**: 전압 프로파일 차이

전압 차이를 불합격 사유로 삼지 않은 이유가 있습니다.

> [!NOTE] 최적해가 "평평(degenerate)"할 수 있습니다
> OPF에서는 **같은 비용을 내는 서로 다른 전압 프로파일**이 존재할 수 있습니다.
> 목적함수가 발전 유효전력만 보는데 무효전력·전압에는 여유가 있으면, 여러 운전점이
> 정확히 같은 비용을 냅니다. 이때 두 솔버가 서로 다른 점을 골라도 **둘 다 옳습니다.**
>
> case118의 전압 차이 $1.1\times10^{-4}$ 가 정확히 이 경우입니다.
> 비용은 소수점 둘째 자리까지 같은데 전압 프로파일만 조금 다릅니다.

### 5.3 비용이 맞는 것만으로는 부족합니다

비용이 같아도 그 해가 물리적으로 성립하지 않으면 의미가 없습니다.
그래서 별도의 테스트를 넣었습니다.

```
OPF 해의 (Pg, Vm_gen)  ──►  뉴턴-랩슨에 다시 투입  ──►  나온 전압이 OPF 전압과 같은가?
```

```python
# tests/test_nnopf.py
@pytest.mark.parametrize("name", ["case9", "case30"])
def test_opf_solution_is_a_true_power_flow_solution(name):
    """OPF 해의 발전 지령을 조류계산에 다시 넣으면 같은 전압이 나와야 한다."""
    sysm = load_case(name)
    opt = solve_acopf(sysm)

    # ① OPF 스스로는 "제약을 만족한다"고 말한다
    assert opt.max_eq_violation < 1e-6

    # ② 그 발전 지령을 진짜 조류계산에 다시 넣는다
    #    Pg 는 OPF 가 정한 발전량, Vm_set 은 발전기 모선의 전압 지정값
    pf = solve_power_flow(sysm, Pg=opt.Pg,
                          Vm_set=opt.Vm[sysm.gen_bus], tol=1e-11)

    # ③ 같은 전압이 나오는가 — 여기가 진짜 판정이다
    assert pf.converged
    assert np.max(np.abs(pf.Vm - opt.Vm)) < 1e-6
```

**① 과 ③ 이 다른 질문입니다.** ① 은 OPF 가 *자기 근사 기준으로* 만족한다는
자기신고이고, ③ 은 *독립적으로 돌린 뉴턴-랩슨* 이 같은 답을 내는지입니다.
대체모델을 넣으면 ① 은 통과하고 ③ 이 깨지는 일이 생깁니다 — 그게 아래 경고입니다.

case9, case30에서 전압 차이 $< 10^{-6}$ 을 확인했습니다.

> [!IMPORTANT] 이 테스트가 4~5단계의 핵심 평가 도구가 됩니다
> 신경망 대체모델을 OPF에 넣으면, 그 OPF는 **자기 근사모델 기준으로는** 제약을
> 만족한다고 말할 것입니다. 하지만 **진짜 조류방정식에 넣으면 위반할 수 있습니다.**
>
> 이 함정이 대체모델 기반 OPF의 본질적 위험이고, 참고자료가 명시적으로 경고하는
> 부분입니다 (→ [07 부록](07_appendix_reference.md) §4.2).
> 위 테스트를 확장한 것이 우리 논문의 **가장 중요한 평가 지표**가 됩니다.

---

## 6. 미해결 항목 — 정직하게

### 6.1 case57 OPF는 양쪽 다 실패합니다

| 사실 | 확인 방법 |
|---|---|
| pandapower `runopp` 이 기본 / `init="flat"` / `init="pf"` **세 초기치 모두에서 미수렴** | 검증 로그 |
| 우리 SLSQP도 등식잔차 0.55 pu로 실패 | 별도 진단 |
| `trust-constr` 도 실패 (등식잔차 0.24 pu) | 별도 진단 |

**원인**: pandapower의 `case57` 데이터셋 자체가 병적입니다.
기저 조류해에서 **57개 모선 중 39개가 전압 하한 0.94 pu 미만**이고,
최저값이 **0.7199 pu**(모선 30)입니다. 이 상태에서 모든 모선을 $[0.94, 1.06]$ 안으로
끌어올리는 실행가능해가 **존재하는지 자체가 불분명**합니다.

**대응**: 우리 구현의 결함이 아니므로 회귀 테스트의 OPF 대상에서 제외했고
(`tests/test_nnopf.py::OPF_CASES`), 조류계산 검증에는 그대로 포함했습니다
(그쪽은 통과합니다). 필요하면 MATPOWER 원본 case57로 교차 확인할 예정입니다.

### 6.2 case118 OPF가 44초 걸립니다

SLSQP는 등식제약 야코비안을 **조밀(dense) 행렬로 요구**합니다. case118은 변수 344개 /
등식제약 236개라 매 반복마다 $236\times344$ 조밀 행렬을 만듭니다.
실제 $Y_{bus}$ 는 희소한데 그 구조를 못 살리는 것입니다.

| 개선안 | 기대 효과 | 비용 |
|---|---|---|
| IPOPT (`cyipopt`) 도입 | 희소성 활용, 10~100배 | 외부 의존성 추가 |
| 희소 SQP 직접 구현 | 학습 효과 큼 | 개발 시간 |
| **신경망 웜스타트** (5단계 목표) | 반복 수 감소 | — |

> [!TIP] 당장은 문제가 되지 않습니다
> 3~4단계 데이터 생성의 주 작업은 OPF가 아니라 **조류계산**(계통당 15~28 ms)입니다.
> 실제로 case118 20,000건을 66초에 만들었습니다. M5에서 재검토합니다.

---

## 7. 회귀 테스트 — 앞으로의 안전망

```
$ .venv/bin/python -m pytest tests/ -q
86 passed
```

1~2단계 테스트 22개의 내용입니다. 나머지는 3단계 22개
([05 문서](05_dataset_generator.md)), 4단계 대체모델 32개와 그래프 어텐션 10개
([06 문서](06_surrogate_training.md)) — 합쳐서 86개입니다.

| 테스트 | 대상 | 고정하는 주장 |
|---|---|---|
| `test_ybus_matches_pandapower` | 5개 계통 | $Y_{bus}$ 오차 정확히 0.0 |
| `test_power_flow_matches_pandapower` | 5개 계통 | 조류해 오차 < 1e-9 |
| `test_opf_matches_pandapower` | 4개 계통 | OPF 비용 상대오차 < 1e-6 |
| `test_opf_solution_is_a_true_power_flow_solution` | 2개 계통 | OPF 해가 실제 조류해 |
| `test_opf_respects_bounds` | 4개 계통 | 모든 상자제약 만족 |
| `test_bus_type_partition` | case30 | 모선 분류의 완전성 |
| `test_analytic_jacobian_matches_finite_difference` | case14 | 야코비안 유도 정확성 |

> [!IMPORTANT] 코드를 바꿀 때마다 이걸 먼저 돌리세요
> 4~5단계에서 신경망을 붙이려면 물리 구현을 건드리게 됩니다.
> 그때 이 테스트가 **"내가 뭘 깨뜨렸는지 즉시 알려주는 장치"** 가 됩니다.
> 테스트가 없으면 신경망이 학습을 못 하는 이유가 모델 문제인지 물리 코드를
> 깨뜨려서인지 구분할 수 없습니다.

---

## 8. 결론

| 항목 | 판정 |
|---|---|
| $Y_{bus}$ 구성 | ✅ 완전 일치 (6개 계통) |
| 뉴턴-랩슨 조류계산 | ✅ 기계정밀도 일치 (6개 계통) |
| AC-OPF | ✅ 4개 계통 일치, case57은 데이터셋 문제로 보류 |
| 회귀 테스트 | ✅ 전부 통과 |

**2단계 목표 달성.** 이제 이 조류계산기를 **정답 생성기**로 신뢰하고
3단계 데이터 생성에 쓸 수 있습니다.

---

## 9. 다음에 읽을 것

물리 쪽 준비는 끝났습니다. 이제 반대편, **신경망**을 배울 차례입니다.

**→ [03 · 신경망 기초와 대체모델 설계](03_neural_networks.md)**

> 재현 방법: `.venv/bin/python scripts/s01_validate_vs_pandapower.py`
> 원본 로그: `results/validation_run.txt`
> 환경: Python 3.11.15, numpy 2.4.6, scipy, pandapower 3.5.4, CPU 4코어
