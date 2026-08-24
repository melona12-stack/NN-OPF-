# 08 · 수식과 코드 대조표 — 이 저장소가 실제로 계산하는 것

00~07 문서는 **무엇을 왜 했는지**를 적었습니다. 이 문서는 그 아래 한 층,
**어떤 식을 어느 줄이 계산하는지**만 모았습니다. 논문의 방법론 절을 쓸 때
여기만 보면 되도록 만든 것입니다.

읽는 법 — 모든 항목이 **식 → 코드 → 주의점** 세 덩어리입니다. 코드는 실제
파일에서 그대로 옮겼고, 파일과 함수 이름을 같이 적었습니다.

> [!NOTE] 표기 약속
> | 기호 | 뜻 | 코드 |
> |---|---|---|
> | $n_b,\ n_l,\ n_g$ | 모선·브랜치·발전기 수 | `sys.nb`, `sys.nl`, `sys.ng` |
> | $V_i = \lvert V_i\rvert e^{j\theta_i}$ | 모선 전압 (복소) | `Vm`, `Va` |
> | $Y_{bus}$ | 어드미턴스 행렬 $(n_b\times n_b)$ | `Ybus` |
> | $S_i = P_i + jQ_i$ | 모선 주입 복소전력 | `sbus_from_V(...)` |
> | $\bar z$ | 복소 켤레 | `np.conj(z)` |
> | pu | per-unit (기준값 `base_mva` 로 나눈 무차원 값) | 모든 전기량 |
>
> 전기량은 **전부 pu** 입니다. MW 로 보고할 때만 `base_mva` 를 곱합니다.

---

## 1. 계통을 행렬로 — $Y_{bus}$

### 1.1 브랜치 하나의 $\pi$ 등가회로

선로와 변압기를 같은 모양으로 씁니다. 직렬 임피던스 $r+jx$, 양단에 나눠
붙인 충전 서셉턴스 $b/2$, 송단의 이상변압기 탭비 $\tau$.

$$
y_s = \frac{1}{r + jx}, \qquad
y_{sh} = g + jb, \qquad
\tau = a\,e^{j\theta_{shift}}
$$

$$
\begin{bmatrix} I_f \\ I_t \end{bmatrix} =
\begin{bmatrix}
(y_s + y_{sh}/2)/\lvert\tau\rvert^2 & -y_s/\bar\tau \\
-y_s/\tau & y_s + y_{sh}/2
\end{bmatrix}
\begin{bmatrix} V_f \\ V_t \end{bmatrix}
$$

```python
# src/nnopf/ybus.py — make_branch_admittance
ys_f = status / (sys.br_r + 1j * sys.br_x)
ysh_f = status * (sys.br_g + 1j * sys.br_b)
tau = sys.br_tap * np.exp(1j * sys.br_shift)

Ytt = ys_t + ysh_t / 2.0
Yff = (ys_f + ysh_f / 2.0) / (tau * np.conj(tau))
Yft = -ys_f / np.conj(tau)
Ytf = -ys_t / tau
```

> [!IMPORTANT] `status` 를 곱하는 자리가 N-1 의 전부입니다
> 개방된 브랜치는 $y_s = y_{sh} = 0$ 이 되어 조립에서 자동으로 빠집니다.
> 상정사고를 위해 행렬을 다시 만들 필요가 없다는 뜻이고, 이 성질이 §7.3 의
> **4개 성분 보정**으로 이어집니다.
>
> 송단/수단 비대칭 파라미터(`br_*_asym`)를 따로 두어 `ys_f` 와 `ys_t` 를
> 나눠 계산합니다. pandapower 의 일부 변압기 모델이 이걸 씁니다.

### 1.2 조립

$$
Y_{bus} = C_f^\top Y_f + C_t^\top Y_t + \mathrm{diag}(G_s + jB_s)
$$

$C_f[l, f_l] = 1$ 인 결합 행렬입니다. $Y_f$ 는 브랜치별 송단 전류를 내놓는
$(n_l \times n_b)$ 행렬이라 $I_f = Y_f V$ 가 됩니다.

```python
# src/nnopf/ybus.py — make_ybus
Yf = sp.csr_matrix((Yff, (rows, sys.f_bus)), shape=(nl, nb)) \
   + sp.csr_matrix((Yft, (rows, sys.t_bus)), shape=(nl, nb))
Yt = sp.csr_matrix((Ytf, (rows, sys.f_bus)), shape=(nl, nb)) \
   + sp.csr_matrix((Ytt, (rows, sys.t_bus)), shape=(nl, nb))
Ysh = sp.diags(sys.Gs + 1j * sys.Bs, format="csr")
Ybus = (Cf.T @ Yf + Ct.T @ Yt + Ysh).tocsr()
```

### 1.3 브랜치 조류

$$
S_f = V_f \overline{(Y_f V)}, \qquad S_t = V_t \overline{(Y_t V)}
$$

$$
\text{이용률}\ [\%] = 100 \cdot \frac{\max(\lvert S_f\rvert, \lvert S_t\rvert)}{S^{max}}
$$

```python
# src/nnopf/ybus.py — branch_flows
Sf = Vf * np.conj(Yf @ V)
St = Vt * np.conj(Yt @ V)

# src/nnopf/powerflow.py — line_loading
smax = np.maximum(np.abs(Sf), np.abs(St))
loading = np.where(sys.rate_a > 0, 100.0 * smax / sys.rate_a, np.nan)
```

---

## 2. 조류계산 — 뉴턴-랩슨

### 2.1 주입식

이 저장소 전체에서 가장 많이 쓰이는 한 줄입니다.

$$
S_i = V_i \overline{\left(\sum_k Y_{ik} V_k\right)}
\qquad\Longleftrightarrow\qquad
S = V \odot \overline{Y_{bus} V}
$$

```python
# src/nnopf/powerflow.py — sbus_from_V
def sbus_from_V(Ybus, V):
    return V * np.conj(Ybus @ V)
```

### 2.2 모선 종류가 미지수를 정합니다

| 종류 | 주어진 값 | 구할 값 | 코드 상수 |
|---|---|---|---|
| 슬랙 | $\lvert V\rvert,\ \theta$ | $P,\ Q$ | `SLACK = 3` |
| PV | $P,\ \lvert V\rvert$ | $Q,\ \theta$ | `PV = 2` |
| PQ | $P,\ Q$ | $\lvert V\rvert,\ \theta$ | `PQ = 1` |

미지수 벡터와 불일치(mismatch) 벡터:

$$
x = \begin{bmatrix}\theta_{PV,PQ} \\ \lvert V\rvert_{PQ}\end{bmatrix},
\qquad
f(x) = \begin{bmatrix}
P^{spec}_{PV,PQ} - P(x) \\ Q^{spec}_{PQ} - Q(x)
\end{bmatrix} = 0
$$

```python
# src/nnopf/powerflow.py — solve_power_flow
Psp = (Cg @ (Pg * on)) - sys.Pd
Qsp = (Cg @ (sys.Qg0 * on)) - sys.Qd
...
S = sbus_from_V(Ybus, V)
mis_p = Psp[pvpq] - np.real(S)[pvpq]
mis_q = Qsp[pq] - np.imag(S)[pq]
F = np.concatenate([mis_p, mis_q])
```

### 2.3 해석적 야코비안

수치미분을 쓰지 않습니다. 복소수로 한 번에 미분한 뒤 실수부/허수부를 뽑으면
네 블록이 전부 나옵니다 (MATPOWER `dSbus_dV` 와 같은 유도).

$$
\frac{\partial S}{\partial \theta} =
j\,\mathrm{diag}(V)\,\overline{\big(\mathrm{diag}(Y V) - Y\,\mathrm{diag}(V)\big)}
$$

$$
\frac{\partial S}{\partial \lvert V\rvert} =
\mathrm{diag}(V)\,\overline{Y\,\mathrm{diag}(V/\lvert V\rvert)}
+ \overline{\mathrm{diag}(YV)}\,\mathrm{diag}(V/\lvert V\rvert)
$$

```python
# src/nnopf/powerflow.py — dSbus_dV
Ibus = Ybus @ V
diagV, diagIbus = sp.diags(V), sp.diags(Ibus)
diagVnorm = sp.diags(V / np.abs(V))

dS_dVm = diagV @ np.conj(Ybus @ diagVnorm) + np.conj(diagIbus) @ diagVnorm
dS_dVa = 1j * diagV @ np.conj(diagIbus - Ybus @ diagV)
```

네 블록으로 자릅니다.

$$
J = \begin{bmatrix}
\partial P/\partial\theta & \partial P/\partial\lvert V\rvert \\
\partial Q/\partial\theta & \partial Q/\partial\lvert V\rvert
\end{bmatrix}
= \begin{bmatrix}
\Re[\partial S/\partial\theta]_{pvpq,pvpq} & \Re[\partial S/\partial \lvert V\rvert]_{pvpq,pq} \\
\Im[\partial S/\partial\theta]_{pq,pvpq} & \Im[\partial S/\partial \lvert V\rvert]_{pq,pq}
\end{bmatrix}
$$

```python
# src/nnopf/powerflow.py — _build_jacobian
J11 = np.real(dS_dVa[np.ix_(pvpq, pvpq)])   # dP/dTheta
J12 = np.real(dS_dVm[np.ix_(pvpq, pq)])     # dP/dVm
J21 = np.imag(dS_dVa[np.ix_(pq, pvpq)])     # dQ/dTheta
J22 = np.imag(dS_dVm[np.ix_(pq, pq)])       # dQ/dVm
return sp.bmat([[J11, J12], [J21, J22]], format="csr")
```

### 2.4 반복

$$
J\,\Delta x = f(x_k), \qquad x_{k+1} = x_k + \Delta x
$$

수렴 판정은 $\lVert f\rVert_\infty < 10^{-10}$ 입니다.

```python
# src/nnopf/powerflow.py — solve_power_flow
J = _build_jacobian(Ybus, V, pvpq, pq)
dx = spla.spsolve(J.tocsc(), F)
Va[pvpq] += dx[:npvpq]
Vm[pq] += dx[npvpq:]
V = Vm * np.exp(1j * Va)
```

> [!NOTE] 해석적 야코비안이라 5~6회에 끝납니다
> 수치미분이면 매 반복마다 $n$ 번의 함수평가가 더 듭니다. case118 에서
> 조류계산 한 건이 4.15 ms 인 것(§8 의 06 문서)은 이 덕입니다.

---

## 3. 최적조류계산 (AC-OPF)

### 3.1 결정변수와 목적함수

$$
x = [\;\theta_1 \dots \theta_{n_b},\; \lvert V\rvert_1 \dots \lvert V\rvert_{n_b},\;
P^g_1 \dots P^g_{n_g},\; Q^g_1 \dots Q^g_{n_g}\;]
$$

$$
\min_x \sum_{i=1}^{n_g}\left(c_{2,i}P_{g,i}^2 + c_{1,i}P_{g,i} + c_{0,i}\right),
\qquad [P_g\ \text{단위: MW}]
$$

비용 계수는 MW 기준인데 변수는 pu 라, **계수 쪽을 pu 로 환산**합니다.

```python
# src/nnopf/opf.py — solve_acopf
a = sys.cost_c2 * base**2          # * Pg_pu^2
b = sys.cost_c1 * base             # * Pg_pu
c = np.where(on, sys.cost_c0, 0.0)

def objective(x):
    pg = x[var.pg]
    return float(np.sum(a * pg**2 + b * pg + c)) / cost_scale

def objective_grad(x):
    g[var.pg] = (2.0 * a * x[var.pg] + b) / cost_scale
```

### 3.2 등식제약 — 조류방정식

**이 함수가 4단계에서 신경망이 대체하려는 대상입니다.** 그래서 따로 떼어
놓았습니다.

$$
\Delta S = V\overline{Y_{bus}V} - \big[(C_g P^g - P^d) + j(C_g Q^g - Q^d)\big] = 0
$$

```python
# src/nnopf/opf.py — power_flow_residual
V = Vm * np.exp(1j * Va)
S_calc = sbus_from_V(Ybus, V)
S_inj = (sys.Cg @ Pg - sys.Pd) + 1j * (sys.Cg @ Qg - sys.Qd)
d = S_calc - S_inj
return np.concatenate([np.real(d), np.imag(d)])
```

야코비안은 §2.3 을 발전기 열까지 넓힌 것입니다.

$$
\frac{\partial\,[\Delta P;\Delta Q]}{\partial x} =
\begin{bmatrix}
\Re[\partial S/\partial\theta] & \Re[\partial S/\partial\lvert V\rvert] & -C_g & 0 \\
\Im[\partial S/\partial\theta] & \Im[\partial S/\partial\lvert V\rvert] & 0 & -C_g
\end{bmatrix}
$$

```python
# src/nnopf/opf.py — power_flow_jacobian
return sp.bmat([
    [np.real(dS_dVa), np.real(dS_dVm), -Cg, Z],
    [np.imag(dS_dVa), np.imag(dS_dVm), Z, -Cg],
], format="csr")
```

### 3.3 부등식제약

$$
V^{min}\le \lvert V\rvert \le V^{max},\quad
P^{min}_g\le P^g\le P^{max}_g,\quad
Q^{min}_g\le Q^g\le Q^{max}_g
$$

$$
\lvert S_f\rvert^2 \le (S^{max})^2, \qquad \lvert S_t\rvert^2 \le (S^{max})^2
\quad(\text{선택})
$$

선로 조류 제약의 기울기에는 브랜치 조류의 편미분이 필요합니다
(MATPOWER `dSbr_dV`).

$$
\frac{\partial S_f}{\partial\theta} =
j\left(\overline{\mathrm{diag}(I)}\,C\,\mathrm{diag}(V)
- \mathrm{diag}(CV)\,\overline{Y\,\mathrm{diag}(V)}\right)
$$

```python
# src/nnopf/opf.py — _dSbr_dV
dS_dVa = 1j * (np.conj(diagI) @ C @ diagV - diagVbr @ np.conj(Y @ diagV))
dS_dVm = diagVbr @ np.conj(Y @ sp.diags(Vnorm)) \
       + np.conj(diagI) @ C @ sp.diags(Vnorm)
```

> [!CAUTION] SLSQP 를 쓰므로 이 등식제약 야코비안이 **조밀행렬**로 요구됩니다
> case118 이면 $236\times344$ 를 반복마다 만듭니다. 44초가 걸리는 원인이
> 여기이고, IPOPT 를 쓰면 10~100배 빨라집니다 (02 문서 §6.2).
> 논문에 "44초 vs 0.2 ms" 라고 쓰면 부정직한 비교입니다 (00 문서 §7.1).

---

## 4. 데이터 생성

### 4.1 운전점 샘플링

$$
P^d_i = P^{d,0}_i \cdot u_i,\quad u_i \sim \mathcal{U}(a_P, b_P), \qquad
Q^d_i = Q^{d,0}_i \cdot u'_i,\quad u'_i \sim \mathcal{U}(a_Q, b_Q)
$$

재생에너지는 용량 대비 비율로 뽑고, **총부하 대비 상한**으로 눌러 둡니다.

$$
p^{ren} \leftarrow p^{ren}\cdot\min\!\left(1,\ \frac{\kappa \sum_i P^d_i}{\sum p^{ren}}\right)
$$

```python
# src/nnopf/dataset.py — sample_scenario
Pd = sys.Pd * rng.uniform(cfg.p_load_lo, cfg.p_load_hi, sys.nb)
Qd = sys.Qd * rng.uniform(cfg.q_load_lo, cfg.q_load_hi, sys.nb)

p_ren = ren_capacity * rng.uniform(cfg.ren_lo, cfg.ren_hi, ren_gen.size)
cap = cfg.ren_share_cap * total_load
if p_ren.sum() > cap > 0:
    p_ren *= cap / p_ren.sum()
```

난수는 `default_rng([seed, index])` 로 파생시킵니다. **워커 수·청크 크기와
무관하게** 같은 index 는 항상 같은 시나리오가 됩니다.

### 4.2 급전 배분 — 여유 비례 재배분

목표 $T$ 를 발전기에 나누되 $[P^{min}, P^{max}]$ 를 지켜야 합니다. 비례배분 후
clip 만 하면 잘린 몫이 **전부 슬랙 한 모선**으로 갑니다.

$$
p^{(0)} = \mathrm{clip}\!\left(T\,w\odot\epsilon,\ P^{min},\ P^{max}\right),
\qquad w = \frac{P^{max}}{\sum P^{max}}
$$

$$
g^{(k)} = T - \textstyle\sum p^{(k)},\qquad
h^{(k)} = \begin{cases} P^{max}-p^{(k)} & g^{(k)}>0\\ p^{(k)}-P^{min} & g^{(k)}\le 0\end{cases}
$$

$$
p^{(k+1)} = \mathrm{clip}\!\left(p^{(k)} + g^{(k)}\frac{h^{(k)}}{\sum h^{(k)}},\ P^{min},\ P^{max}\right)
$$

```python
# src/nnopf/dataset.py — allocate_dispatch
alloc = np.clip(target * w * jitter, pmin, pmax)
for _ in range(max_iter):
    gap = target - float(alloc.sum())
    if abs(gap) < 1e-10:
        break
    headroom = (pmax - alloc) if gap > 0 else (alloc - pmin)
    total = float(headroom.sum())
    if total < 1e-12:
        break
    alloc = np.clip(alloc + gap * headroom / total, pmin, pmax)
```

> [!IMPORTANT] 이걸 안 하면 case300 이 100% 발산합니다
> 잘린 몫이 수천 MW 에 이르면 슬랙 한 모선이 그걸 다 공급해야 하고,
> 조류계산이 발산하거나 수렴해도 물리적으로 말이 안 되는 운전점이 됩니다.
> 실제 급전이 하는 일과 같은 계산입니다.

### 4.3 float32 내림 — 잔차 바닥을 만드는 함정

입력을 float32 로 **저장하기 전에** float32 로 내린 뒤 그 값으로 풉니다.

$$
\tilde x = \mathrm{float64}\big(\mathrm{float32}(x)\big) \quad\text{로 풀고 저장}
$$

```python
# src/nnopf/dataset.py — sample_scenario
f32 = lambda a: a.astype(np.float32).astype(np.float64)
return Scenario(Pd=f32(Pd), Qd=f32(Qd), p_ren=f32(p_ren_bus), ...)
```

안 하면 "저장된 입력"과 "라벨을 만든 입력"이 미세하게 달라지고, 그 차이가
$Y_{bus}$ 를 거치며 $\max\lvert Y\rvert$ 배로 증폭돼 case300 에서 $2.5\times10^{-4}$ pu
까지 커집니다 — 재려는 신경망 잔차와 같은 자릿수입니다.

---

## 5. 대체모델 (MLP)

### 5.1 무엇을 예측하지 **않는가**

| 값 | 예측? | 이유 |
|---|---|---|
| 슬랙 $\theta$ | ❌ 계통 기준값 | 위상 기준점 |
| PV·슬랙 $\lvert V\rvert$ | ❌ 설정값 그대로 | 발전기가 유지 (데이터 확인: 오차 7e-16) |
| PQ $\lvert V\rvert$ | ✅ | 조류방정식이 결정 |
| 비슬랙 $\theta$ | ✅ | 조류방정식이 결정 |
| $P, Q$ | ❌ 닫힌 식 | $S = V\overline{YV}$ |

case30 이면 출력이 60개가 아니라 **53개** ($\lvert V\rvert$@PQ 24 + $\theta$@비슬랙 29)
이고, 나머지 7개는 근사가 아니라 **정확한 값**이 들어갑니다.

```python
# src/nnopf/models.py — IOLayout
self.pq = np.flatnonzero(sys.bus_type == PQ)
self.nonslack = np.flatnonzero(sys.bus_type != SLACK)
self.va_ref = np.zeros(self.nb)
self.va_ref[sys.bus_type == SLACK] = sys.Va0[sys.bus_type == SLACK]
self.in_dim = 4 * self.nb + self.nl
self.out_dim = len(self.pq) + len(self.nonslack)
```

> [!CAUTION] 슬랙 기준위상을 0 이라고 가정하면 안 됩니다
> pandapower `case118` 은 슬랙(모선 68)의 기준위상이 **30°** 입니다. 이걸
> 0 으로 두면 case118 결과가 통째로 오염됩니다 (06 문서 §7.4).

### 5.2 입력과 정규화

$$
x = [\,P^d \mid Q^d \mid p^{ren} \mid p^{gen} \mid s\,] \in \mathbb{R}^{4n_b + n_l},
\qquad s_l \in \{0, 1\}
$$

$$
z = \frac{x - \mu_{tr}}{\sigma_{tr}}, \qquad
\mu_l = 0,\ \sigma_l = 1 \ \ (\text{선로상태 열})
$$

```python
# src/nnopf/train.py — input_stats
mean = X[tr].mean(0)
std = X[tr].std(0)
std = np.where(std > 1e-8, std, 1.0)   # 상수 열: 중심화만, 확대 금지
mean[n_phys:] = 0.0                     # 선로상태: 손대지 않는다
std[n_phys:] = 1.0
```

> [!CAUTION] 선로상태를 표준화하면 미지 N-1 실험이 통째로 망가집니다
> 어떤 선로가 학습 분할에서 한 번도 고장 안 나면 그 열의 $\sigma = 0$ 입니다.
> 하한 $10^{-6}$ 을 씌워 나누면 그 선로가 고장 난 **시험 표본**에서 정규화
> 입력이 $10^{6}$ 으로 폭주합니다. 그리고 미지 N-1 분할은 **정의상** 시험용
> 고장이 학습에 안 나오게 만듭니다. case30·400표본에서 41개 선로 중 12개가
> 학습 분할 상수였고 그중 4개가 시험에서 변했습니다.

### 5.3 전압 출력 헤드

**scaled** (기본값, P1 §5.2.2 의 스케일링 인자):

$$
\lvert V\rvert = V^{lo} + \sigma(z_{vm})\,(V^{hi} - V^{lo}), \qquad \sigma(\cdot)\in(0,1)
$$

**raw** (절제용):

$$
\lvert V\rvert = \mu_{vm} + z_{vm}\,\sigma_{vm}
$$

위상은 항상 표준화 역변환입니다.

$$
\theta = \mu_\theta + z_{va}\,\sigma_\theta
$$

```python
# src/nnopf/models.py — PowerFlowMLP.forward
if self.spec.vm_head == "scaled":
    vm = self.vm_lo + torch.sigmoid(vm_raw) * (self.vm_hi - self.vm_lo)
else:
    vm = self.vm_lo + vm_raw * self.vm_hi   # raw: (평균, 표준편차)
va = self.va_mean + va_raw * self.va_std

Vm = self.v_set.expand(b, self.nb).index_copy(1, self.pq_idx, vm)
Va = self.va_ref.expand(b, self.nb).index_copy(1, self.va_idx, va)
```

> [!CAUTION] 박스를 계통 전압한계 $[V^{min}, V^{max}]$ 로 잡으면 안 됩니다
> 조류계산은 전압한계를 강제하지 않으므로 **라벨이 그 밖으로 나갑니다**
> (case30 기준 PQ 표본의 0.63% 가 0.95 pu 미만). 모델이 도달할 수 없는 정답이
> 생겨 계통적 오차가 남습니다. 그래서 박스는 **학습 데이터 범위 + 5% 여유**로
> 잡습니다.
>
> $$V^{lo} = \min_{tr}\lvert V\rvert - \delta,\quad V^{hi} = \max_{tr}\lvert V\rvert + \delta,\quad \delta = 0.05\,(\max-\min)$$

### 5.4 선형 지름길

$$
h = \mathrm{MLP}(z) + W_{skip}\,z + b_{skip}
$$

조류방정식은 정상 운전 영역에서 거의 선형입니다 (case30 최소제곱 $R^2 = 0.987$).
지름길이 없으면 신경망이 **선형 부분을 재현하는 데만 용량과 epoch 을 다 씁니다.**

```python
# src/nnopf/models.py — PowerFlowMLP.forward
z = (x - self.in_mean) / self.in_std
h = self.net(z)
if self.skip is not None:
    h = h + self.skip(z)
```

### 5.5 지름길을 최소제곱 해에서 출발시키기 (`--skip-init lstsq`)

지름길이 0 에서 시작하면 그 분업은 **구조가 아니라 희망**입니다. 선형 해는
이미 닫힌 형태로 있으니 거기서 출발시킵니다. 핵심은 **모델 자신의 출력
공간**, 즉 헤드 **직전** 자리에서 푼다는 것입니다.

$$
t^{vm} = \mathrm{logit}\!\left(\frac{\lvert V\rvert - V^{lo}}{V^{hi} - V^{lo}}\right)
= \log\frac{u}{1-u}, \qquad
t^{va} = \frac{\theta - \mu_\theta}{\sigma_\theta}
$$

$$
W^\star = \arg\min_W \lVert A W - T\rVert_F^2,
\qquad A = [\,z_{tr}[:, \text{keep}] \mid \mathbf{1}\,]
$$

```python
# src/nnopf/train.py — init_skip_lstsq
u = np.clip((vm - lo) / np.maximum(hi - lo, 1e-12), 1e-6, 1 - 1e-6)
t_vm = np.log(u / (1.0 - u))          # 시그모이드의 역함수
t_va = (Va[idx][:, vi] - am) / asd

keep = np.flatnonzero(z[tr].std(0) > 0)     # 상수열 제거
A = np.c_[z[tr][:, keep], np.ones(len(tr))]
for rc in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
    W, *_ = np.linalg.lstsq(A, T, rcond=rc)
    mse = float(((Av @ W - Tv) ** 2).mean())   # 절단선은 val 로 고른다
```

> [!IMPORTANT] 상자에 5% 여유가 있어서 로짓이 발산하지 않습니다
> 라벨이 항상 상자 안쪽이므로 $u \in (0,1)$ 이 보장됩니다. 여유가 없으면
> $u = 0$ 또는 $1$ 에서 $\mathrm{logit}$ 이 $\pm\infty$ 가 됩니다.
>
> 그리고 **상수열을 반드시 뺍니다.** 안 빼면 초기화 직후 P/부하가 7.3% (뺀 뒤
> 3.0%) 였습니다 — §6.1 의 설계행렬 조건수 문제가 여기에도 그대로 있습니다.

---

## 6. 그래프 어텐션 대체모델 (GAT)

MLP 에게 선로 상태는 입력 끝에 붙은 **0/1 비트**일 뿐입니다. 그 비트가 계통
그래프의 어느 간선인지 모델에 정보가 없습니다. 그래서 **선로를 지우는 것이
곧 메시지 경로를 지우는 것**이 되게 만듭니다.

### 6.1 노드·엣지 특징

$$
\text{node}_i = [\,P^d_i,\ Q^d_i,\ p^{ren}_i,\ p^{gen}_i,\
V^{set}_i,\ \mathbb{1}_{slack},\ \mathbb{1}_{PV},\ \mathbb{1}_{PQ},\ \mathrm{id}_i\,]
$$

$$
e_{ij} = [\,r,\ x,\ b_{sh},\ \tau,\ s,\ g,\ b\,] \in \mathbb{R}^7
$$

$\mathrm{id}_i \in \mathbb{R}^{16}$ 은 **모선마다 학습되는 고유 벡터**입니다.
GNN 은 정의상 노드를 구분하지 않지만(어느 계통에도 쓰려고), 우리는 **고정된
하나의 계통**을 다루므로 모선 5와 27이 다른 자리라는 걸 알려 줘야 합니다.

엣지 특징은 자릿수가 크게 벌어져서 **부호 보존 로그** 후 표준화합니다.

$$
\tilde e = \frac{\mathrm{sgn}(e)\log(1+\lvert e\rvert) - \mu_e}{\sigma_e},
\qquad (\mu_e, \sigma_e)_{\text{status}} = (0, 1)
$$

```python
# src/nnopf/gnn.py — PowerFlowGAT.__init__ / _unfold
t = np.sign(base) * np.log1p(np.abs(base))
mu, sd = t.mean(0), t.std(0)
mu[4], sd[4] = 0.0, 1.0                # status 열은 건드리지 않는다
...
ea[:, :, 4] = alive                    # 상정사고를 status 열에 반영
ea = (torch.sign(ea) * torch.log1p(ea.abs()) - self.e_mean) / self.e_std
```

### 6.2 어텐션 한 층

$$
\ell_{ij} = a^\top\,\mathrm{LeakyReLU}_{0.2}\!\left(W_s h_i + W_d h_j + W_e \tilde e_{ij}\right)
$$

**끊긴 선로는 소프트맥스 전에 로짓을 눌러 막습니다** (게이팅, 이 모델의 존재 이유):

$$
\ell_{ij} \leftarrow \ell_{ij} + (1 - \mathrm{alive}_{ij})\cdot(-10^9)
$$

$$
\alpha_{ij} = \mathrm{softmax}_{j\,:\,(i\to j)}\big(\ell_{ij}\big)
\quad\text{(받는 노드별 정규화)}
$$

$$
h'_j = \mathrm{LayerNorm}\!\left(h_j + W_o \Big\Vert_{k=1}^{H}
\sum_{i\in\mathcal{N}(j)}\alpha^{(k)}_{ij}\big(W_v h_i + W_{ve}\tilde e_{ij}\big)\right)
$$

```python
# src/nnopf/gnn.py — EdgeGAT.forward
logit = (F.leaky_relu(hs + hd + he, 0.2) * self.att).sum(-1)   # (B, E, H)
if self.gate:
    logit = logit + (1.0 - alive).unsqueeze(-1) * NEG          # NEG = -1e9

# 받는 노드별 소프트맥스 — scatter 로 직접
idx = d.view(1, -1, 1).expand(B, -1, H)
big = torch.full((B, N, H), NEG, ...).scatter_reduce(1, idx, logit, "amax", ...)
ex = (logit - big.gather(1, idx)).exp()
den = torch.zeros(B, N, H, ...).scatter_add(1, idx, ex)
alpha = ex / den.gather(1, idx).clamp(min=1e-16)

msg = (self.val(x)[:, s].view(B, -1, H, D)
       + self.val_e(ea).view(B, -1, H, D)) * alpha.unsqueeze(-1)
agg = torch.zeros(B, N, H, D, ...).scatter_add(1, d.view(...).expand(...), msg)
return self.norm(x + self.out(agg.reshape(B, N, H * D)))
```

> [!NOTE] `-inf` 대신 `-1e9` 를 씁니다
> 어떤 모선의 인접 선로가 **전부** 끊기면 소프트맥스 분모가 0 이 되고
> `-inf` 는 NaN 을 만듭니다. 게다가 자기 자신으로 가는 엣지를 하나씩 붙여
> (절대 안 끊김) 그 경우에도 값이 살아 있게 했습니다.
> 최댓값 빼기(`amax`)도 같은 이유의 수치 안정화입니다.

### 6.3 합 집계 (`--agg sum`) — 절제용

조류방정식은 이웃 기여의 **합**이지 평균이 아닙니다. $YV$ 어디에도 정규화가
없습니다. 소프트맥스는 $\sum_i\alpha_{ij}=1$ 을 강제해서, 선로 5개가 붙은
모선과 1개가 붙은 모선이 **같은 크기의 메시지**를 받게 만듭니다.

$$
\alpha_{ij} = \mathrm{sigmoid}(\ell_{ij})\cdot \mathrm{alive}_{ij}
$$

```python
# src/nnopf/gnn.py — EdgeGAT.forward
if self.agg == "sum":
    alpha = torch.sigmoid(logit) * alive.unsqueeze(-1)
```

### 6.4 모선별 출력 헤드

공유 헤드는 "모든 모선이 같은 함수"를 강제하는데, 전압은 모선마다 다른
사상입니다.

$$
\mathrm{out}_{i} = h_i^\top W^{head}_i + b^{head}_i \in \mathbb{R}^2,
\qquad W^{head} \in \mathbb{R}^{n_b\times d\times 2}
$$

```python
# src/nnopf/gnn.py — PowerFlowGAT.forward
out = torch.einsum("bnd,ndk->bnk", h, self.head_w) + self.head_b
vm_raw = out[:, self.pq_idx, 0]
va_raw = out[:, self.va_idx, 1]
```

출력 헤드(scaled/raw)와 선형 지름길은 MLP 와 **완전히 같습니다** — 그래야
사과 대 사과 비교가 됩니다.

---

## 7. 손실 함수

### 7.1 지도 손실 — 표준화 공간에서

$$
\mathcal{L}_{sup} =
\frac{1}{\lvert PQ\rvert}\sum_{i\in PQ}\left(\frac{\hat V_{m,i} - V_{m,i}}{\sigma_{V_m,i}}\right)^2
+ \frac{1}{\lvert \overline{S}\rvert}\sum_{i\notin S}\left(\frac{\hat\theta_i - \theta_i}{\sigma_{\theta,i}}\right)^2
$$

```python
# src/nnopf/train.py — supervised_loss
dv = (Vm[:, model.pq_idx] - Vm_true[:, model.pq_idx]) * model.vm_w
da = (Va[:, model.va_idx] - Va_true[:, model.va_idx]) * model.va_w
return (dv**2).mean() + (da**2).mean()
# vm_w = 1/max(std, 1e-6),  va_w = 1/max(std, 1e-6)
```

> [!IMPORTANT] 표준화가 없으면 weight decay 가 학습을 죽입니다
> 라벨 분산이 $\lvert V\rvert$ 8.1e-05, $\theta$ 8.1e-04 라 원단위 MSE 는
> $10^{-4}$ 규모이고 기울기도 그 규모입니다. PyTorch `Adam(weight_decay=)` 는
> L2 를 **기울기에 더하는** 방식이라, $wd\cdot\lvert w\rvert$ 가 과제 기울기와
> 맞먹습니다.
>
> | lr | wd | train MSE |
> |---|---|---|
> | 5e-4 | 0 | 9.3e-06 |
> | 5e-4 | 1e-5 | 1.0e-04 ← 갇힘 |
> | 2e-3 | 0 | 2.2e-06 |
> | 2e-3 | 1e-5 | 1.0e-04 ← 갇힘 (lr 과 무관) |
>
> 덤으로 $\lvert V\rvert$ 와 $\theta$ 의 분산이 10배 차이라 원단위로는 $\theta$ 가
> 손실을 지배했는데, 표준화하면 둘이 대등해집니다.

### 7.2 물리 잔차 — 실수 산술 전개

복소 autograd 를 피하고 속도도 얻으려고 $V = e + jf$ 로 풀어 씁니다.

$$
e = \lvert V\rvert\cos\theta, \qquad f = \lvert V\rvert\sin\theta
$$

$$
YV = (Ge - Bf) + j(Gf + Be)
$$

$$
P = e(Ge - Bf) + f(Gf + Be), \qquad
Q = f(Ge - Bf) - e(Gf + Be)
$$

```python
# src/nnopf/physics_torch.py — ACPhysics._yv / injection
yv_re = e @ self.G.T - f @ self.B.T
yv_im = f @ self.G.T + e @ self.B.T
...
return e * yv_re + f * yv_im, f * yv_re - e * yv_im
```

잔차는 **지정값이 있는 자리에만** 겁니다.

| 모선 | 잔차 | 이유 |
|---|---|---|
| PQ | $\Delta P,\ \Delta Q$ | P, Q 모두 지정값 |
| PV | $\Delta P$ 만 | Q 는 조류방정식이 정하는 종속변수 |
| 슬랙 | 없음 | P, Q 모두 종속변수 |

$$
\Delta P = (P^{sp} - P)\odot m_P, \qquad \Delta Q = (Q^{sp} - Q)\odot m_Q
$$

$$
P^{sp} = p^{gen} + p^{ren} - P^d, \qquad Q^{sp} = -Q^d
$$

```python
# src/nnopf/physics_torch.py — ACPhysics.residual
P, Q = self.injection(Vm, Va, outage)
return (p_spec - P) * self.mask_p, (q_spec - Q) * self.mask_q
# mask_p = (bus_type != SLACK),  mask_q = (bus_type == PQ)
```

> [!CAUTION] 고정 병렬 소자를 $Q^{sp}$ 에 넣으면 이중계산입니다
> 커패시터·리액터는 $Y_{bus}$ 대각에 이미 들어 있어 계산된 $\hat Q$ 쪽에
> 반영됩니다. 양쪽에 넣으면 두 번 세게 됩니다 (2단계 검증에서 겪은 이슈).

### 7.3 N-1 을 4개 성분 보정으로

상정사고마다 $Y_{bus}$ 를 통째로 들고 있으면 case118 기준 178 × 118² × 4바이트
= **20 MB** 이고 배치마다 gather 해야 합니다. 대신 기저 $Y_{bus}$ 하나만 두고
고장 선로가 기여하던 4개 성분만 뺍니다.

$$
(YV)_{f} \leftarrow (YV)_{f} - \big(Y_{ff}V_f + Y_{ft}V_t\big),\qquad
(YV)_{t} \leftarrow (YV)_{t} - \big(Y_{tf}V_f + Y_{tt}V_t\big)
$$

정확히 같은 값이고 표본당 $O(1)$ 입니다.

```python
# src/nnopf/physics_torch.py — ACPhysics._yv
fr1, fi1 = cmul(self.yff[0][ln], self.yff[1][ln], ep, fp)
fr2, fi2 = cmul(self.yft[0][ln], self.yft[1][ln], eq, fq)
tr1, ti1 = cmul(self.ytf[0][ln], self.ytf[1][ln], ep, fp)
tr2, ti2 = cmul(self.ytt[0][ln], self.ytt[1][ln], eq, fq)

corr_re = zero.index_put((hit, p), -(fr1 + fr2), accumulate=True)
corr_re = corr_re.index_put((hit, q), -(tr1 + tr2), accumulate=True)
```

### 7.4 물리 손실 — 모선 종류별로 나눠 평균

$$
\mathcal{L}_{phys} =
\frac{1}{\lvert\mathcal{N}_{PQ}\rvert}\sum_{i\in PQ}\left(\Delta P_i^2 + \Delta Q_i^2\right)
+ \frac{1}{\lvert\mathcal{N}_{PV}\rvert}\sum_{i\in PV}\Delta P_i^2
$$

전체를 한 번에 평균 내면 모선 수 비율에 따라 PV 항의 비중이 **계통마다**
달라집니다.

```python
# src/nnopf/physics_torch.py — ACPhysics.loss
n_pq = self.mask_q.sum().clamp(min=1.0)
n_pv = (self.mask_p - self.mask_q).sum().clamp(min=1.0)
pq_term = ((dP * self.mask_q) ** 2 + dQ**2).sum(-1) / n_pq
pv_term = ((dP * (self.mask_p - self.mask_q)) ** 2).sum(-1) / n_pv
return (pq_term + pv_term).mean()
```

### 7.5 무차원화

$$
\Delta P \leftarrow \frac{\Delta P}{\max(s_P,\ 10^{-3})}, \qquad
\Delta Q \leftarrow \frac{\Delta Q}{\max(s_Q,\ 10^{-3})}
$$

```python
# src/nnopf/physics_torch.py — set_scale / loss
self.p_scale = p_scale.to(self.dtype).clamp(min=1e-3)
...
if self.p_scale is not None:
    dP, dQ = dP / self.p_scale, dQ / self.q_scale
```

### 7.6 λ 스케줄과 상대 가중치

**워밍업-램프** (P2 §4.1). 초기에는 전압 예측이 엉망인데 그 값을 조류방정식에
넣으면 잔차 기울기가 폭주합니다.

$$
\lambda(t) = \begin{cases}
0 & t < T_w \\[2pt]
\lambda\,\dfrac{t - T_w + 1}{T_r} & T_w \le t < T_w + T_r \\[6pt]
\lambda & t \ge T_w + T_r
\end{cases}
$$

```python
# src/nnopf/train.py — lambda_at
if cfg.lam <= 0 or epoch < cfg.lam_warmup:
    return 0.0
if epoch >= cfg.lam_warmup + cfg.lam_ramp:
    return cfg.lam
return cfg.lam * (epoch - cfg.lam_warmup + 1) / cfg.lam_ramp
```

**상대 가중치.** 램프 시작 시점에 두 항의 비를 한 번 재서 나눠 줍니다.

$$
\rho = \frac{\mathcal{L}_{phys}(t_0)}{\mathcal{L}_{sup}(t_0)},
\qquad
\mathcal{L} = \mathcal{L}_{sup} + \frac{\lambda(t)}{\rho}\,\mathcal{L}_{phys}
$$

그러면 $\lambda=1$ 이 "그 순간 두 항이 같은 크기" 를 뜻하게 됩니다.

```python
# src/nnopf/train.py — train
phys_ref = max(p0, 1e-12) / max(s0, 1e-12)
...
loss = sup
if lam > 0:
    loss = loss + (lam / phys_ref) * b.physics.loss(
        Vm, Va, b.p_spec[j], b.q_spec[j], b.outage[j])
```

> [!CAUTION] λ 는 정규화 방식이 같아야만 논문 사이에 옮길 수 있습니다
> 우리 지도 손실은 표준화 공간이라 $O(1)$ 인데 물리 잔차를 원단위 $\mathrm{pu}^2$
> 로 두면 같은 숫자 λ 가 전혀 다른 뜻이 됩니다. 두 항의 절대 크기 비가 계통마다
> **case30 2.2e3배 · case118 2.1e7배**로 4자리씩 다릅니다.
>
> 실제로 P2 의 λ=3e-3 을 그대로 썼더니 우리 실험에서 **가장 나쁜 설정**이
> 됐습니다 (03 문서 §3.1). 환산 후 최적은 두 계통 모두 **λ=5e-3** 입니다.

### 7.7 야코비안 가중 (`--jac-alpha`, 효과 없음으로 결론)

같은 크기의 전압 오차라도 **어느 모선에 있느냐**에 따라 전력 잔차 기여가
1.7배까지 달라집니다. 그 민감도로 손실을 가중해 봤습니다.

$$
s_j = \left\lVert \frac{\partial r}{\partial x_j}\right\rVert_2,
\qquad r = \big[\Delta P_{\text{비슬랙}};\ \Delta Q_{PQ}\big]
$$

두 블록은 단위가 다르므로 **각각 RMS 로 나눠** 대등하게 만든 뒤 합칩니다.

$$
s_j = \sqrt{\sum_{i\in\overline{S}}\left(\frac{\Re[\partial S/\partial x_j]_i}{\mathrm{rms}_P}\right)^2
+ \sum_{i\in PQ}\left(\frac{\Im[\partial S/\partial x_j]_i}{\mathrm{rms}_Q}\right)^2}
$$

$$
m_j = \frac{s_j^\alpha}{\sqrt{\overline{s^{2\alpha}}}}
\quad\Longrightarrow\quad \overline{m^2} = 1\ \ (\text{손실 크기 불변})
$$

```python
# src/nnopf/train.py — jacobian_weights
def _sens(dS):
    blk_p = np.real(M[non_slack, :])
    blk_q = np.imag(M[pq, :])
    rp = float(np.sqrt((blk_p**2).mean())) or 1.0
    rq = float(np.sqrt((blk_q**2).mean())) or 1.0
    return np.sqrt(((blk_p / rp) ** 2).sum(0) + ((blk_q / rq) ** 2).sum(0))

def _norm(s):
    m = np.power(np.maximum(s, 1e-12), float(alpha))
    rms = float(np.sqrt((m**2).mean()))
    return (m / rms).astype(np.float32)
```

$\alpha = 0$ 이면 곱수가 **전부 정확히 1** 이라 기존 손실과 같습니다 — 한 변수
실험이 되도록 만든 대조군입니다. 결과는 06 문서 §6.2: **효과 없음.**

---

## 8. 물리 기반 비교군

셋의 성격이 다르다는 점이 표를 읽을 때 중요합니다.

| | 무엇을 보는가 | 데이터 |
|---|---|---|
| 최소제곱 선형 | 학습 분할을 보고 계수를 맞춤 | 필요 |
| DC 조류계산 | $\lvert V\rvert=1$, 손실 0, Q 무시 | 불필요 |
| 야코비안 선형화 | 기저해 근처 1차 테일러 전개 | 불필요 |

### 8.1 최소제곱 선형

$$
W^\star = \arg\min_W \lVert AW - Y\rVert_F^2,
\qquad A = [\,X_{tr}[:,\text{keep}] \mid \mathbf{1}\,]
$$

절단선 $\rho$(rcond)는 **검증 분할의 표준화 지도손실**로 고릅니다 — 신경망의
조기 종료 기준과 같게 맞추려고요.

```python
# src/nnopf/baselines.py — fit_linear
keep = np.flatnonzero(X.std(0) > 0)      # 상수열 제거
A, Y = np.c_[X[:, keep], np.ones(len(X))], tgt(train_idx)
for rc in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
    W, *_ = np.linalg.lstsq(A, Y, rcond=rc)
    loss = float((((Av @ W - Yv) * w) ** 2).mean())
```

> [!CAUTION] 상수열을 빼지 않으면 **컴퓨터마다 다른 답**이 나옵니다
> case118 은 658 열 중 **239 열**이 정보가 0 입니다 (부하 없는 모선의 $P^d$,
> 발전기 없는 모선의 $p^{gen}$, 상정사고에서 제외된 선로의 status).
> 그대로 두면 설계행렬 조건수가 $2.7\times10^{58}$ 이 됩니다.
>
> | rcond | Vm MAE | P/부하 |
> |---|---|---|
> | 1e-10 이하 | 4.79e-05 | 30.80 % |
> | 1e-08 이상 | 3.53e-05 | 0.90 % |
>
> numpy 기본값(`rcond=None`)은 $6.2\times10^{-9}$ 로 **그 전환 구간 한가운데**
> 입니다. 같은 데이터·같은 코드로 두 컴퓨터에서 30.80% 와 22.99% 가 나왔습니다.

### 8.2 DC 조류계산

세 가지를 버립니다 — 전압 크기($\lvert V\rvert = 1$), 손실($r = 0$), 무효전력(Q 방정식 없음).

$$
P = B'\theta, \qquad
B'_{ik} = -\frac{1}{x_{ik}\tau_{ik}},\quad
B'_{ii} = \sum_{k}\frac{1}{x_{ik}\tau_{ik}}
$$

$$
\theta_{\overline{S}} = \big(B'_{\overline{S},\overline{S}}\big)^{-1} P_{\overline{S}} + \theta_{ref}
$$

```python
# src/nnopf/baselines.py — _susceptance_matrix / DCPowerFlow
b = alive / (sysm.br_x * sysm.br_tap)     # 탭비까지는 반영한다
vals = np.concatenate([b, b, -b, -b])
...
self._cache[out] = (spla.factorized(B[ns][:, ns].tocsc()), ns)
...
Va[i, ns] = solve(Psp[i, ns]) + ref
```

상정사고마다 $B'$ 가 달라지므로 **고장 종류별로 한 번씩만** LU 분해해 재사용
합니다 (case118 이면 178가지).

### 8.3 야코비안 선형화

뉴턴-랩슨이 **매번 새로 만드는** 야코비안을 기저 케이스에서 **한 번만** 만들어
재사용합니다. 반복이 없으니 선형 풀이 한 번으로 끝납니다.

$$
\begin{bmatrix}\Delta\theta \\ \Delta\lvert V\rvert\end{bmatrix}
= J(V_0)^{-1}
\begin{bmatrix}P^{sp} - P_0 \\ Q^{sp} - Q_0\end{bmatrix}
$$

```python
# src/nnopf/baselines.py — JacobianLinear
pf = solve_power_flow(sysm, tol=1e-10)      # 기저해를 진짜로 한 번 푼다
J = _build_jacobian(Ybus, pf.V, pvpq, pq)
S0 = sbus_from_V(Ybus, pf.V)
self._cache[out] = (spla.factorized(J.tocsc()), pf.Vm, pf.Va, ...)
...
F = np.concatenate([Psp[i, pvpq] - p0[pvpq], Qsp[i, pq] - q0[pq]])
dx = solve(F)
Va[i, pvpq] += dx[: len(pvpq)]
Vm[i, pq] += dx[len(pvpq):]
```

> [!NOTE] Bolognani & Zampieri(2015) 그대로는 아닙니다
> 원 고정점 선형화는 배전계통(방사형·불평형)을 겨냥한 형태라 식이 조금
> 다릅니다. 여기 구현한 것은 **같은 아이디어를 송전계통 표준형(극좌표
> 야코비안)으로 옮긴 것**입니다. 논문에 쓸 때 이 점을 밝혀야 합니다.

---

## 9. 평가 지표

전부 `train.evaluate` 한 곳에서 나오고, **잔차는 반드시 float64** 입니다.

$$
\mathrm{MAE}_{V_m} = \frac{1}{N\,n_b}\sum_{s,i}\lvert \hat V_{m}^{(s)} - V_m^{(s)}\rvert_i,
\qquad
\mathrm{MAE}_{\theta} \ \text{도 같은 꼴}
$$

$$
\Delta P = \frac{1}{N}\sum_{s}\sum_{i}\lvert\Delta P^{(s)}_i\rvert \ \ [\mathrm{pu}],
\qquad
\Delta Q \ \text{도 같은 꼴}
$$

**주 지표 — 부하 대비 유효전력 잔차** (P2 Table 6/7 과 맞춤):

$$
\text{P/부하}\ [\%] = 100\cdot
\frac{\frac{1}{N}\sum_s \sum_i \lvert\Delta P^{(s)}_i\rvert}
{\frac{1}{N}\sum_s \sum_i \lvert P^{sp,(s)}_i\rvert}
$$

```python
# src/nnopf/train.py — evaluate
load = b.p_spec[ii].double().cpu().abs().sum(-1).mean().clamp(min=1e-9)
...
"p_over_load_pct": (dP.sum(-1).mean() / load).item() * 100,
```

**전압 한계 위반률:**

$$
\text{위반}\ [\%] = 100\cdot\frac{1}{N n_b}
\left\lvert\left\{(s,i):\ \hat V^{(s)}_{m,i} < V^{min}_i - \epsilon
\ \lor\ \hat V^{(s)}_{m,i} > V^{max}_i + \epsilon\right\}\right\rvert,
\quad \epsilon = 10^{-6}
$$

```python
# src/nnopf/train.py — evaluate
VLIM_TOL = 1e-6
Vmin = torch.as_tensor(b.sys.Vmin, dtype=torch.float64) - VLIM_TOL
Vmax = torch.as_tensor(b.sys.Vmax, dtype=torch.float64) + VLIM_TOL
viol = ((Vm_all < Vmin) | (Vm_all > Vmax)).double()
```

> [!CAUTION] 허용오차 $\epsilon$ 이 없으면 이 지표가 아무것도 재지 않습니다
> case118 슬랙(모선 68)의 상자는 $[1.0349999999,\ 1.0350000001]$ 로 폭이
> $2\times10^{-10}$ 인데, float32 는 1.035 를 1.03499997 로밖에 못 씁니다.
> 상자 폭이 float32 해상도(1.035 근처에서 $1.2\times10^{-7}$)보다 **600배 좁으니
> 어떤 모델이든 항상 위반**이었고, 지표가 정확히 $100/118 = 0.8475\%$ 에
> 못박혀 있었습니다 (06 문서 §7.10).
>
> 고친 뒤 실제 위반은 미지 N-1 에서 **0.00%** 입니다.

> [!IMPORTANT] 잔차를 float32 로 재면 안 됩니다
> 전압 오차는 $Y_{bus}$ 를 거치며 $\max\lvert Y\rvert$ 배로 증폭됩니다. float32
> 로 재면 **측정 바닥**이 재려는 값과 같은 자릿수가 됩니다. 그래서 예측을
> CPU float64 로 내린 뒤에 잽니다.
>
> 그리고 float64 판은 **GPU 로 옮기지 않습니다** — 소비자용 GeForce 는 배정밀도
> 처리율이 단정밀도의 1/64 입니다.

---

## 10. 최적화

| 항목 | 식 / 값 | 코드 |
|---|---|---|
| 옵티마이저 | **AdamW**(분리형 감쇠), `weight_decay=1e-5` | `torch.optim.AdamW` |
| 기울기 클리핑 | $\lVert g\rVert_2 \le 2$ | `clip_grad_norm_(..., cfg.grad_clip)` |
| 학습률 스케줄 | $\eta \leftarrow 0.5\,\eta$ (`lr_patience`=30 동안 개선 없으면) | `ReduceLROnPlateau(factor=0.5, patience=lr_patience)` |
| 조기 종료 | `patience`(기본 200) epoch 개선 없으면 중단 | `cfg.patience` |
| 모델 선택 | `crit` = 검증 P/부하 % (`phys`) 또는 검증 지도손실 (`loss`) | `cfg.select` |
| 스케줄러 입력 | `crit` 또는 검증 지도손실을 따로 고를 수 있음 | `cfg.sched_on` |

검증 지표 둘은 이렇게 정의됩니다 (`validate`).

$$
\mathrm{val} = \frac{1}{N_{val}}\sum_{s\in val}\mathcal{L}_{sup}^{(s)},
\qquad
\mathrm{val\_phys} = 100\cdot\frac{\frac{1}{N_{val}}\sum_{s\in val}\sum_i\lvert\Delta P_i^{(s)}\rvert}{\overline{\text{부하}}_{val}}
$$

`val_phys` 는 §9 의 주 지표와 **같은 식**입니다 (검증 분할에서, float32 로).
그래서 `--select phys` 는 "우리가 보고할 값으로 모델을 고른다" 는 뜻입니다.


```python
# src/nnopf/train.py — train
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                   patience=cfg.lr_patience)
...
crit = vphys if cfg.select == "phys" else vloss
sched.step(vloss if cfg.sched_on == "loss" else crit)
```

> [!NOTE] 일반 Adam 이 아니라 **AdamW** 인 것이 §7.1 과 짝입니다
> 일반 `Adam(weight_decay=)` 는 L2 를 **기울기에 더하는** 방식이라 손실이
> 작을 때 과제 기울기를 눌러 버립니다. AdamW 의 분리형 감쇠는 그 결합을
> 끊습니다. 손실 표준화(§7.1)와 AdamW 는 **같은 사고에 대한 두 겹의 대응**
> 입니다.

> [!CAUTION] 신호 하나가 소비자 셋을 몰고 있습니다
> `crit` 하나가 **모델 선택 · 조기 종료 · 학습률 스케줄러**를 다 몹니다.
> 셋이 원하는 성질이 다릅니다.
>
> | | 필요한 성질 |
> |---|---|
> | 모델 선택 | 우리가 보고하는 값 |
> | 조기 종료 | 추세를 읽을 것 |
> | 학습률 스케줄러 | 조용할 것 |
>
> 이 하나가 06 문서에서 **세 번** 사고를 냈습니다 — 모델 선택(§7.8),
> 학습률 붕괴(작업 51), 조기 종료(작업 57). 미지 N-1 에서는 `val_phys` 가
> epoch 마다 3.07% 튀고 `val` 은 바닥에 닿아 평평해져서, **어느 쪽을 물려도
> 스케줄러가 오작동합니다.**

---

## 11. 한눈에 — 어느 파일이 무슨 식을 갖고 있나

| 파일 | 담고 있는 식 |
|---|---|
| `ybus.py` | $\pi$ 등가회로, $Y_{bus}$ 조립, 브랜치 조류 |
| `powerflow.py` | $S = V\overline{YV}$, $\partial S/\partial V$, 뉴턴 반복, 선로 이용률 |
| `opf.py` | AC-OPF 정식화, 등식/부등식 제약과 그 야코비안, $\partial S_{br}/\partial V$ |
| `dataset.py` | 시나리오 샘플링, 급전 배분, NumPy 판 물리 잔차 |
| `physics_torch.py` | 미분 가능한 물리 잔차 (실수 산술), N-1 4성분 보정, 물리 손실 |
| `models.py` | MLP 입출력 계약, 전압 헤드, 선형 지름길 |
| `gnn.py` | 엣지 어텐션, 게이팅, 모선별 헤드, 엣지 특징 정규화 |
| `train.py` | 지도 손실, λ 스케줄, 야코비안 가중, lstsq 초기화, 평가 지표 |
| `baselines.py` | 최소제곱 선형, DC 조류계산, 야코비안 선형화 |

---

## 12. 논문에 쓸 때 반드시 같이 적어야 하는 것

이 문서를 만들면서 드러난 것들입니다. **식만 적으면 재현되지 않습니다.**

1. **λ 는 정규화 방식과 한 쌍입니다** (§7.6). 우리 λ=5e-3 은 "지도 항 대비
   비율" 이고, 원단위 $\mathrm{pu}^2$ 로 정의한 논문의 λ 와 **숫자가 같아도 뜻이
   다릅니다.**
2. **최소제곱의 rcond 는 하이퍼파라미터입니다** (§8.1). 안 적으면 답이
   30.80% 와 0.90% 사이 어디에도 있을 수 있습니다.
3. **전압 박스는 데이터 범위 + 여유이지 계통 한계가 아닙니다** (§5.3).
4. **잔차 측정 정밀도를 적어야 합니다** (§9). float32 로 재면 바닥이 올라갑니다.
5. **전압 위반 지표의 허용오차** (§9). $\epsilon$ 없이 재면 계통에 따라 지표가
   상수가 됩니다.
6. **뉴턴-랩슨 기준선의 측정 조건** ($Y_{bus}$ 재사용 여부, 06 문서 §8).
   같은 계통에서 3.4 ms 와 9.0 ms 가 둘 다 나옵니다.
