# 배경 논문

정독 정리는 [`../docs/04_paper_review.md`](../docs/04_paper_review.md) 를 보세요.

| 파일 | 논문 | 라이선스 |
|---|---|---|
| `P1_2024_Mohammadi_*.pdf` | Mohammadi, S.; Bui, V.-H.; Su, W.; Wang, B. **Surrogate Modeling for Solving OPF: A Review.** *Sustainability* **2024**, 16, 9851. | CC BY 4.0 (MDPI 오픈액세스) |
| `P2_2026_Wen_*.pdf` | Wen, T.; Wang, W.; Chen, J.; Wang, Z. **Power Flow Surrogate for Power Systems with High Renewable Penetration via a Physics-Informed Graph Attention Network.** *Energies* **2026**, 19, 2972. | CC BY 4.0 (MDPI 오픈액세스) |
| `P3_2024_Cheng_*.pdf` | Cheng, R.; Yang, Y.; Liu, W.; Liu, N.; Wang, Z. **Input Convex Neural Network-Assisted Optimal Power Flow in Distribution Networks.** arXiv:2407.20675. | arXiv 프리프린트 |
| `P4_2026_Panagi_*.pdf` | Panagi, S.; Spanias, C.; Aristidou, P. **Enhanced Optimal Power Flow Using a Trained Neural Network Surrogate for Distribution Grid Constraints.** arXiv:2604.12422. | arXiv 프리프린트 |

## 네 편의 관계

```
P1 (리뷰, 2024)  ──▶ "GNN, 특히 graph attention network를 살펴볼 필요가 있다"
                            │
                            ▼
                      P2 (PI-GAT, 2026)          4단계: 조류계산 대체
                      정확·물리일관·N-1 일반화
                      단, 비볼록 → OPF 삽입 불가
                            ┆
                            ┆  (빈칸 = 우리 자리)
                            ┆
P4 (MILP, 2026) ──▶ "이진변수를 피하는 볼록 연속 정식화가 필요하다"
   정확한 인코딩              │
   단, 이진변수 폭발          ▼
                      P3 (ICNN, 2024)            5단계: OPF에 삽입
                      볼록·수렴/최적성 증명
                      단, MLP 수준·정확도 손실
```

**빈칸**: 그래프 구조 + 물리정보 + 볼록성(삽입 가능)을 **동시에** 만족하는 대체모델.

## 추가로 읽을 것

* Amos, B.; Xu, L.; Kolter, J.Z. **Input Convex Neural Networks.** ICML 2017 — P3의 토대
* Nellikkath, R.; Chatzivasileiadis, S. **Physics-informed neural networks for AC optimal power flow.** *EPSR* 2022
* Baker, K. **Learning warm-start points for AC optimal power flow.** MLSP 2019 — 5단계 (c) 방식
* Farivar, M.; Low, S.H. **Branch flow model: relaxations and convexification.** *IEEE TPWRS* 2013 — SOCP 기준선
