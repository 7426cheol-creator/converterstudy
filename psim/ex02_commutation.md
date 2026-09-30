# PSIM 이식표 — EX02 half-bridge 전환, 비선형 Coss (800 V, 4 A)

**상태: NOT_RUN_ENVIRONMENT** — PSIM 12.0.2가 이 환경에 없다. 이 시트는 손으로 회로를 그리기 위한 자료이고 PSIM 결과가 아니다. PSIM 12.0.2의 schematic 생성 API는 가정하지 않는다.

기준 실행: `python -m convlab run EX02 hb_constant_current --preset textbook` (app 4.0.0, request_hash `c5e5ade61f01cbe7`, 실행 2026-09-30T19:53:56, 결과 상태 SCREEN_ONLY, 모델 수준 A (charge screen) + D-합성 해석)

교재 E02의 학습용 비선형 소자 C(v) = C₀/√(1 + v/V₀), C₀ = 2 nF, V₀ = 40 V (특정 소자의 fit이 아님)를 두 개 쓴
half-bridge다. 양쪽 rail이 고정이고, dead time 동안 인덕터 전류를 일정한 전류원 I로 본다. 이것은 E02의 첫 단계
(전하–전류 원리)이며 결과 상태도 SCREEN_ONLY다.

## 1. 기준 회로 (물리 회로)

| 소자 | 종류 | 노드 (+ → −) | 값 | 값의 출처 |
|---|---|---|---|---|
| V_b | DC 전압원 (rail) | RAIL → 0 | 800 V | 입력 `Vb` |
| C_H | 상측 소자 Coss (gate off) | RAIL → X | C(v_H), v_H = V_b − v_X, 시작 800 V | 입력 `C0`, `V0` |
| C_L | 하측 소자 Coss (gate off) | X → 0 | C(v_L), v_L = v_X, 시작 0 V | 입력 `C0`, `V0` |
| I | DC 전류원 (dead time 동안의 인덕터 전류) | 0 → X (X로 주입) | 4 A | 입력 `I` |
| C_par | 커패시터 (node 기생) | X → 0 | 0 (없음) | 입력 `Cpar` |

## 2. PSIM에서 비선형 C를 만드는 방법

**A. 설치본에 전압 의존 커패시턴스(또는 Coss(v) 표를 받는 MOSFET)가 있으면** 위 표를 그대로 그린다. 두 소자에
C(v) = C₀/√(1 + v/V₀)를 넣고 초기 전압을 C_H 800 V, C_L 0 V로 준다. 이 방법이 소자별 전하를 그대로 보여 주므로
우선이다. 12.0.2에 그런 소자가 있는지는 설치본에서 확인한다.

**B. 없으면 node 전하로 만든다.** rail이 고정이면 node에서 본 두 소자는 하나의 비선형 커패시터다. node에 들어간
전하 q로 전압이 닫힌 식으로 정해진다:

Q_node(v) = Q_oss(v) + Q_oss(V_b) − Q_oss(V_b − v), Q_oss(v) = 2C₀V₀(√(1 + v/V₀) − 1).
a = √(1 + v/V₀), b = √(1 + (V_b − v)/V₀)라 두면 a² + b² = S = 2 + V_b/V₀ = 22 (상수)이고
Q_node = 2C₀V₀(a − b) + Q_oss(V_b)이므로

  d = (q − Q_oss(V_b)) / (2C₀V₀),  a = (d + √(2S − d²)) / 2,  **v = V₀(a² − 1)**

(q = 0 → v = 0, q = 2Q_oss(V_b) → v = 800 V; 2C₀V₀ = 160 nC, Q_oss(V_b) = 573.2121 nC).

| 소자 | 종류 | 연결 | 값 |
|---|---|---|---|
| I | DC 전류원 | 0 → X | 4 A |
| CS | 전류 센서 (gain 1) | X → X1 (node 커패시턴스로 들어가는 전류) | |
| INT | 적분기 (제어회로) | 입력 CS 출력, 초기 출력 0 | q(t) = ∫ i dt [C] |
| F | 수식 블록 | 입력 q | 위 식 v(q) (V₀ = 40, S = 22, 2C₀V₀ = 1.6e-7, Q_oss(V_b) = 5.732121e-7) |
| E_X | 제어 전압원 (제어 신호 입력) | X1 → 0 | e = F 출력 |

소자별 전압은 v_L = v_X, v_H = V_b − v_X로 읽는다. 이 회로에는 전압원 loop가 없다 (V_b, C_H, C_L을 모두 전압원으로
바꾸면 loop가 생겨 풀 수 없다). 단일점 비교(앱 series `v_node_pt`)는 F를 v = q / (2·C(V_b)) (= q / 0.8728716 nF)로
바꾼 선형 run이다.

## 3. 시뮬레이션 설정

| 항목 | 값 |
|---|---|
| time step | 0.1 ns (t_trans의 약 1/2900) |
| total time | 320 ns (앱 series 구간 0 → 315.3 ns) |
| 저장 | 전 구간 |
| 초기값 | 방법 A: v_CH = 800 V, v_CL = 0 V / 방법 B: 적분기 0 |

제어회로와 전력회로 사이의 한 step 지연은 방법 B에서 결과에 거의 영향이 없다 (node 전류가 항상 I).
800 V에 닿은 뒤에는 실제로는 상측 body diode가 node를 rail에 묶는다. 이 모델(두 방법 모두)은 그 clamp가 없으므로
rail 도달 시각까지만 비교한다.

## 4. Probe ↔ 앱 series

| PSIM probe | 측정 | 앱 series |
|---|---|---|
| V(X) | switch node 전압 (0 → 800 V) | `v_node` |
| V(X), 선형 단일점 run | 같은 전류, C_node = 2·Coss(V_b) | `v_node_pt` |
| 800 V 도달 시각 | 완전 전환시간 | `t_trans` |
| t = 200 ns의 V(X) | dead time 끝 node 전압 | `v_td` |

## 5. 기대값 (Python 실제 실행)

`python -m convlab run EX02 hb_constant_current --preset textbook` (app 4.0.0, request_hash `c5e5ade61f01cbe7`, 실행 2026-09-30T19:53:56, 결과 상태 SCREEN_ONLY, 모델 수준 A (charge screen) + D-합성 해석)

| metric | 뜻 | 값 | 단위 |
|---|---|---|---|
| `t_trans` | 완전 전환시간 t = Q_node(V_bus)/\|I\| | 2.866061e-07 | s |
| `t_pt` | Coss(V_bus) 한 점으로 계산한 전환시간 | 1.745743e-07 | s |
| `Qreq` | 필요 전하 Q_node(V_bus) = 2Q_oss + C_par·V | 1.146424e-06 | C |
| `v_td` | dead time 끝 노드 전압 v(t_d) | 583.6993 | V |
| `vds_on` | H측 turn-on 직전 V_DS = V_bus − v(t_d) | 216.3007 | V |
| `mQ` | 전하 여유 m_Q = (I·t_d − Q_req)/Q_req | -0.302178 |  |
| `class` | 판정 (charge screen) | 부분 전환 |  |

같은 입력의 `EX02 qe_integrals --preset textbook`: Q_oss(800 V) = 573.212 nC, E_oss(800 V) = 180.238 µJ.
dead time 표(같은 실행 `t_dt`): t_d = 100 / 200 / 300 / 500 ns에서 v(t_d) = 258.306 / 583.699 / 800 / 800 V.

## 6. 비교

V(X)가 800 V에 닿는 시각, t_d = 200 ns의 V(X), 잔류 V_DS = 800 V − V(X)를 비교한다. 단일점 run의 전환시간
174.574 ns가 비선형 286.606 ns보다 39 % 짧다는 것(단일점 Coss는 시간을 과소평가)이 이 시트의 요점이다.

## 7. 이 회로가 말하지 않는 것

실제 공진 전류(전환 중 i(t) 변화), channel 전류, gate 지연, 온도별 Q_oss는 없다. 이 회로의 PASS는 E02의 첫 단계가
PSIM에서 같게 나온다는 뜻이지 실제 converter의 ZVS PASS가 아니다 (교재 E02).
