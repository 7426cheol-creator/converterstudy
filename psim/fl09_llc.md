# PSIM 이식표 — FL09 LLC (400 V → 48 V, full bridge, 다이오드 정류)

**상태: NOT_RUN_ENVIRONMENT** — PSIM 12.0.2가 이 환경에 없다. 이 시트는 손으로 회로를 그리기 위한 자료이고 PSIM 결과가 아니다. PSIM 12.0.2의 schematic 생성 API는 가정하지 않는다.

기준 실행:
- f = f_r: `python -m convlab run FL09 switching_vs_fha --preset at_fr` (app 4.0.0, request_hash `92d34978920b2b94`, 실행 2026-09-30T19:53:55, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 C (이상 스위칭·다이오드 정류) + A (FHA 비교))
- F = 0.9: `python -m convlab run FL09 switching_vs_fha --preset mid` (app 4.0.0, request_hash `ce23d1c88995dc00`, 실행 2026-09-30T19:53:54, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 C (이상 스위칭·다이오드 정류) + A (FHA 비교))

교재 12장의 합성 tank(L_r 40 µH, C_r 28.1448 nF, L_m 200 µH, f_r ≈ 150 kHz)를 이상 bridge와 이상 다이오드 정류기로
푼 스위칭 해다. 같은 점의 FHA 예측과 앱이 비교한다. PSIM에서는 이 스위칭 회로를 그대로 그린다.

## 1. 회로

| 소자 | 종류 | 노드 (+ → −) | 값 | 값의 출처 |
|---|---|---|---|---|
| V_in | DC 전압원 | IN+ → IN− | 400 V | 입력 `Vin` |
| S1…S4 | MOSFET full bridge (이상), leg A(S1/S2), B(S3/S4) | IN+ → A → IN−, IN+ → B → IN− | | 입력 `bridge` = FB |
| L_r | 인덕터 | A → N1 | 40 µH | 입력 `Lr` |
| C_r | 커패시터 | N1 → T1 | 28.1448 nF | 입력 `Cr` |
| L_m | 인덕터 (자화) | T1 → B | 200 µH | 입력 `Lm` |
| TR | 이상 변압기, N_p:N_s = 25:3 | 1차 T1(점) – B, 2차 T2(점) – T3 | n = 8.333333 | export `n` = V_in/V_o (FB, f_r에서 V_o = 48 V) |
| D1…D4 | 이상 다이오드 full-bridge 정류 | D1: T2 → O+, D2: T3 → O+, D3: O− → T2, D4: O− → T3 | V_f = 0, R_on = 0 | 앱 가정 |
| C_o | 커패시터 | O+ → O− | 200 µF | 입력 `Co` (ASSUMED) |
| R_L | 저항 | O+ → O− | 0.8371691 Ω | export `RL` (Q = 0.8에서 R_ac = Z₀/Q, R_L = (π²/8)R_ac/n²) |

1차 저항 `R1` = 0. 입력 `td` (100 ns), `C0`, `V0`는 앱의 ZVS screen 표에만 쓰인다 (스위칭 해는 dead time 0).

## 2. Gate timing (dead time 0)

| leg | 상측 on [deg] | at_fr (f = 149 999.93 Hz, T = 6.666670 µs) | mid (f = 134 999.94 Hz, T = 7.407411 µs) |
|---|---|---|---|
| A (S1; S2 보수) | 0 → 180 | 0 → 3.333335 µs | 0 → 3.703705 µs |
| B (S3; S4 보수) | 180 → 360 | 3.333335 → 6.666670 µs | 3.703705 → 7.407411 µs |

t = 0은 v₁의 상승 edge이다 (앱 series `v1`, export `i_edge` = i_r(0)).

## 3. 시뮬레이션 설정

| 항목 | 값 |
|---|---|
| time step | 2 ns |
| total time | 1 ms (C_o·R_L = 167 µs; 초기값을 아래처럼 주면 충분) |
| 저장 | 마지막 2주기 |
| 초기값 at_fr | i_Lr(0) = −3.330537 A, i_Lm(0) = −3.330537 A, v_Cr(0) = −407.5252 V, v_Co(0) = 48.02100 V |
| 초기값 mid | i_Lr(0) = −3.443672 A, i_Lm(0) = −3.443672 A, v_Cr(0) = −506.7436 V, v_Co(0) = 50.74029 V |

초기값은 series `i1`, `im`, `vC1`, `vo`의 t = 0 값이다 (edge 순간 정류 전류 0). v_Cr의 부호는 i_r이 N1 → T1로 흐를 때
증가하는 방향 (v_Cr = V(N1) − V(T1)). 초기값을 줄 수 없으면 0에서 시작해 3 ms 이상 돌린다.

## 4. Probe ↔ 앱 series

| PSIM probe | 측정 | 앱 series |
|---|---|---|
| V(A) − V(B) | bridge 출력 v₁ | `v1` |
| V(T1) − V(B) | L_m 전압 v_m (= 이상 변압기 1차 전압) | `vm` |
| n·(V(T2) − V(T3)) | 정류기 입력의 1차 환산 v₂′ | `v2` |
| I(L_r) | 공진 전류 i_r | `i1` |
| I(L_m) | 자화 전류 | `im` |
| I(L_r) − I(L_m) (= 2차 전류/n) | 정류 전류의 1차 환산 | `i2` |
| V(N1) − V(T1) | C_r 전압 | `vC1` |
| V(O+) − V(O−) | 출력 전압 | `vo` |

## 5. 기대값 (Python 실제 실행)

`python -m convlab run FL09 switching_vs_fha --preset at_fr` (app 4.0.0, request_hash `92d34978920b2b94`, 실행 2026-09-30T19:53:55, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 C (이상 스위칭·다이오드 정류) + A (FHA 비교))

| metric | 뜻 | 값 | 단위 |
|---|---|---|---|
| `f` | 스위칭 주파수 f = F·f_r | 149999.9 | Hz |
| `vo_td` | 출력 V_o (스위칭 해, 주기 평균) | 48.00516 | V |
| `vo_fha` | 출력 V_o (FHA 예측) | 48 | V |
| `g_td` | 정규화 gain (스위칭) k_b n V_o/V_in | 1.000107 |  |
| `P` | 출력 전력 (부하) | 2752.73 | W |
| `I1_rms` | 공진 전류 i_r RMS (1차) | 7.999811 | A |
| `Im_pk` | 여자전류 peak | 3.330632 | A |
| `vC1_pk` | C_r 전압 peak (DC 포함) | 426.4266 | V |
| `vo_pp` | 출력 리플 p-p | 0.2044489 | V |
| `i_edge` | 1차 bridge 상승 edge 전류 i_r(0) | -3.330537 | A |
| `off_frac` | 정류 off 구간 비율 (i₂ = 0) | 0.0009321627 |  |

`python -m convlab run FL09 switching_vs_fha --preset mid` (app 4.0.0, request_hash `ce23d1c88995dc00`, 실행 2026-09-30T19:53:54, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 C (이상 스위칭·다이오드 정류) + A (FHA 비교))

| metric | 뜻 | 값 | 단위 |
|---|---|---|---|
| `f` | 스위칭 주파수 f = F·f_r | 134999.9 | Hz |
| `vo_td` | 출력 V_o (스위칭 해, 주기 평균) | 50.78385 | V |
| `vo_fha` | 출력 V_o (FHA 예측) | 49.59013 | V |
| `g_td` | 정규화 gain (스위칭) k_b n V_o/V_in | 1.057997 |  |
| `g_fha` | FHA \|H\| | 1.033128 |  |
| `P` | 출력 전력 (부하) | 3080.633 | W |
| `I1_rms` | 공진 전류 i_r RMS (1차) | 8.981167 | A |
| `Im_pk` | 여자전류 peak | 3.576761 | A |
| `vC1_pk` | C_r 전압 peak (DC 포함) | 523.8484 | V |
| `vo_pp` | 출력 리플 p-p | 0.3074277 | V |
| `i_edge` | 1차 bridge 상승 edge 전류 i_r(0) | -3.443672 | A |
| `off_frac` | 정류 off 구간 비율 (i₂ = 0) | 0.104165 |  |

F = 0.9에서는 스위칭 해가 FHA보다 2.4 % 높다 (정류 off 구간 10.4 %). 이 차이가 PSIM에서도 같은 크기로 나오는지가
이 시트의 핵심 질문이다. FHA 표(|H|, Q 0.2/0.8/1.5)는 `FL09 fha_gain --preset textbook`에 있다.

## 6. 비교

마지막 한 주기에서 V_o 평균, 출력 전력, i_r RMS, i_m peak, v_Cr peak(DC 포함), 출력 리플 p-p, t = 0 edge 전류를
비교한다. 정류 off 비율은 i₂ = 0인 시간의 비율로 구한다.

## 7. 이 회로가 말하지 않는 것

ZVS(Coss 전하·dead time), SR 동작, 손실, 동특성(G_vf), 기동은 없다. 이 시트의 PASS는 "LLC switching model이
PSIM에서 같은 주기해를 준다"는 뜻이지 ZVS PASS가 아니다.
