# PSIM 이식표 — FL10 CLLC (n = 0.93, link 850 V → battery 920 V, 두 분기)

**상태: NOT_RUN_ENVIRONMENT** — PSIM 12.0.2가 이 환경에 없다. 이 시트는 손으로 회로를 그리기 위한 자료이고 PSIM 결과가 아니다. PSIM 12.0.2의 schematic 생성 API는 가정하지 않는다.

**기대값 자리만 있다.** FL10(CLLC) 앱이 아직 병합되지 않아 Python 스위칭 해가 없다. 아래 5절의 표는 FL10이 병합되면
`python -m convlab run FL10 <experiment> --preset <preset>`의 값으로 채운다. 지금 있는 숫자는 교재 13장의 FHA 값뿐이며,
이것은 스위칭 결과가 아니다.

교재 13장 수정 후보 A: seed(n = 1)는 920/850 V corner에서 FHA 해가 없다 (필요 gain 1.082353 > inductive 최대 1.016401).
n = 0.93으로 바꾸고 1차 L/C와 환산 대칭을 유지하면 136.099 kHz와 147.061 kHz 두 inductive FHA 해가 생긴다.
두 해는 같은 gain이지만 기울기 부호와 전류가 다르다. 이 시트는 두 분기를 각각 PSIM으로 돌리기 위한 것이다.

## 1. 회로 (교재 13장 값; FL10 앱 정의가 나오면 맞춘다)

| 소자 | 종류 | 노드 (+ → −) | 값 | 값의 출처 |
|---|---|---|---|---|
| V_link | DC 전압원 | L+ → L− | 850 V | 교재 13장 (link 700~850 V의 high corner) |
| S1…S4 | MOSFET 1차 full bridge (이상), leg A(S1/S2), B(S3/S4) | L+ → A → L−, L+ → B → L− | | |
| L_r1 | 인덕터 | A → N1 | 40 µH | 교재 13장 |
| C_r1 | 커패시터 | N1 → T1 | 28.1448 nF | 교재 13장 (f_r = 150 kHz; 정확값 28.144773 nF) |
| L_m | 인덕터 (자화, 1차 쪽) | T1 → B | 200 µH | 교재 13장 |
| TR | 이상 변압기, n = N_p/N_s = 0.93 | 1차 T1(점) – B, 2차 T2(점) – T3 | 0.93 (예: 93:100) | 교재 13장 수정안 A |
| L_r2 | 인덕터 (2차 실제값) | T2 → N2 | 46.2481 µH (= 40 µH/0.93²) | 교재 13장 |
| C_r2 | 커패시터 (2차 실제값) | N2 → R1 | 24.3424 nF (= 28.1448 nF·0.93²) | 교재 13장 |
| D1…D4 | 이상 다이오드 full-bridge 정류 (SR이면 FL10 정의를 따른다) | D1: R1 → B+, D2: T3 → B+, D3: B− → R1, D4: B− → T3 | | FL10 정의 대기 |
| V_bat | DC 전압원 (배터리, 이상) 또는 C_o + 부하 | B+ → B− | 920 V, 11 kW 점 | 교재 13장; 출력 모델은 FL10 정의 대기 |

2차 L/C는 실제값이다. 1차 환산하면 L_r2′ = n²L_r2 = 40 µH, C_r2′ = C_r2/n² = 28.1448 nF (환산 대칭). n만 바꾸고
2차 L/C를 그대로 둔 회로와 다르다 (교재 13장).

## 2. Gate timing (dead time 0, 1차만; 2차는 다이오드)

| leg | 상측 on [deg] | 분기 1: f = 136.099 kHz (T = 7.347593 µs) | 분기 2: f = 147.061 kHz (T = 6.799899 µs) |
|---|---|---|---|
| A (S1; S2 보수) | 0 → 180 | 0 → 3.673796 µs | 0 → 3.399950 µs |
| B (S3; S4 보수) | 180 → 360 | 3.673796 → 7.347593 µs | 3.399950 → 6.799899 µs |

## 3. 시뮬레이션 설정 (제안)

| 항목 | 값 |
|---|---|
| time step | 1 ns |
| total time | 1 ms (FHA 등가회로의 가장 느린 모드 시정수 약 33 µs; 정류기·직렬 C의 DC 정착 여유 포함) |
| 저장 | 마지막 20주기 |
| 초기값 | 0 (FL10 주기해가 나오면 그 초기 상태로 바꾼다) |

## 4. Probe ↔ 앱 series

FL10 series 이름이 정해지지 않았다. 측정할 양: i_r1 = I(L_r1), i_m = I(L_m), i_r2 = I(L_r2), v_Cr1, v_Cr2, v_m = V(T1) − V(B),
배터리 전류와 출력 전력, 1차 bridge 상승 edge의 i_r1. FL10이 병합되면 이 표에 series 이름을 적는다.

## 5. 기대값

### 5.1 FL10 스위칭 해 — **자리만 (FL10 병합 후 채운다)**

기준 실행: `python -m convlab run FL10 <experiment> --preset <preset>` (미정)

| 양 | 분기 1 (136.099 kHz) | 분기 2 (147.061 kHz) | 단위 |
|---|---|---|---|
| 배터리 전력 P_out | TBD | TBD | W |
| i_r1 RMS | TBD | TBD | A |
| i_r2 RMS (2차 실제) | TBD | TBD | A |
| v_Cr1 peak | TBD | TBD | V |
| v_Cr2 peak | TBD | TBD | V |
| 1차 상승 edge 전류 i_r1(0) | TBD | TBD | A |
| 정류 off 구간 비율 | TBD | TBD | |

### 5.2 교재 13장 FHA 값 (스위칭 결과가 아님, 참고)

| 양 | 분기 1 | 분기 2 |
|---|---|---|
| FHA 해 주파수 | 136.099 kHz | 147.061 kHz |
| 필요 gain n·V_bat/V_link | 1.006588 | 1.006588 |
| 입력 impedance | inductive | inductive |
| gain 기울기 (±10 Hz 중앙차분) | +0.001871 /kHz | −0.001803 /kHz |
| FHA 1차 series RMS | 14.390 A | 14.765 A |

이 FHA 값은 Octave 독립 계산(`matlab/xc_cllc.m`, 결과 `matlab/results/octave_crosscheck.json`의 `FL10.*` 항목)으로
교재 자릿수 안에서 재현했다. C_r1을 f_r = 150 kHz 정확값으로 두면 해는 136 099.47 Hz, 147 060.86 Hz다.

## 6. 비교

FL10 값이 채워지면 두 분기 각각에서 5.1 표를 README 허용 범위로 비교한다. 스위칭 해가 FHA 해 주파수에서 11 kW를
정확히 주지 않는 것은 정상이다 (FHA 근사). 두 분기의 전류·C 전압·기울기 차이가 PSIM에서도 같은 방향인지 본다.

## 7. 이 회로가 말하지 않는 것

ZVS, SR 타이밍, 기동, 역방향(battery → link), 공차(L/C ±5 %, L_m ±15 %), 폐루프는 없다. 교재 E05: 주기해를 찾았다는
것은 안정성 증명이 아니다.
