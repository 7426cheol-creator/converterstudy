# PSIM 이식표 — FL10 CLLC (n = 0.93, link 850 V → battery 920 V, 두 FHA 분기)

**상태: NOT_RUN_ENVIRONMENT** — PSIM 12.0.2가 이 환경에 없다. 이 시트는 손으로 회로를 그리기 위한 자료이고 PSIM 결과가 아니다. PSIM 12.0.2의 schematic 생성 API는 가정하지 않는다.

기준 실행:
- 스위칭 해 (강한 배터리, 손실 0): `python -m convlab run FL10 time_domain --preset textbook` (app 4.0.0, request_hash `ddb87128e4163f80`, 결과 상태 FAIL_CONSTRAINT, 모델 수준 C (이상 스위칭, 강한 배터리) + A)
- 스위칭 해 (R₁ = R₂′ = 50 mΩ): `python -m convlab run FL10 time_domain --preset lossy` (app 4.0.0, request_hash `746acd3f7d09f297`, 결과 상태 FAIL_CONSTRAINT, 모델 수준 C (이상 스위칭, 강한 배터리) + A)
- FHA 해: `python -m convlab run FL10 fix_n093 --preset textbook` (app 4.0.0, request_hash `0dcca4257bed3a12`, 결과 상태 UNRESOLVED_RANKING, 모델 수준 A (FHA))
- 출력 C·R 부하 (EX05): `python -m convlab run EX05 operating_points --preset textbook` (app 4.0.0, request_hash `0c2725755e2c5df9`, 결과 상태 OUT_OF_VALIDITY, 모델 수준 C (스위칭 주기해) + A (FHA))

교재 13장 수정 후보 A다. seed(n = 1)는 920/850 V corner에서 FHA 해가 없다 (필요 gain 1.082353 > inductive 최대
1.016401). n = 0.93으로 바꾸고 1차 L/C와 환산 대칭을 유지하면 136.099 kHz와 147.061 kHz 두 inductive FHA 해가 생긴다.
**앱의 스위칭 모델(이상 bridge·다이오드, 강한 link·배터리)은 이 두 주파수에서 11 kW가 아니라 17.34 kW와 22.24 kW를
주고, 같은 위 branch에서 11 kW는 148.005 kHz의 매우 가파른 곳(dP/df ≈ −1630 W/Hz)에 있다.** 이 시트는 그 결과를
PSIM에서 다시 보기 위한 것이다.

## 1. 회로

| 소자 | 종류 | 노드 (+ → −) | 값 | 값의 출처 |
|---|---|---|---|---|
| V_link | DC 전압원 (강한 link) | L+ → L− | 850 V | 입력 `Vlink_hi` |
| S1…S4 | MOSFET 1차 full bridge (이상), leg A(S1/S2), B(S3/S4) | L+ → A → L−, L+ → B → L− | | |
| L_r1 | 인덕터 | A → N1 | 40 µH | 입력 `L1` |
| R_1 | 저항 (lossy preset만) | N1 → N1′ | 0 / 50 mΩ | 입력 `R` (textbook 0, lossy 0.05 Ω) |
| C_r1 | 커패시터 | N1′ → T1 | 28.144773 nF (f_r = 150 kHz) | 입력 `C1` (교재 인쇄값 28.1448 nF) |
| L_m | 인덕터 (자화, 1차 쪽) | T1 → B | 200 µH | 입력 `Lm` |
| TR | 이상 변압기, n = N_p/N_s = 0.93 | 1차 T1(점) – B, 2차 T2(점) – T3 | 0.93 (예: 93:100) | 입력 `n` |
| L_r2 | 인덕터 (2차 실제값) | T2 → N2 | 46.248121 µH (= L₁/n²) | export `L2` |
| R_2 | 저항 (2차 실제값, lossy만) | N2 → N2′ | 0 / 57.810 mΩ (= R₂′/n², R₂′ = 50 mΩ) | 입력 `R` (1차 환산값) |
| C_r2 | 커패시터 (2차 실제값) | N2′ → R1 | 24.342414 nF (= C₁·n²) | export `C2` |
| D1…D4 | 이상 다이오드 full-bridge 정류 | D1: R1 → B+, D2: T3 → B+, D3: B− → R1, D4: B− → T3 | V_f = 0 | 앱 가정 (이상 다이오드) |
| V_bat | DC 전압원 (강한 배터리) | B+ → B− | 920 V | 입력 `Vbat_hi` |

EX05 비교에서는 V_bat 자리에 C_o = 100 µF ∥ R_L = V²/P = 76.945 Ω (실제값)를 둔다 (EX05 입력 `Co`, `Vref`, `P`).
다른 corner(650/700 V, 800/800 V)는 V_link·V_bat만 바꾼다.

## 2. Gate timing (dead time 0, 1차만)

| leg | 상측 on [deg] | 분기 1: f = 136099.4667 Hz (T = 7.347567 µs) | 분기 2: f = 147060.8627 Hz (T = 6.799906 µs) | 11 kW 점: f = 148004.7409 Hz (T = 6.756540 µs) |
|---|---|---|---|---|
| A (S1; S2 보수) | 0 → 180 | 0 → 3.673784 µs | 0 → 3.399953 µs | 0 → 3.378270 µs |
| B (S3; S4 보수) | 180 → 360 | 3.673784 → 7.347567 µs | 3.399953 → 6.799906 µs | 3.378270 → 6.756540 µs |

11 kW 점은 1 Hz가 15 % 전력이다. PSIM의 주파수는 0.01 Hz까지 넣고, 같은 주파수에서 비교한다. t = 0은 v₁ 상승 edge다.

## 3. 시뮬레이션 설정

| 항목 | 값 |
|---|---|
| time step | 1 ns 이하 (11 kW 점은 0.2 ns 권장: 전력이 edge 시각에 매우 민감) |
| total time | 10 ms (가장 느린 Floquet 승수 0.9891 → 1e-4까지 약 840주기 ≈ 5.7 ms) |
| 저장 | 마지막 20주기 |
| 초기값 (11 kW 점, series t = 0) | i_Lr1 = -7.017792 A, i_Lm = -7.017792 A, i_Lr2 = 0 (정류 off), v_Cr1 = -821.3003 V, v_Cr2 = -877.3385 V (실제; 1차 환산 -815.9248 V) |

FHA 해 두 점은 초기값 0에서 10 ms를 돌린다.

## 4. Probe ↔ 앱 series (`FL10 time_domain`, 11 kW 점 2주기)

| PSIM probe | 측정 | 앱 series |
|---|---|---|
| I(L_r1) | 1차 공진 전류 | `i1` |
| I(L_m) | 자화 전류 | `im` |
| I(L_r2) / 0.93 | 정류 전류의 1차 환산 (i₂′ = i₂/n) | `i2` |
| V(A) − V(B) | bridge 출력 v₁ | `v1` |
| 0.93·(V(R1) − V(T3)) | 정류기 입력의 1차 환산 v₂′ | `v2` |
| V(N1′) − V(T1) | C_r1 전압 | `vC1` |
| 0.93·(V(N2′) − V(R1)) | C_r2 전압의 1차 환산 | `vC2` |

## 5. 기대값 (Python 실제 실행)

### 5.1 강한 배터리, 손실 0 (`FL10 time_domain --preset textbook`)

`python -m convlab run FL10 time_domain --preset textbook` (app 4.0.0, request_hash `ddb87128e4163f80`, 결과 상태 FAIL_CONSTRAINT, 모델 수준 C (이상 스위칭, 강한 배터리) + A)

| corner | FHA 해 | branch | 스위칭 전력 (FHA 대비) | I₁ RMS (FHA) | 정류 off | Floquet \|λ\|max |
|---|---|---|---|---|---|---|
| 저전압 650/700 | 164.390 kHz | peak 위 (−기울기) | 9.13 kW (-17.0 %) | 17.61 A (FHA 21.05) | 0.0 % | 0.9537 |
| 중간 800/800 | 162.811 kHz | peak 위 (−기울기) | 7.63 kW (-30.6 %) | 12.38 A (FHA 17.23) | 0.0 % | 0.9314 |
| 고전압 920/850 | 136.099 kHz | peak 아래 (+기울기) | 17.34 kW (+57.6 %) | 23.38 A (FHA 14.39) | 0.0 % | 0.9224 |
| 고전압 920/850 | 147.061 kHz | peak 위 (−기울기) | 22.24 kW (+102.2 %) | 29.37 A (FHA 14.77) | 0.0 % | 0.9891 |

| corner | branch | FHA 해 | 11 kW 스위칭 동작점 | I₁ RMS | dP/df | ZVS screen |
|---|---|---|---|---|---|---|
| 저전압 650/700 | peak 위 (−기울기) | 164.390 kHz | 162.422 kHz | 21.00 A | -1.1 kW/kHz | 부호·전하 screen 통과 |
| 중간 800/800 | peak 위 (−기울기) | 162.811 kHz | 160.262 kHz | 17.24 A | -1.7 kW/kHz | 부호·전하 screen 통과 |
| 고전압 920/850 | peak 아래 (+기울기) | 136.099 kHz | 범위 내 없음 | 범위 끝 120 kHz까지 11 kW에 도달하지 않음 (P = 13.16 kW) | - | - |
| 고전압 920/850 | peak 위 (−기울기) | 147.061 kHz | 148.005 kHz | 15.87 A | -1630.0 kW/kHz | 부호·전하 screen 통과 |

| metric | 뜻 | 값 |
|---|---|---|
| `P_td_lo` | 136.099 kHz FHA 해의 스위칭 전력 | 17341.45 W |
| `P_td_hi` | 147.061 kHz FHA 해의 스위칭 전력 | 22242.00 W |
| `f_sw_hi` | 위 branch 11 kW 주파수 (1 Hz 구간 이분의 중간점) | 148004.74091 Hz |
| `sens_hi` | 그 점의 dP/df (±2 Hz 중앙차분) | -1630030.4 W/kHz |
| `vC1_pk` | 그 점의 C_r1 전압 peak | 860.110 V |
| `vC2_pk` | 그 점의 C_r2 전압 peak (1차 환산) | 815.925 V (실제 877.3 V) |

### 5.2 R₁ = R₂′ = 50 mΩ (`FL10 time_domain --preset lossy`)

`python -m convlab run FL10 time_domain --preset lossy` (app 4.0.0, request_hash `746acd3f7d09f297`, 결과 상태 FAIL_CONSTRAINT, 모델 수준 C (이상 스위칭, 강한 배터리) + A)

| corner | FHA 해 | branch | 스위칭 전력 (FHA 대비) | I₁ RMS (FHA) | 정류 off | Floquet \|λ\|max |
|---|---|---|---|---|---|---|
| 저전압 650/700 | 164.202 kHz | peak 위 (−기울기) | 9.16 kW (-16.7 %) | 17.65 A (FHA 21.05) | 0.0 % | 0.9535 |
| 중간 800/800 | 162.540 kHz | peak 위 (−기울기) | 7.72 kW (-29.8 %) | 12.49 A (FHA 17.23) | 0.0 % | 0.9316 |
| 고전압 920/850 | 137.183 kHz | peak 아래 (+기울기) | 17.40 kW (+58.2 %) | 23.46 A (FHA 14.43) | 0.0 % | 0.9217 |
| 고전압 920/850 | 145.905 kHz | peak 위 (−기울기) | 19.08 kW (+73.4 %) | 25.38 A (FHA 14.73) | 0.0 % | 0.9784 |

| corner | branch | FHA 해 | 11 kW 스위칭 동작점 | I₁ RMS | dP/df | ZVS screen |
|---|---|---|---|---|---|---|
| 저전압 650/700 | peak 위 (−기울기) | 164.202 kHz | 162.261 kHz | 20.99 A | -1.1 kW/kHz | 부호·전하 screen 통과 |
| 중간 800/800 | peak 위 (−기울기) | 162.540 kHz | 160.051 kHz | 17.23 A | -1.6 kW/kHz | 부호·전하 screen 통과 |
| 고전압 920/850 | peak 아래 (+기울기) | 137.183 kHz | 범위 내 없음 | 범위 끝 120 kHz까지 11 kW에 도달하지 않음 (P = 13.10 kW) | - | - |
| 고전압 920/850 | peak 위 (−기울기) | 145.905 kHz | 147.442 kHz | 15.11 A | -19.4 kW/kHz | 부호·전하 screen 통과 |

| metric | 뜻 | 값 |
|---|---|---|
| `P_td_lo` | 137.183 kHz FHA 해의 스위칭 전력 | 17401.57 W |
| `P_td_hi` | 145.905 kHz FHA 해의 스위칭 전력 | 19075.77 W |
| `f_sw_hi` | 위 branch 11 kW 주파수 | 147441.89420 Hz |
| `sens_hi` | 그 점의 dP/df | -19378.5 W/kHz |
| `vC1_pk` | 그 점의 C_r1 전압 peak | 822.305 V |
| `vC2_pk` | 그 점의 C_r2 전압 peak (1차 환산) | 774.908 V |

손실 50 mΩ만 넣어도 11 kW 점의 dP/df가 −1630에서 −19.4 kW/kHz로 줄어든다. 무손실 이상 모델의 가파름은 f_r 근처의
작은 순 리액턴스 때문이다. PSIM과 비교할 때는 lossy preset이 수치적으로 덜 민감하다.

### 5.3 출력 C·R 부하 (`EX05 operating_points --preset textbook`)

`python -m convlab run EX05 operating_points --preset textbook` (app 4.0.0, request_hash `0c2725755e2c5df9`, 결과 상태 OUT_OF_VALIDITY, 모델 수준 C (스위칭 주기해) + A (FHA))

| FHA 해 | 스위칭 V_o | FHA 기울기 | 스위칭 기울기 | 부호 |
|---|---|---|---|---|
| 136.099 kHz | 961.52 V | +1.710 V/kHz | -3.758 V/kHz | 반대 |
| 147.061 kHz | 922.94 V | -1.648 V/kHz | -3.139 V/kHz | 같음 |

| metric | 뜻 | 값 |
|---|---|---|
| `vo_at_lo` | 136.099 kHz FHA 해에서 스위칭 V_o | 961.518 V |
| `vo_at_hi` | 147.061 kHz FHA 해에서 스위칭 V_o | 922.936 V |
| `f0` | 스위칭 모델에서 V_o = 920 V인 주파수 (brentq, xtol 1 mHz) | 148005.2304 Hz |
| `f_peak_fha` | FHA gain peak | 141529.6 Hz |

EX05는 이 부하에서 FHA 아래 branch 해(136.099 kHz)의 스위칭 기울기 부호가 FHA와 **반대**라고 판정한다
(OUT_OF_VALIDITY). 같은 출력이라도 제어 부호가 바뀐다는 뜻이다. V_o = 920 V 운전점 148.0052 kHz는 강한 배터리의 11 kW 점
(148.0047 kHz, 1 Hz 이분)과 같은 물리 운전점이다.

### 5.4 Octave 독립 검산 (`matlab/xc_cllc_td.m`, 참고)

앱 코드를 쓰지 않은 hybrid `ode45` 모델(교재 E05 상태식, 3-상태 이상 다이오드 정류기, Newton shooting)로 같은 점을
다시 풀었다. 위 표의 값과 모두 허용오차 안에서 같다 (`matlab/results/octave_crosscheck.json`의 `FL10.td_*`, `EX05.op.*`).
추가로 확인한 것:

| 점 | 전력 | I₁ RMS | v_Cr1 peak | v_Cr2′ peak | 정류 off | Floquet \|λ\|max |
|---|---|---|---|---|---|---|
| 앱의 11 kW 주파수 148 004.7409 Hz (무손실) | 11 632 W | 15.871 A | 860.109 V | 815.923 V | 1.33 % | 0.99999 |
| 이 모델의 11 kW 주파수 148 004.8618 Hz (무손실) | 11 000 W | 15.077 A | 817.53 V | 771.59 V | 1.32 % | 0.99998 |
| 앱의 11 kW 주파수 147 441.8942 Hz (lossy) | 11 005 W | 15.113 A | 822.305 V | 774.908 V | 1.70 % | 0.9958 |

무손실 모델에서는 1 Hz 구간 이분의 중간점이 11 kW에서 +5.7 % 떨어져 있고, 그 점의 Floquet 승수가 거의 1이다
(교란이 e배 줄어드는 데 수만 주기가 걸린다: 1/(1 − 0.9999853) ≈ 6.8만 주기). PSIM에서 이 점을 볼 때는 같은 주파수를 넣고 C 전압 peak로 비교하는 편이 낫다.

### 5.5 교재 13장 FHA 값 (참고, 스위칭 결과가 아님)

| 양 | 분기 1 | 분기 2 |
|---|---|---|
| FHA 해 | 136099.4667 Hz | 147060.8627 Hz |
| 필요 gain n·V_bat/V_link | 1.006588 | 1.006588 |
| gain 기울기 (±10 Hz) | +0.001871 /kHz | -0.001803 /kHz |
| FHA 1차 series RMS | 14.3898 A | 14.7652 A |

## 6. 비교

FHA 해 두 주파수에서 배터리 전력, I₁ RMS, 정류 off 비율을 비교한다 (README 허용 범위). 11 kW 점은 주파수 1 Hz가
15 % 전력이므로 전력 대신 C 전압 peak와 I₁ RMS를 같은 주파수에서 비교하고, 전력이 11 kW가 되는 주파수를 따로 찾는다.
앱의 `f_sw_hi`는 1 Hz 구간의 중간점이라 그 주파수의 전력은 11 kW에서 ±0.5 Hz × |dP/df| 안에서 다르다.

## 7. 이 회로가 말하지 않는 것

ZVS(Coss·dead time), SR 타이밍, 기동, 역방향, 공차, 배터리 내부저항, 폐루프는 없다. 주기해를 찾았다는 것은 안정성
증명이 아니다 (교재 E05). 강한 배터리 모델의 극단적 민감도는 실제 회로의 저항·배터리 임피던스가 바꾼다.
