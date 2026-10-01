# PSIM 이식표 — FL10 CLLC (n = 0.93, link 850 V → battery 920 V, 두 FHA 분기)

**상태: NOT_RUN_ENVIRONMENT** — PSIM 12.0.2가 이 환경에 없다. 이 시트는 손으로 회로를 그리기 위한 자료이고 PSIM 결과가 아니다. PSIM 12.0.2의 schematic 생성 API는 가정하지 않는다.

기준 실행:
- 스위칭 해 (강한 배터리, 손실 0): `python -m convlab run FL10 time_domain --preset textbook` (app 4.0.0, request_hash `ddb87128e4163f80`, 결과 상태 FAIL_CONSTRAINT, 모델 수준 C (이상 스위칭, 강한 배터리) + A)
- 스위칭 해 (R₁ = R₂′ = 50 mΩ): `python -m convlab run FL10 time_domain --preset lossy` (app 4.0.0, request_hash `746acd3f7d09f297`, 결과 상태 FAIL_CONSTRAINT, 모델 수준 C (이상 스위칭, 강한 배터리) + A)
- FHA 해: `python -m convlab run FL10 fix_n093 --preset textbook` (app 4.0.0, request_hash `0dcca4257bed3a12`, 결과 상태 UNRESOLVED_RANKING, 모델 수준 A (FHA))
- 출력 C·R 부하 (EX05): `python -m convlab run EX05 operating_points --preset textbook` (app 4.0.0, request_hash `0c2725755e2c5df9`, 결과 상태 OUT_OF_VALIDITY, 모델 수준 C (스위칭 주기해) + A (FHA))

교재 13장 수정 후보 A다. seed(n = 1)는 920/850 V corner에서 FHA 해가 없다 (필요 gain 1.082353 > inductive 최대
1.016401). n = 0.93으로 바꾸고 1차 L/C와 환산 대칭을 유지하면 136.099 kHz와 147.061 kHz 두 inductive FHA 해가 생긴다.
**앱의 스위칭 모델(이상 bridge·다이오드, 강한 link·배터리)은 이 두 주파수에서 11 kW가 아니라 17.34 kW와
22.24 kW를 주고, 같은 위 branch에서 11 kW는 148.0049 kHz (148004.8618 Hz)의 매우 가파른 곳
(dP/df ≈ -1675 W/Hz)에 있다. 그 점의 Floquet 승수 최대값은 0.9999837로 거의 중립 안정이다.**
이 시트는 그 결과를 PSIM에서 다시 보기 위한 것이다.

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

| leg | 상측 on [deg] | 분기 1: f = 136099.4667 Hz (T = 7.347567 µs) | 분기 2: f = 147060.8627 Hz (T = 6.799906 µs) | 11 kW 점: f = 148004.8618 Hz (T = 6.756535 µs) |
|---|---|---|---|---|
| A (S1; S2 보수) | 0 → 180 | 0 → 3.673784 µs | 0 → 3.399953 µs | 0 → 3.378267 µs |
| B (S3; S4 보수) | 180 → 360 | 3.673784 → 7.347567 µs | 3.399953 → 6.799906 µs | 3.378267 → 6.756535 µs |

11 kW 점은 1 Hz가 15 % 전력이다. PSIM의 주파수는 0.001 Hz까지 넣고, 같은 주파수에서 비교한다. t = 0은 v₁ 상승 edge다.

## 3. 시뮬레이션 설정

| 항목 | 값 |
|---|---|
| time step | 1 ns 이하 (11 kW 점은 0.2 ns 권장: 전력이 edge 시각에 매우 민감) |
| total time | 10 ms (가장 느린 Floquet 승수 0.9891 → 1e-4까지 약 840주기 ≈ 5.7 ms) |
| 저장 | 마지막 20주기 |
| 초기값 (11 kW 점, series t = 0) | i_Lr1 = -7.022349 A, i_Lm = -7.022349 A, i_Lr2 = 0 (정류 off), v_Cr1 = -776.6769 V, v_Cr2 = -829.6704 V (실제; 1차 환산 -771.5935 V) |

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

| corner | FHA 해 | branch | 스위칭 전력 | I₁,rms | 정류 off | Floquet \|λ\|max |
|---|---|---|---|---|---|---|
| 저전압 650/700 | 164.390 kHz | peak 위 (−기울기) | 9.13 kW (-17.0 %) | 17.61 A (FHA 21.05) | 0.0 % | 0.9537 |
| 중간 800/800 | 162.811 kHz | peak 위 (−기울기) | 7.63 kW (-30.6 %) | 12.38 A (FHA 17.23) | 0.0 % | 0.9314 |
| 고전압 920/850 | 136.099 kHz | peak 아래 (+기울기) | 17.34 kW (+57.6 %) | 23.38 A (FHA 14.39) | 0.0 % | 0.9224 |
| 고전압 920/850 | 147.061 kHz | peak 위 (−기울기) | 22.24 kW (+102.2 %) | 29.37 A (FHA 14.77) | 0.0 % | 0.9891 |

| corner | branch | FHA 해 | 스위칭 동작점 | 그 점의 전력 | I₁,rms | dP/df | Floquet \|λ\|max | ZVS screen |
|---|---|---|---|---|---|---|---|---|
| 저전압 650/700 | peak 위 (−기울기) | 164.390 kHz | 162422.055 Hz | 11.0000 kW | 21.00 A | -1.1 kW/kHz | 0.959727 | 부호·전하 screen 통과 |
| 중간 800/800 | peak 위 (−기울기) | 162.811 kHz | 160262.166 Hz | 11.0000 kW | 17.24 A | -1.7 kW/kHz | 0.945623 | 부호·전하 screen 통과 |
| 고전압 920/850 | peak 아래 (+기울기) | 136.099 kHz | 범위 내 없음 | 범위 끝 120 kHz까지 11 kW에 도달하지 않음 (P = 13.16 kW) | - | - |  |  |
| 고전압 920/850 | peak 위 (−기울기) | 147.061 kHz | 148004.862 Hz | 11.0000 kW | 15.08 A | -1675.3 kW/kHz | 0.999984 | 부호·전하 screen 통과 |

| metric | 뜻 | 값 |
|---|---|---|
| `P_td_lo` | 136.099 kHz FHA 해의 스위칭 전력 | 17341.45 W |
| `P_td_hi` | 147.061 kHz FHA 해의 스위칭 전력 | 22242.00 W |
| `f_sw_hi` | 위 branch 11 kW 주파수 (Brent로 \|P − 11 kW\| ≤ 0.1 %까지) | 148004.8618014 Hz |
| `P_sw_hi` | 그 주파수에서 다시 푼 스위칭 전력 | 11000.009 W |
| `sens_hi` | 그 점의 dP/df (±2 Hz 중앙차분) | -1675345.5 W/kHz |
| `rho_sw_hi` | 그 점의 Floquet \|λ\|max | 0.9999837 |
| `vC1_pk` | 그 점의 C_r1 전압 peak | 817.532 V |
| `vC2_pk` | 그 점의 C_r2 전압 peak (1차 환산) | 771.593 V (실제 829.7 V) |

### 5.2 R₁ = R₂′ = 50 mΩ (`FL10 time_domain --preset lossy`)

`python -m convlab run FL10 time_domain --preset lossy` (app 4.0.0, request_hash `746acd3f7d09f297`, 결과 상태 FAIL_CONSTRAINT, 모델 수준 C (이상 스위칭, 강한 배터리) + A)

| corner | FHA 해 | branch | 스위칭 전력 | I₁,rms | 정류 off | Floquet \|λ\|max |
|---|---|---|---|---|---|---|
| 저전압 650/700 | 164.202 kHz | peak 위 (−기울기) | 9.16 kW (-16.7 %) | 17.65 A (FHA 21.05) | 0.0 % | 0.9535 |
| 중간 800/800 | 162.540 kHz | peak 위 (−기울기) | 7.72 kW (-29.8 %) | 12.49 A (FHA 17.23) | 0.0 % | 0.9316 |
| 고전압 920/850 | 137.183 kHz | peak 아래 (+기울기) | 17.40 kW (+58.2 %) | 23.46 A (FHA 14.43) | 0.0 % | 0.9217 |
| 고전압 920/850 | 145.905 kHz | peak 위 (−기울기) | 19.08 kW (+73.4 %) | 25.38 A (FHA 14.73) | 0.0 % | 0.9784 |

| corner | branch | FHA 해 | 스위칭 동작점 | 그 점의 전력 | I₁,rms | dP/df | Floquet \|λ\|max | ZVS screen |
|---|---|---|---|---|---|---|---|---|
| 저전압 650/700 | peak 위 (−기울기) | 164.202 kHz | 162260.825 Hz | 11.0000 kW | 20.99 A | -1.1 kW/kHz | 0.959428 | 부호·전하 screen 통과 |
| 중간 800/800 | peak 위 (−기울기) | 162.540 kHz | 160051.606 Hz | 11.0000 kW | 17.23 A | -1.6 kW/kHz | 0.945349 | 부호·전하 screen 통과 |
| 고전압 920/850 | peak 아래 (+기울기) | 137.183 kHz | 범위 내 없음 | 범위 끝 120 kHz까지 11 kW에 도달하지 않음 (P = 13.10 kW) | - | - |  |  |
| 고전압 920/850 | peak 위 (−기울기) | 145.905 kHz | 147442.165 Hz | 11.0000 kW | 15.11 A | -19.4 kW/kHz | 0.995819 | 부호·전하 screen 통과 |

| metric | 뜻 | 값 |
|---|---|---|
| `P_td_lo` | 137.183 kHz FHA 해의 스위칭 전력 | 17401.57 W |
| `P_td_hi` | 145.905 kHz FHA 해의 스위칭 전력 | 19075.77 W |
| `f_sw_hi` | 위 branch 11 kW 주파수 (Brent로 \|P − 11 kW\| ≤ 0.1 %까지) | 147442.1646241 Hz |
| `P_sw_hi` | 그 주파수에서 다시 푼 스위칭 전력 | 11000.000 W |
| `sens_hi` | 그 점의 dP/df | -19378.4 W/kHz |
| `rho_sw_hi` | 그 점의 Floquet \|λ\|max | 0.9958189 |
| `vC1_pk` | 그 점의 C_r1 전압 peak | 821.948 V |
| `vC2_pk` | 그 점의 C_r2 전압 peak (1차 환산) | 774.538 V |

손실 50 mΩ만 넣어도 11 kW 점의 dP/df가 -1675에서 -19.4 kW/kHz로, Floquet 승수 최대값이 0.999984에서 0.995819로 줄어든다. 무손실 이상 모델의 가파름은 f_r 근처의
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
(OUT_OF_VALIDITY). 같은 출력이라도 제어 부호가 바뀐다는 뜻이다. 이 C·R 부하의 V_o = 920 V 운전점 148005.2304 Hz는
강한 배터리의 11 kW 점 148004.8618 Hz와 0.37 Hz 다르다. 강한 배터리 모델의 기울기로
0.37 Hz는 약 618 W이므로 두 운전점을 이 정밀도에서 바꿔 쓰지 않는다.

### 5.4 Octave 독립 검산 (`matlab/xc_cllc_td.m`, 참고)

앱 코드를 쓰지 않은 hybrid `ode45` 모델(교재 E05 상태식, 3-상태 이상 다이오드 정류기, Newton shooting)로 같은 점을
다시 풀었다. 위 표의 값과 모두 허용오차 안에서 같다 (`matlab/results/octave_crosscheck.json`, 2026-10-01T00:57:38,
220 PASS / 0 FAIL; `FL10.td_*`, `EX05.op.*`). 앱의 11 kW 점에서:

| 점 | 전력 | I₁ RMS | v_Cr1 peak | v_Cr2′ peak | 정류 off | Floquet \|λ\|max |
|---|---|---|---|---|---|---|
| 무손실, 앱 148004.8618014 Hz | 앱 11000.009 W / Octave 11000.006 W | 앱 15.08 A / Octave 15.0775 A | 앱 817.5324 V / Octave 817.5322 V | 앱 771.5935 V / Octave 771.5933 V | Octave 1.32 % | 앱 0.9999837 / Octave 0.9999837 |
| lossy, 앱 147442.1646241 Hz | 앱 11000.000 W / Octave 11000.000 W | 앱 15.11 A / Octave 15.1066 A | 앱 821.9483 V / Octave 821.9483 V | 앱 774.5375 V / Octave 774.5375 V | Octave 1.70 % | 앱 0.9958189 / Octave 0.9958189 |

Octave가 따로 찾은 11 kW 주파수는 148004.8618028 Hz (무손실), 147442.1646089 Hz (lossy)로
앱과 1.3e-06 Hz, 1.5e-05 Hz 다르다. 정류 off와 Floquet 열의 Octave 값은
같은 함수(`xc_cllc_orbit`, 중앙차분 monodromy)로 앱의 주파수에서 따로 계산했다. 무손실 점은 교란이 1/e로 줄어드는 데
약 61249주기가 걸린다 (앱 verdict와 같다). PSIM에서 이 점을 볼 때는 같은 주파수를 넣고 C 전압 peak로 비교하는 편이 낫다.

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
앱의 `f_sw_hi`는 Brent로 \|P − 11 kW\| ≤ 0.1 %까지 다듬은 값이다 (그 점에서 다시 푼 전력 `P_sw_hi` = 11000.009 W).

## 7. 이 회로가 말하지 않는 것

ZVS(Coss·dead time), SR 타이밍, 기동, 역방향, 공차, 배터리 내부저항, 폐루프는 없다. 주기해를 찾았다는 것은 안정성
증명이 아니다 (교재 E05). 강한 배터리 모델의 극단적 민감도는 실제 회로의 저항·배터리 임피던스가 바꾼다.
