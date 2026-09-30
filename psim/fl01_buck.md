# PSIM 이식표 — FL01 동기 Buck (48 V → 12 V, 5 A)

**상태: NOT_RUN_ENVIRONMENT** — PSIM 12.0.2가 이 환경에 없다. 이 시트는 손으로 회로를 그리기 위한 자료이고 PSIM 결과가 아니다. PSIM 12.0.2의 schematic 생성 API는 가정하지 않는다.

기준 실행: `python -m convlab run FL01 buck_ccm --preset nominal` (app 4.0.0, request_hash `7d143cfc5cac8cd7`, 실행 2026-09-30T19:53:52, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 C (+B 평균모델 비교))

교재 03장의 손계산 예제 회로다. 앱은 이 회로를 이상 스위치로 풀어 주기해를 찾는다.

## 1. 회로

| 소자 | 종류 | 노드 (+ → −) | 값 | 값의 출처 |
|---|---|---|---|---|
| V_in | DC 전압원 | IN → 0 | 48 V | 입력 `Vin` (교재 03장) |
| Q1 | MOSFET, 상측 (이상: R_on = 0, 역병렬 diode 이상) | drain IN, source SW | gate 표 참고 | 입력 `D` = 0.25 |
| Q2 | MOSFET, 하측 동기정류 (같은 이상 조건) | drain SW, source 0 | gate 표 참고 | 입력 `rect` = sync |
| L | 인덕터 | SW → OUT | 100 µH, 초기 전류 4.549968 A | 입력 `L`; 초기값은 export `I_valley` |
| C | 커패시터 | OUT → N_ESR | 100 µF, 초기 전압 11.99624 V | 입력 `C`; 초기값 v_C(0) = v_o(0)(1 + ESR/R) − ESR·i_L(0), v_o(0) = 11.99177 V (series `vo`) |
| ESR | 저항 | N_ESR → 0 | 10 mΩ | 입력 `esr` (C 가지에 직렬) |
| R_load | 저항 | OUT → 0 | 2.4 Ω | 앱 가정 R = D·V_in/I_o (입력 `Io` = 5 A) |

DCR, R_DS(on)은 0 (입력 `dcr`, `rds_hi`, `rds_lo`). 다이오드 V_f(0.7 V)는 `rect` = diode일 때만 쓰므로 이 회로에는 없다.

## 2. Gate timing (f_s = 100 kHz, T = 10 µs, dead time 0)

| 스위치 | on 구간 [deg] | on 구간 [µs] |
|---|---|---|
| Q1 | 0 → 90 | 0 → 2.5 |
| Q2 | 90 → 360 | 2.5 → 10 |

t = 0은 Q1 turn-on이다 (앱 series `vsw`가 t = 0부터 48 V).

## 3. 시뮬레이션 설정

| 항목 | 정상상태 비교 | 기동 비교 (선택) |
|---|---|---|
| time step | 5 ns (주기당 2000 step) | 10 ns |
| total time | 200 µs | 4 ms |
| 저장 시작 (print time) | 180 µs (마지막 2주기) | 0 |
| 초기값 | 표 1의 i_L(0), v_C(0) | i_L = 0, v_C = 0 |

초기값을 주기해로 주면 첫 주기부터 정상상태다. 초기값을 줄 수 없으면 0에서 시작해 10 ms 이상 돌리고
마지막 2주기를 본다 (LC 감쇠 시정수 약 0.5 ms).

## 4. Probe ↔ 앱 series

| PSIM probe | 측정 | 앱 series | 부호 |
|---|---|---|---|
| I(L) | 인덕터 전류, SW → OUT | `iL` | 같음 |
| V(SW) | 스위치 노드 | `vsw` | 같음 |
| V(OUT) | 출력 전압 | `vo` | 같음 |
| I(Q1) | 상측 drain 전류 | `i_hi` | 같음 |
| I(Q2) | 하측 소자 전류 | `i_lo` | 앱은 환류 전류(0 → SW)를 +로 센다. MOSFET probe가 drain → source를 +로 재면 부호를 뒤집는다 |
| I(R_load) | 부하 전류 | `io` | 같음 |
| V(SW) − V(OUT) | 인덕터 전압 | `vL` | 같음 |
| 기동 run의 I(L), V(OUT) | 0에서 시작한 과도 | `su_iL`, `su_vo` | 같음 |

## 5. 기대값 (Python 실제 실행)

`python -m convlab run FL01 buck_ccm --preset nominal` (app 4.0.0, request_hash `7d143cfc5cac8cd7`, 실행 2026-09-30T19:53:52, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 C (+B 평균모델 비교))

| metric | 뜻 | 값 | 단위 |
|---|---|---|---|
| `Vo` | 출력전압 평균 V_o | 12 | V |
| `IL_avg` | 인덕터 평균전류 | 5 | A |
| `dI_pp` | 인덕터 리플 ΔI_pp | 0.9001395 | A |
| `I_peak` | 인덕터 peak | 5.450108 | A |
| `I_valley` | 인덕터 valley | 4.549968 | A |
| `IL_rms` | 인덕터 RMS | 5.006749 | A |
| `Q1_avg` | 상측 스위치 평균전류 (= 입력 DC 전류) | 1.250014 | A |
| `Q1_rms` | 상측 스위치 RMS | 2.503402 | A |
| `Q2_avg` | 하측 소자 평균전류 | 3.749986 | A |
| `Q2_rms` | 하측 소자 RMS | 4.335955 | A |
| `dvo_pp` | 출력전압 리플 Δv_o (pp) | 0.01360318 | V |
| `I_boundary` | CCM/DCM 경계 부하전류 (ΔI/2) | 0.45 | A |
| `mode` | 전도 모드 | CCM |  |

기동 (같은 실행의 series `su_iL`, `su_vo`, 4 ms, 1.33 µs 간격 샘플): i_L 최대 13.59645 A (t ≈ 182.5 µs), v_o 최대
18.05883 V (t ≈ 315.5 µs), 4 ms에서 i_L 4.551945 A, v_o 11.99072 V. 샘플 간격 때문에 최대값은 리플 꼭짓점을 조금
놓칠 수 있으므로 같은 간격으로 뽑아 비교하거나 2 % 여유를 둔다.

교재 손계산과의 관계: ΔI = (V_in − V_o)D/(L f_s) = 0.9 A는 v_o를 상수로 본 식이고, 스위칭 회로에서는 C 리플
때문에 0.9001395 A가 된다. 용량성 리플 11.25 mV와 ESR 리플 9 mV는 위상이 달라 합이 20.25 mV가 아니라
13.603 mV다 (교재 03장의 경고).

## 6. 비교

PSIM에서 마지막 한 주기로 평균·RMS·peak를 구하고 README의 허용 범위로 비교한다. Q2 전류는 부호 약속을 맞춘 뒤
비교한다.

## 7. 이 회로가 말하지 않는 것

스위칭 손실, dv/dt, ringing, ZVS, 코어 포화, 폐루프 응답은 없다 (앱의 `not_valid_for`와 같다). 기동은 0 V
출력에 이상 전원을 바로 거는 수학적 기동이다.
