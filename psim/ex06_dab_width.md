# PSIM 이식표 — EX06 DAB width modulation (900 V ↔ 36 V, w₁ = 0.7π, 1.5 kW)

**상태: NOT_RUN_ENVIRONMENT** — PSIM 12.0.2가 이 환경에 없다. 이 시트는 손으로 회로를 그리기 위한 자료이고 PSIM 결과가 아니다. PSIM 12.0.2의 schematic 생성 API는 가정하지 않는다.

기준 실행: `python -m convlab run EX06 general_modulation --preset textbook` (app 4.0.0, request_hash `797c20c9559db4c0`, 실행 2026-09-30T19:53:54, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 C (정확 구간 적분) + A (SPS 닫힌 식))

교재 E06의 제한적 최적화 예제다. 같은 전력 1.5 kW에서 SPS(w₁ = w₂ = π)와 1차 bridge에 zero 구간을 넣은 candidate
(w₁ = 0.7π, w₂ = π)를 비교한다. 앱은 E06 switching function b(θ; w, φ)로 1차 환산 회로를 정확히 적분한다.

## 1. 회로

FL08 시트와 같은 위상구조(1차 full bridge – L – 이상 변압기 – 2차 full bridge)에 값만 다르다.

| 소자 | 종류 | 노드 (+ → −) | 값 | 값의 출처 |
|---|---|---|---|---|
| V_H | DC 전압원 | HV+ → HV− | 900 V | 입력 `V1` |
| S1…S4 | MOSFET 1차 leg A(S1/S2), B(S3/S4), 이상 | HV+ → A → HV−, HV+ → B → HV− | | |
| L | 인덕터 | A → T1 | 200 µH | 입력 `L` (L_σ) |
| TR | 이상 변압기, N_p:N_s = 50:3 | 1차 T1(점) – B, 2차 T2(점) – T3 | n = 16.6667 | 입력 `n` |
| S5…S8 | MOSFET 2차 leg C(S5/S6), D(S7/S8), 이상 | LV+ → T2 → LV−, LV+ → T3 → LV− | | |
| V_L | DC 전압원 | LV+ → LV− | 36 V (V₂′ = n·V_L = 600 V) | 입력 `V2` = 600 V (1차 환산) |

자화 인덕턴스·dead time·손실 없음 (앱 가정: 이상 bridge, 선형 L, zero-mean 전류 해).

## 2. Gate timing (f_s = 100 kHz, T = 10 µs, dead time 0)

**시간 원점은 FL08과 다르다.** E06 정의 b(θ; w, 0)는 v₁의 + pulse **중심**이 θ = 0이다 (앱 series도 같다).
v₁ = V_H(a − b), v₂′ = n·V_L(c − d). 하측 스위치는 상측의 보수.

candidate (w₁ = 0.7π = 126°, w₂ = π, φ_c = 0.4990164 rad = 28.59153°, pattern = alternating):

| leg | 상측 on [deg] | 상측 on [µs] |
|---|---|---|
| A (S1) | 297 → 117 (= −63 → 117) | 8.25 → 3.25 (다음 주기) |
| B (S3) | 63 → 243 | 1.75 → 6.75 |
| C (S5) | φ_c − 90 → φ_c + 90 = 298.59153 → 118.59153 | 8.294209 → 3.294209 (다음 주기) |
| D (S7) | 118.59153 → 298.59153 | 3.294209 → 8.294209 |

결과: v₁ = +900 V (297°→63°), 0 (63°→117°, 두 상측 on), −900 V (117°→243°), 0 (243°→297°, 두 하측 on).
zero state가 상측·하측으로 번갈아 오는 것이 `alternating` pattern이다. v₂′ = +600 V (298.59°→118.59°), −600 V (나머지).

SPS 비교 run (w₁ = w₂ = π, φ_sps = 0.3999939 rad = 22.91796°):

| leg | 상측 on [deg] | 상측 on [µs] |
|---|---|---|
| A (S1) | 270 → 90 | 7.5 → 2.5 (다음 주기) |
| B (S3) | 90 → 270 | 2.5 → 7.5 |
| C (S5) | 292.91796 → 112.91796 | 8.136610 → 3.136610 (다음 주기) |
| D (S7) | 112.91796 → 292.91796 | 3.136610 → 8.136610 |

## 3. 시뮬레이션 설정

| 항목 | 값 |
|---|---|
| time step | 1 ns |
| total time | 100 µs |
| 저장 시작 | 80 µs |
| 초기값 | candidate i_L(0) = 2.382628 A, SPS i_L(0) = 1.909830 A (series `i_c`, `i_sps`의 t = 0 값) |

## 4. Probe ↔ 앱 series

| PSIM probe | 측정 | 앱 series |
|---|---|---|
| I(L), candidate run | 1차 직렬 전류 | `i_c` |
| I(L), SPS run | 1차 직렬 전류 | `i_sps` |
| V(A) − V(B) | v₁ (3-level) | `v1` |
| V(T1) − V(B) | v₂′ | `v2` |
| S1, S3, S5, S7 gate 신호 | leg 상태 | `gA`, `gB`, `gC`, `gD` (그림용 offset이 붙어 있다: 높은 값 = 상측 on) |

## 5. 기대값 (Python 실제 실행)

`python -m convlab run EX06 general_modulation --preset textbook` (app 4.0.0, request_hash `797c20c9559db4c0`, 실행 2026-09-30T19:53:54, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 C (정확 구간 적분) + A (SPS 닫힌 식))

| metric | 뜻 | 값 | 단위 |
|---|---|---|---|
| `phi_sps` | SPS φ | 0.3999939 | rad |
| `irms_sps` | SPS I_rms | 3.113563 | A |
| `ipk_sps` | SPS I_pk | 5.65983 | A |
| `phi_c` | candidate φ (가장 작은 해) | 0.4990164 | rad |
| `P_c` | candidate 전달전력 | 1500 | W |
| `irms_c` | candidate I_rms | 2.893092 | A |
| `ipk_c` | candidate I_pk | 5.007628 | A |
| `drms` | RMS 변화 | -0.07081001 |  |
| `dpk` | peak 변화 | -0.1152336 |  |

교재 E06 표: SPS φ 0.399994 rad, I_rms 3.113563 A, I_pk 5.659830 A; candidate φ 0.499016 rad, I_rms 2.893092 A,
I_pk 5.007628 A; RMS −7.08 %, peak −11.52 %.

## 6. 비교

두 run에서 전달전력 (1/T)∫v₂′·i₂ dt가 1500 W인지 먼저 확인하고(φ가 맞다는 증거), I_rms·I_pk를 비교한다.

## 7. 이 회로가 말하지 않는 것

global optimum, ZVS 달성, 전체 효율 개선은 이 비교로 말할 수 없다 (교재 E06). zero state 선택에 따라 소자별
commutation은 달라진다. dead time·Coss를 넣은 판단은 EX06의 commutation 실험과 EX02의 방법으로 따로 한다.
