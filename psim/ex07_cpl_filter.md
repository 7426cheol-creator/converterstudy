# PSIM 이식표 — EX07 입력 R-L-C 필터와 이상 CPL (400 V, 10 kW)

**상태: NOT_RUN_ENVIRONMENT** — PSIM 12.0.2가 이 환경에 없다. 이 시트는 손으로 회로를 그리기 위한 자료이고 PSIM 결과가 아니다. PSIM 12.0.2의 schematic 생성 API는 가정하지 않는다.

기준 실행:
- C = 100 µF (불안정): `python -m convlab run EX07 cpl_exact --preset c100u` (app 4.0.0, request_hash `d6942c6d1e9b676f`, 실행 2026-09-30T19:53:57, 결과 상태 UNSTABLE, 모델 수준 A (해석 극점) + 비선형 ODE + 수치 Jacobian)
- C = 1 mF (안정): `python -m convlab run EX07 cpl_exact --preset c1m` (app 4.0.0, request_hash `be3638aadc04fdd8`, 실행 2026-09-30T19:53:57, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 A (해석 극점) + 비선형 ODE + 수치 Jacobian)

교재 E07의 완전히 유도되는 constant-power-load 예제다. L i′ = V_s − R i − v, C v′ = i − P/v. 평형점은 정적으로 존재해도
C < L·g/R = 320.566 µF이면 동적으로 불안정하다.

## 1. 회로

| 소자 | 종류 | 노드 (+ → −) | 값 | 값의 출처 |
|---|---|---|---|---|
| V_s | DC 전압원 | S → 0 | 400 V | 입력 `Vs` |
| R | 저항 | S → M | 0.2 Ω | 입력 `R` |
| L | 인덕터 | M → X | 1 mH, 초기 전류 25.32057 A (= I_e) | 입력 `L`; 초기값 export `Ie` |
| C | 커패시터 | X → 0 | 100 µF 또는 1 mF, 초기 전압 395.9359 V (= V_e + 1 V) | 입력 `C`, `dv0`; 초기값 export `Ve` + 1 V |
| VS_X | 전압 센서 | X → 0 | gain 1 | |
| F | 수식 블록 | 입력 v | i = P/v (v ≥ 200 V), 200 V 아래는 P·v/200² | 입력 `P` = 10 kW, `v_min_frac` = 0.5 |
| I_CPL | 제어 전류원 | X → 0 (X에서 빠져나가는 전류) | F 출력 | 이상 CPL (제어 대역 무한) |

## 2. Gate timing

스위치가 없다. 교란은 초기값 v_C(0) = V_e + 1 V 하나다 (앱과 같은 1 V).

## 3. 시뮬레이션 설정

| 항목 | c100u | c1m |
|---|---|---|
| time step | 0.2 µs | 0.2 µs |
| total time | 19.97 ms (export `t_exit`: \|Δv\|가 20 % V_e에 닿는 시각) | 30 ms |
| 저장 | 전 구간 | 전 구간 |

c100u는 t_exit 뒤에 이상 CPL 모델의 유효범위를 벗어나므로 더 돌리지 않는다. 센서 → 수식 → 제어 전원 사이의 한 step
지연(0.2 µs)은 500 Hz에서 0.04° 위상이라 무시할 수 있다.

## 4. Probe ↔ 앱 series

| PSIM probe | 측정 | 앱 series |
|---|---|---|
| V(X) − 394.9359 V | 평형점에서 벗어난 전압 Δv | `dv_nl` (비선형), `dv_lin` (선형 e^(At)x₀) |
| I(L) − 25.32057 A | 평형점에서 벗어난 전류 Δi | `di_nl` |

## 5. 기대값 (Python 실제 실행)

`python -m convlab run EX07 cpl_exact --preset c100u` (app 4.0.0, request_hash `d6942c6d1e9b676f`, 실행 2026-09-30T19:53:57, 결과 상태 UNSTABLE, 모델 수준 A (해석 극점) + 비선형 ODE + 수치 Jacobian)

| metric | 뜻 | 값 | 단위 |
|---|---|---|---|
| `Ve` | 고전압 평형 V_e | 394.9359 | V |
| `Ie` | 평형 전류 I_e = P/V_e | 25.32057 | A |
| `Rinc` | 증분 입력저항 dv/di = −V_e²/P | -15.59744 | Ω |
| `Ccrit` | 임계 C = L·g/R (tr A = 0) | 0.0003205655 | F |
| `pole_re` | 극점 실수부 σ | 220.5655 | 1/s |
| `pole_im` | 극점 허수부 ω_d | 3134.186 | rad/s |
| `f0` | 진동 주파수 ω_d/2π | 498.8213 | Hz |
| `sig_nl` | 비선형 적분에서 측정한 성장률 σ (봉우리 fit) | 220.5655 | 1/s |
| `wd_nl` | 비선형 적분에서 측정한 ω_d (봉우리 간격) | 3134.549 | rad/s |
| `t_exit` | \|Δv\|가 20 % V_e를 넘은 시각 (여기서 적분 중단) | 0.01996954 | s |

`python -m convlab run EX07 cpl_exact --preset c1m` (app 4.0.0, request_hash `be3638aadc04fdd8`, 실행 2026-09-30T19:53:57, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 A (해석 극점) + 비선형 ODE + 수치 Jacobian)

| metric | 뜻 | 값 | 단위 |
|---|---|---|---|
| `Ve` | 고전압 평형 V_e | 394.9359 | V |
| `Ie` | 평형 전류 I_e = P/V_e | 25.32057 | A |
| `Ccrit` | 임계 C = L·g/R (tr A = 0) | 0.0003205655 | F |
| `pole_re` | 극점 실수부 σ | -67.94345 | 1/s |
| `pole_im` | 극점 허수부 ω_d | 991.2422 | rad/s |
| `f0` | 진동 주파수 ω_d/2π | 157.7611 | Hz |
| `sig_nl` | 비선형 적분에서 측정한 성장률 σ (봉우리 fit) | -67.94346 | 1/s |
| `wd_nl` | 비선형 적분에서 측정한 ω_d (봉우리 간격) | 991.2387 | rad/s |

파형 끝값 (series `dv_nl`): c100u는 t = 19.97 ms에 Δv = 78.99 V (여기서 중단), c1m은 30 ms에 Δv = −0.0313 V.
교재 E07: V_e 394.9359 V, 증분 입력저항 −15.5974 Ω, 임계 C 320.566 µF, 극점 220.57 ± j3134.19 s⁻¹ (100 µF),
−67.94 ± j991.24 s⁻¹ (1 mF).

## 6. 비교

Δv의 극값 중 |Δv| < 5 % V_e인 것들로 ln|Δv|의 기울기(성장률 σ)와 극값 간격(π/ω_d)을 구해 `pole_re`, `pole_im`과
비교한다 (2 %). c100u에서 Δv가 20 % V_e에 닿는 시각을 `t_exit`와 비교한다.

## 7. 이 회로가 말하지 않는 것

이상 CPL은 제어 대역이 무한한 부하다. 실제 converter의 입력 임피던스는 대역 밖에서 달라진다 (EX07의
converter_impedance 실험). 이 결과는 국소(local) 안정성이며, 큰 교란의 전압 붕괴를 예측하지 않는다.
