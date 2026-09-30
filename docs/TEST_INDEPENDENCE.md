# 테스트 독립성 지도 (Test independence map)

생성: `tools/gen_docs.py` · run-all 2026-09-30T19:46:57

**독립** = 서로 다른 식/적분기/표현으로 계산한 두 경로의 비교 (`Check.independent = True`). **회귀** = 같은 코드 경로의 재현성 확인. 교재 기준값은 테스트 파일에 숫자로 직접 적혀 있다 (`reference/` 모듈을 다시 부르지 않는다).


## FL01 · Buck와 Boost — 컨버터의 문법


**buck_ccm** — 독립 6 · 회귀 6

- [PASS] (nominal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (nominal) 독립 solver (RK45) 허용오차 수렴 — 손으로 쓴 ODE(행렬 미사용)를 RK45로 rtol 1e-6→1e-8→1e-10 적분한 주기 끝 상태 vs 정확 해
- [PASS] (half_fs) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (half_fs) 독립 solver (RK45) 허용오차 수렴 — 손으로 쓴 ODE(행렬 미사용)를 RK45로 rtol 1e-6→1e-8→1e-10 적분한 주기 끝 상태 vs 정확 해
- [PASS] (lossy) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (lossy) 독립 solver (RK45) 허용오차 수렴 — 손으로 쓴 ODE(행렬 미사용)를 RK45로 rtol 1e-6→1e-8→1e-10 적분한 주기 끝 상태 vs 정확 해
- [회귀 PASS] (nominal) shooting 주기해 잔차
- [회귀 INFO] (nominal) 주기별 정상상태 기준 (상태변화 < 1e-6, \|ΔW\|/E_ref < 1e-6, 3주기 연속)
- [회귀 PASS] (half_fs) shooting 주기해 잔차
- [회귀 INFO] (half_fs) 주기별 정상상태 기준 (상태변화 < 1e-6, \|ΔW\|/E_ref < 1e-6, 3주기 연속)
- [회귀 PASS] (lossy) shooting 주기해 잔차
- [회귀 INFO] (lossy) 주기별 정상상태 기준 (상태변화 < 1e-6, \|ΔW\|/E_ref < 1e-6, 3주기 연속)

**buck_light_load** — 독립 1 · 회귀 0

- [PASS] (light) 부하 sweep: 스위칭 V_o vs 해석식 (최대 상대오차) — 각 부하점의 정확 주기해 vs 교과서 DCM/CCM 변환비 식

**buck_ripple** — 독립 0 · 회귀 0


**boost_rhpz** — 독립 2 · 회귀 0

- [PASS] (nominal) RHP zero 부호: duty 증가 직후 출력 감소 — 스위칭 모델 주기평균(C)과 평균모델(B)이 모두 초기 역응답을 보이는지
- [PASS] (light) RHP zero 부호: duty 증가 직후 출력 감소 — 스위칭 모델 주기평균(C)과 평균모델(B)이 모두 초기 역응답을 보이는지

## FL05 · OBC·PFC — 전력품질을 설계 변수로 바꾸기


**grid_boundary** — 독립 4 · 회귀 0

- [PASS] (nominal) 선전류: 시간영역 Σv·i 경로 (PF를 변위각으로) vs √3 식 — 상별 정현파 v_x·i_x를 한 주기 평균해 P_bat/η가 되는 전류를 brentq로 풀기
- [PASS] (nominal) 선전류: 시간영역 경로 (PF를 왜곡으로, cosφ₁ = 1, THD = √(1/PF²−1)) vs √3 식 — 5·7차 고조파로 PF 0.995를 만든 전류의 RMS
- [PASS] (svpwm) 선전류: 시간영역 Σv·i 경로 (PF를 변위각으로) vs √3 식 — 상별 정현파 v_x·i_x를 한 주기 평균해 P_bat/η가 되는 전류를 brentq로 풀기
- [PASS] (svpwm) 선전류: 시간영역 경로 (PF를 왜곡으로, cosφ₁ = 1, THD = √(1/PF²−1)) vs √3 식 — 5·7차 고조파로 PF 0.995를 만든 전류의 RMS

**pf_thd** — 독립 4 · 회귀 0

- [PASS] (nominal) THD: DFT (정수 창) vs 구성한 고조파 — rfft bin h·n vs √(Σa_h²)/a_1
- [PASS] (nominal) PF: 시간영역 평균 vs 해석식 — mean(v·i)/(√mean v²·√mean i²) vs 성분별 해석 적분
- [PASS] (offset) THD: DFT (정수 창) vs 구성한 고조파 — rfft bin h·n vs √(Σa_h²)/a_1
- [PASS] (offset) PF: 시간영역 평균 vs 해석식 — mean(v·i)/(√mean v²·√mean i²) vs 성분별 해석 적분

**boost_pfc** — 독립 24 · 회귀 6

- [PASS] (nominal) 에너지 잔차 (E_grid − E_load − E_loss − ΔW), 마지막 1주기 — v_g·i_g, v_C²/R, R_L·i_L², ½Li² + ½Cv²의 정확 2차 모멘트 적분 (상태식과 독립인 포트 전력 정의)
- [PASS] (nominal) 기본파: Hermite–GL Fourier vs 정확 모멘트 ∫i_g·s, ∫i_g·c (마지막 1주기) — 구간 끝 정확값·미분의 3차 Hermite + Gauss–Legendre vs 발진기 상태 s, c를 쓴 엔진 2차 모멘트
- [PASS] (nominal) I_rms: Hermite–GL vs 정확 모멘트 (마지막 1주기) — 같은 두 경로
- [PASS] (nominal) 입력 전력: Hermite × v_g vs 정확 모멘트 (마지막 1주기) — 같은 두 경로
- [PASS] (nominal) DC-link 2ω 리플: 스위칭 모델 vs P/(ωCV_dc) — 스위칭 해의 2차 Fourier 성분(C) vs 작은 리플 해석식(A)
- [PASS] (nominal) 평균모델(B) vs 스위칭(C): DC-link 평균 — 손으로 쓴 평균 ODE + RK4 vs 행렬지수 스위칭 해 (같은 제어기 코드)
- [PASS] (nominal) 평균모델(B) vs 스위칭(C): 기본파 전류 (B vs C) — 손으로 쓴 평균 ODE + RK4 vs 행렬지수 스위칭 해 (같은 제어기 코드)
- [PASS] (nominal) 평균모델(B) vs 스위칭(C): 2ω 리플 pp (B vs C) — 손으로 쓴 평균 ODE + RK4 vs 행렬지수 스위칭 해 (같은 제어기 코드)
- [PASS] (fast_v) 에너지 잔차 (E_grid − E_load − E_loss − ΔW), 마지막 1주기 — v_g·i_g, v_C²/R, R_L·i_L², ½Li² + ½Cv²의 정확 2차 모멘트 적분 (상태식과 독립인 포트 전력 정의)
- [PASS] (fast_v) 기본파: Hermite–GL Fourier vs 정확 모멘트 ∫i_g·s, ∫i_g·c (마지막 1주기) — 구간 끝 정확값·미분의 3차 Hermite + Gauss–Legendre vs 발진기 상태 s, c를 쓴 엔진 2차 모멘트
- [PASS] (fast_v) I_rms: Hermite–GL vs 정확 모멘트 (마지막 1주기) — 같은 두 경로
- [PASS] (fast_v) 입력 전력: Hermite × v_g vs 정확 모멘트 (마지막 1주기) — 같은 두 경로
- [INFO] (fast_v) DC-link 2ω 리플: 스위칭 모델 vs P/(ωCV_dc) — C vs A
- [PASS] (fast_v) 평균모델(B) vs 스위칭(C): DC-link 평균 — 손으로 쓴 평균 ODE + RK4 vs 행렬지수 스위칭 해 (같은 제어기 코드)
- [PASS] (fast_v) 평균모델(B) vs 스위칭(C): 기본파 전류 (B vs C) — 손으로 쓴 평균 ODE + RK4 vs 행렬지수 스위칭 해 (같은 제어기 코드)
- [PASS] (fast_v) 평균모델(B) vs 스위칭(C): 2ω 리플 pp (B vs C) — 손으로 쓴 평균 ODE + RK4 vs 행렬지수 스위칭 해 (같은 제어기 코드)
- [PASS] (light) 에너지 잔차 (E_grid − E_load − E_loss − ΔW), 마지막 1주기 — v_g·i_g, v_C²/R, R_L·i_L², ½Li² + ½Cv²의 정확 2차 모멘트 적분 (상태식과 독립인 포트 전력 정의)
- [PASS] (light) 기본파: Hermite–GL Fourier vs 정확 모멘트 ∫i_g·s, ∫i_g·c (마지막 1주기) — 구간 끝 정확값·미분의 3차 Hermite + Gauss–Legendre vs 발진기 상태 s, c를 쓴 엔진 2차 모멘트
- [PASS] (light) I_rms: Hermite–GL vs 정확 모멘트 (마지막 1주기) — 같은 두 경로
- [PASS] (light) 입력 전력: Hermite × v_g vs 정확 모멘트 (마지막 1주기) — 같은 두 경로
- [INFO] (light) DC-link 2ω 리플: 스위칭 모델 vs P/(ωCV_dc) — C vs A
- [INFO] (light) 평균모델(B) vs 스위칭(C): DC-link 평균 — 손으로 쓴 평균 ODE + RK4 vs 행렬지수 스위칭 해 (같은 제어기 코드)
- [INFO] (light) 평균모델(B) vs 스위칭(C): 기본파 전류 (B vs C) — 손으로 쓴 평균 ODE + RK4 vs 행렬지수 스위칭 해 (같은 제어기 코드)
- [INFO] (light) 평균모델(B) vs 스위칭(C): 2ω 리플 pp (B vs C) — 손으로 쓴 평균 ODE + RK4 vs 행렬지수 스위칭 해 (같은 제어기 코드)
- [회귀 PASS] (nominal) Hermite 보간 오차 상한 (i_g)
- [회귀 INFO] (nominal) 정상상태 판정 (계통 주기별 상태 변화)
- [회귀 PASS] (fast_v) Hermite 보간 오차 상한 (i_g)
- [회귀 INFO] (fast_v) 정상상태 판정 (계통 주기별 상태 변화)
- [회귀 PASS] (light) Hermite 보간 오차 상한 (i_g)
- [회귀 INFO] (light) 정상상태 판정 (계통 주기별 상태 변화)

**dclink_power** — 독립 9 · 회귀 0

- [PASS] (nominal) 평형 3상: 순시전력 합의 리플 — 세 상의 v·i 곱을 더한 파형의 2ω Fourier 성분
- [PASS] (nominal) 2ω 리플: 비선형 ODE vs 에너지 정확해 √(V²+P/ωC) − √(V²−P/ωC) — DOP853 적분 vs ½Cv² = W₀ − (P/2ω)sin2ωt
- [PASS] (nominal) hold-up: 정전력 방전 ODE의 V_lo 도달 시각 (C = C_min) vs Δt — C·v·dv/dt = −P를 event로 적분 vs 에너지 식
- [PASS] (unbalance) 불평형 3상 리플: 시간영역 파형 vs 대칭좌표 3V⁺I⁻ — 상별 v·i 곱의 합 vs Fortescue 역상분 전류
- [PASS] (unbalance) 2ω 리플: 비선형 ODE vs 에너지 정확해 √(V²+P/ωC) − √(V²−P/ωC) — DOP853 적분 vs ½Cv² = W₀ − (P/2ω)sin2ωt
- [PASS] (unbalance) hold-up: 정전력 방전 ODE의 V_lo 도달 시각 (C = C_min) vs Δt — C·v·dv/dt = −P를 event로 적분 vs 에너지 식
- [PASS] (small_c) 평형 3상: 순시전력 합의 리플 — 세 상의 v·i 곱을 더한 파형의 2ω Fourier 성분
- [PASS] (small_c) 2ω 리플: 비선형 ODE vs 에너지 정확해 √(V²+P/ωC) − √(V²−P/ωC) — DOP853 적분 vs ½Cv² = W₀ − (P/2ω)sin2ωt
- [PASS] (small_c) hold-up: 정전력 방전 ODE의 V_lo 도달 시각 (C = C_min) vs Δt — C·v·dv/dt = −P를 event로 적분 vs 에너지 식

## FL06 · 제어 — PI보다 plant와 부호가 먼저다


**pi_design** — 독립 5 · 회귀 0

- [PASS] (nominal) crossover: 복소 루프이득 근 찾기 vs 설계 목표 f_c — Kp = Lω_c, Ki = Rω_c로 만든 PI와 plant 1/(Ls+R)의 전체 복소 이득에서 \|L\| = 1 을 brentq로 탐색 (게인 공식을 다시 쓰지 않음)
- [PASS] (nominal) 위상여유: 수치 (지연 포함) vs 90° − 360 f_c T_d — unwrap한 위상에서 f_c의 정확한 위상 vs 손계산
- [PASS] (duty) crossover: 복소 루프이득 근 찾기 vs 설계 목표 f_c — Kp = Lω_c, Ki = Rω_c로 만든 PI와 plant 1/(Ls+R)의 전체 복소 이득에서 \|L\| = 1 을 brentq로 탐색 (게인 공식을 다시 쓰지 않음)
- [PASS] (duty) 위상여유: 수치 (지연 포함) vs 90° − 360 f_c T_d — unwrap한 위상에서 f_c의 정확한 위상 vs 손계산
- [PASS] (duty) duty 출력(환산) 루프 = 전압 출력 루프 — duty 출력 PI × K_PWM vs 전압 출력 PI의 샘플링 루프 crossover

**discrete_loop** — 독립 10 · 회귀 0

- [PASS] (load_step) ZOH 이산 map vs 연속 plant 정확 적분 (엔진) — i[k+1] = a·i[k] + b·(u−e) (a = e^(−RT_s/L)) vs 같은 인가전압을 샘플마다 바꾸는 RL 회로의 행렬지수 해
- [PASS] (load_step) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (input_dip) ZOH 이산 map vs 연속 plant 정확 적분 (엔진) — i[k+1] = a·i[k] + b·(u−e) (a = e^(−RT_s/L)) vs 같은 인가전압을 샘플마다 바꾸는 RL 회로의 행렬지수 해
- [PASS] (input_dip) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (current_limit) ZOH 이산 map vs 연속 plant 정확 적분 (엔진) — i[k+1] = a·i[k] + b·(u−e) (a = e^(−RT_s/L)) vs 같은 인가전압을 샘플마다 바꾸는 RL 회로의 행렬지수 해
- [PASS] (current_limit) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (sensor_offset) ZOH 이산 map vs 연속 plant 정확 적분 (엔진) — i[k+1] = a·i[k] + b·(u−e) (a = e^(−RT_s/L)) vs 같은 인가전압을 샘플마다 바꾸는 RL 회로의 행렬지수 해
- [PASS] (sensor_offset) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (extra_delay) ZOH 이산 map vs 연속 plant 정확 적분 (엔진) — i[k+1] = a·i[k] + b·(u−e) (a = e^(−RT_s/L)) vs 같은 인가전압을 샘플마다 바꾸는 RL 회로의 행렬지수 해
- [PASS] (extra_delay) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

**dq_sign** — 독립 6 · 회귀 0

- [PASS] (correct) dq 평균모델 vs 독립 abc 시뮬레이션 — dq 식(±ωL 교차결합)의 정확 이산 map vs 3상 L di/dt = v_g − v_c − R i를 DOP853으로 적분하고 측정 전류를 Park 변환
- [PASS] (correct) 순시전력: Σ v_x·i_x (abc) = 1.5 v_d i_d (dq) — abc 상전압·상전류 곱의 합 vs 진폭불변 dq 전력식 (v_q = 0)
- [PASS] (reversed) dq 평균모델 vs 독립 abc 시뮬레이션 — dq 식(±ωL 교차결합)의 정확 이산 map vs 3상 L di/dt = v_g − v_c − R i를 DOP853으로 적분하고 측정 전류를 Park 변환
- [PASS] (reversed) 순시전력: Σ v_x·i_x (abc) = 1.5 v_d i_d (dq) — abc 상전압·상전류 곱의 합 vs 진폭불변 dq 전력식 (v_q = 0)
- [PASS] (reactive) dq 평균모델 vs 독립 abc 시뮬레이션 — dq 식(±ωL 교차결합)의 정확 이산 map vs 3상 L di/dt = v_g − v_c − R i를 DOP853으로 적분하고 측정 전류를 Park 변환
- [PASS] (reactive) 순시전력: Σ v_x·i_x (abc) = 1.5 v_d i_d (dq) — abc 상전압·상전류 곱의 합 vs 진폭불변 dq 전력식 (v_q = 0)

## FL08 · DAB — 식에서 파형, 파형에서 설계 판단으로


**sps_nominal** — 독립 12 · 회귀 2

- [PASS] (nominal) 전력: 닫힌 식 φ → 구간 적분 P — P = V₁V₂φ(1−\|φ\|/π)/(ωL)의 φ를 넣고 PWL 정확 적분한 v₂′·i
- [PASS] (nominal) RMS: 닫힌 식 vs 구간 적분 — I_pk√(1−2φ/(3π)) / 일반식 vs Σ Δt(i_a²+i_ai_b+i_b²)/3
- [PASS] (nominal) 포트 전력 보존 (무손실): P₁ = P₂ — 1차 포트 ∫v₁·i vs 2차 포트 ∫v₂′·i₂ (구간 적분)
- [PASS] (nominal) 구간 적분 vs 행렬지수 스위칭 해: 전력 — PWL 정확 적분 vs expm 전파·Kronecker 모멘트
- [PASS] (nominal) 구간 적분 vs 스위칭 해: RMS — PWL vs expm
- [PASS] (nominal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (reverse) 전력: 닫힌 식 φ → 구간 적분 P — P = V₁V₂φ(1−\|φ\|/π)/(ωL)의 φ를 넣고 PWL 정확 적분한 v₂′·i
- [PASS] (reverse) RMS: 닫힌 식 vs 구간 적분 — I_pk√(1−2φ/(3π)) / 일반식 vs Σ Δt(i_a²+i_ai_b+i_b²)/3
- [PASS] (reverse) 포트 전력 보존 (무손실): P₁ = P₂ — 1차 포트 ∫v₁·i vs 2차 포트 ∫v₂′·i₂ (구간 적분)
- [PASS] (reverse) 구간 적분 vs 행렬지수 스위칭 해: 전력 — PWL 정확 적분 vs expm 전파·Kronecker 모멘트
- [PASS] (reverse) 구간 적분 vs 스위칭 해: RMS — PWL vs expm
- [PASS] (reverse) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [회귀 PASS] (nominal) 반주기 반대칭 기준해 x(T/2) = −x(0)
- [회귀 PASS] (reverse) 반주기 반대칭 기준해 x(T/2) = −x(0)

**zero_power_mismatch** — 독립 4 · 회귀 0

- [PASS] (mismatch) φ=0 순환전류 RMS: 삼각파 해석값 vs 구간 적분 — I_pk/√3 (삼각파) vs PWL
- [PASS] (mismatch) 스위칭 해 vs 구간 적분: 1차 RMS — expm vs PWL
- [PASS] (matched) φ=0 순환전류 RMS: 삼각파 해석값 vs 구간 적분 — I_pk/√3 (삼각파) vs PWL
- [PASS] (matched) 스위칭 해 vs 구간 적분: 1차 RMS — expm vs PWL

**voltage_corners** — 독립 2 · 회귀 0

- [PASS] (textbook) corner 전체: 닫힌 식 vs 구간 적분 (P, I_rms) — 9개 corner 각각 닫힌 식 φ로 PWL 정확 적분
- [PASS] (light) corner 전체: 닫힌 식 vs 구간 적분 (P, I_rms) — 9개 corner 각각 닫힌 식 φ로 PWL 정확 적분

**offset_startup** — 독립 4 · 회귀 0

- [PASS] (lossless) 무손실: offset이 유지된다 — 사이클 평균 비교 (첫 주기 vs 마지막 주기)
- [PASS] (lossless) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (damped) offset 감쇠: 스위칭 해 vs 해석 e^(−t·R/L) — 사이클 평균 (정확 적분) vs 1차 지수 감쇠 (R만 있는 이상 L)
- [PASS] (damped) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

**two_modules** — 독립 4 · 회귀 0

- [PASS] (textbook) 분담: 구간 적분 vs P ∝ 1/L 해석 — PWL 전력비 vs L 비 (SPS 전력식이 1/L에 비례)
- [PASS] (textbook) 180° 이동이 리플을 줄이지 못함 (포트 전류의 T/2 주기성) — 동일 모듈(δ = 0)에서 0°와 180°의 합계 리플 비교
- [PASS] (ideal) 분담: 구간 적분 vs P ∝ 1/L 해석 — PWL 전력비 vs L 비 (SPS 전력식이 1/L에 비례)
- [PASS] (ideal) 180° 이동이 리플을 줄이지 못함 (포트 전류의 T/2 주기성) — 동일 모듈(δ = 0)에서 0°와 180°의 합계 리플 비교

## FL09 · LLC — 공진을 말로 설명하고 식으로 확인하기


**fha_gain** — 독립 8 · 회귀 0

- [PASS] (textbook) 교재 정규화식 vs 페이저 절점해석 — 1/H = 1 + (1−F⁻²)/k + jQ(F−1/F) (reference) vs 절점 전압 방정식 (Z_r, Z_m, R_ac)
- [PASS] (textbook) FHA 회로의 시간영역 주기해 vs 페이저 — 사인파 전원(발진기 상태)·R_ac 부하 회로를 행렬지수로 풀어 v_load 진폭 비교
- [PASS] (textbook) 극한: F = 1에서 모든 Q의 \|H\| = 1 — 직렬 공진에서 Z_r = 0
- [PASS] (textbook) 극한: Q → 0 (무부하) \|H\| = 1/\|1 + (1−F⁻²)/k\| — L_r·C_r·L_m 분압만 남는다
- [PASS] (hb) 교재 정규화식 vs 페이저 절점해석 — 1/H = 1 + (1−F⁻²)/k + jQ(F−1/F) (reference) vs 절점 전압 방정식 (Z_r, Z_m, R_ac)
- [PASS] (hb) FHA 회로의 시간영역 주기해 vs 페이저 — 사인파 전원(발진기 상태)·R_ac 부하 회로를 행렬지수로 풀어 v_load 진폭 비교
- [PASS] (hb) 극한: F = 1에서 모든 Q의 \|H\| = 1 — 직렬 공진에서 Z_r = 0
- [PASS] (hb) 극한: Q → 0 (무부하) \|H\| = 1/\|1 + (1−F⁻²)/k\| — L_r·C_r·L_m 분압만 남는다

**switching_vs_fha** — 독립 15 · 회귀 5

- [PASS] (mid) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (mid) 독립 경로: 교재 상태식 + DOP853 한 주기 — affine 행렬·행렬지수·guard 대신 상태식을 직접 적고 solve_ivp 이벤트로 다이오드 전환
- [PASS] (mid) FHA 대 switching gain (f_r 부근 중부하 5 % 초기 점검값) — 교재 19장 기준: 0.9 ≤ F ≤ 1.1, 0.5 ≤ Q ≤ 1.2
- [PASS] (at_fr) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (at_fr) 독립 경로: 교재 상태식 + DOP853 한 주기 — affine 행렬·행렬지수·guard 대신 상태식을 직접 적고 solve_ivp 이벤트로 다이오드 전환
- [PASS] (at_fr) FHA 대 switching gain (f_r 부근 중부하 5 % 초기 점검값) — 교재 19장 기준: 0.9 ≤ F ≤ 1.1, 0.5 ≤ Q ≤ 1.2
- [PASS] (below_heavy) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (below_heavy) 독립 경로: 교재 상태식 + DOP853 한 주기 — affine 행렬·행렬지수·guard 대신 상태식을 직접 적고 solve_ivp 이벤트로 다이오드 전환
- [INFO] (below_heavy) FHA 대 switching gain (5 % 점검값 적용 범위 밖) — f_r에서 멀거나 경부하/과부하 — 차이는 모델 적용범위의 결과로 해석한다
- [PASS] (above_light) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (above_light) 독립 경로: 교재 상태식 + DOP853 한 주기 — affine 행렬·행렬지수·guard 대신 상태식을 직접 적고 solve_ivp 이벤트로 다이오드 전환
- [INFO] (above_light) FHA 대 switching gain (5 % 점검값 적용 범위 밖) — f_r에서 멀거나 경부하/과부하 — 차이는 모델 적용범위의 결과로 해석한다
- [PASS] (hb) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (hb) 독립 경로: 교재 상태식 + DOP853 한 주기 — affine 행렬·행렬지수·guard 대신 상태식을 직접 적고 solve_ivp 이벤트로 다이오드 전환
- [PASS] (hb) FHA 대 switching gain (f_r 부근 중부하 5 % 초기 점검값) — 교재 19장 기준: 0.9 ≤ F ≤ 1.1, 0.5 ≤ Q ≤ 1.2
- [회귀 PASS] (mid) 주기해 잔차 (shooting)
- [회귀 PASS] (at_fr) 주기해 잔차 (shooting)
- [회귀 PASS] (below_heavy) 주기해 잔차 (shooting)
- [회귀 PASS] (above_light) 주기해 잔차 (shooting)
- [회귀 PASS] (hb) 주기해 잔차 (shooting)

**gain_curve** — 독립 0 · 회귀 0


**zvs_lm** — 독립 0 · 회귀 0


## FL10 · CLLC — 설계 실패와 다중 해를 직접 방어하기


**seed_fail** — 독립 4 · 회귀 0

- [PASS] (seed) 교재 CLLC 식 vs 어드미턴스 절점해석 — H = Z_p/(Z_r1+Z_p)·R_ac′/(Z_r2′+R_ac′) (reference) vs I₁ = V₁/(Z₁ + 1/(Y_m+Y_b))
- [PASS] (seed) 최대 inductive gain: golden refine vs 90001점 격자 — 두 탐색 방법의 비교 (격자 1 Hz)
- [PASS] (fix) 교재 CLLC 식 vs 어드미턴스 절점해석 — H = Z_p/(Z_r1+Z_p)·R_ac′/(Z_r2′+R_ac′) (reference) vs I₁ = V₁/(Z₁ + 1/(Y_m+Y_b))
- [PASS] (fix) 최대 inductive gain: golden refine vs 90001점 격자 — 두 탐색 방법의 비교 (격자 1 Hz)

**fix_n093** — 독립 10 · 회귀 0

- [PASS] (textbook) 교재 CLLC 식 vs 어드미턴스 절점해석 — H = Z_p/(Z_r1+Z_p)·R_ac′/(Z_r2′+R_ac′) (reference) vs I₁ = V₁/(Z₁ + 1/(Y_m+Y_b))
- [PASS] (textbook) FHA 전력 보존 저전압 164.390 kHz — 입력 기본파 V·I·cosθ = R_ac 소비전력 (무손실 tank)
- [PASS] (textbook) FHA 전력 보존 중간 162.811 kHz — 입력 기본파 V·I·cosθ = R_ac 소비전력 (무손실 tank)
- [PASS] (textbook) FHA 전력 보존 고전압 136.099 kHz — 입력 기본파 V·I·cosθ = R_ac 소비전력 (무손실 tank)
- [PASS] (textbook) FHA 전력 보존 고전압 147.061 kHz — 입력 기본파 V·I·cosθ = R_ac 소비전력 (무손실 tank)
- [PASS] (kept_lc) 교재 CLLC 식 vs 어드미턴스 절점해석 — H = Z_p/(Z_r1+Z_p)·R_ac′/(Z_r2′+R_ac′) (reference) vs I₁ = V₁/(Z₁ + 1/(Y_m+Y_b))
- [PASS] (kept_lc) FHA 전력 보존 저전압 165.366 kHz — 입력 기본파 V·I·cosθ = R_ac 소비전력 (무손실 tank)
- [PASS] (kept_lc) FHA 전력 보존 중간 163.545 kHz — 입력 기본파 V·I·cosθ = R_ac 소비전력 (무손실 tank)
- [PASS] (kept_lc) FHA 전력 보존 고전압 133.260 kHz — 입력 기본파 V·I·cosθ = R_ac 소비전력 (무손실 tank)
- [PASS] (kept_lc) FHA 전력 보존 고전압 147.164 kHz — 입력 기본파 V·I·cosθ = R_ac 소비전력 (무손실 tank)

**branch_selection** — 독립 0 · 회귀 0


**reverse** — 독립 1 · 회귀 0

- [PASS] (textbook) reverse: 교재 식(탱크 교환) vs 절점해석 — 같은 식에 뒤집힌 회로 대입 vs 어드미턴스 해석

**time_domain** — 독립 4 · 회귀 0

- [PASS] (textbook) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (textbook) 독립 경로: 교재 E05 상태식 + DOP853 한 주기 — 상태식을 직접 적어 solve_ivp 이벤트로 적분 (affine 행렬·행렬지수 미사용)
- [PASS] (textbook) 독립 경로: 고조파 중첩 (정류 연속 도통) 136.099 kHz — 홀수 고조파 3999차까지 선형 회로 해 + 정류 전압 edge = i₂ 영점 조건 (주파수 영역) vs 시간영역 주기해
- [PASS] (textbook) 독립 경로: 고조파 중첩 (정류 연속 도통) 147.061 kHz — 홀수 고조파 3999차까지 선형 회로 해 + 정류 전압 edge = i₂ 영점 조건 (주파수 영역) vs 시간영역 주기해

## EX02 · 비선형 Coss·dead time·ZVS — 에너지식 하나로 판정하지 않기


**qe_integrals** — 독립 2 · 회귀 0

- [PASS] (textbook) Q_oss: 닫힌 식 vs 수치 적분(quad) — 해석 적분식 vs 적응 Gauss–Kronrod 구적
- [PASS] (textbook) E_oss: 닫힌 식 vs 수치 적분(quad) — 해석 적분식 vs 적응 Gauss–Kronrod 구적

**hb_constant_current** — 독립 2 · 회귀 0

- [PASS] (textbook) dead time 끝 노드 전압: 전하 역함수 vs ODE 적분 — Q_node(v) = I·t_d 역함수(해석) vs dv/dt = I/C_node(v) 수치적분
- [PASS] (td300) dead time 끝 노드 전압: 전하 역함수 vs ODE 적분 — Q_node(v) = I·t_d 역함수(해석) vs dv/dt = I/C_node(v) 수치적분

**resonant_transition** — 독립 8 · 회귀 1

- [PASS] (mid) 에너지 잔차 (source − ΔW_L − ΔW_C − rail − diode) — v_x·∫i, V_bus·(rail 전하), ½Li², E_node(v) — 상태식과 독립적으로 정의한 에너지 항
- [PASS] (mid) 독립 solver: 전하좌표 RK4 (h 반감 수렴) — (q, i) 상태·고정 step RK4·v = Q⁻¹(q) vs (v, i) 상태·DOP853 적응 step
- [PASS] (mid) 한계 경우: 고정 C에서 ODE vs 해석적 LC 해 — DOP853 ODE vs v(t) = v_x − v_x cos ωt + I₀/(Cω) sin ωt
- [PASS] (mid300) 에너지 잔차 (source − ΔW_L − ΔW_C − rail − diode) — v_x·∫i, V_bus·(rail 전하), ½Li², E_node(v) — 상태식과 독립적으로 정의한 에너지 항
- [PASS] (mid300) 독립 solver: 전하좌표 RK4 (h 반감 수렴) — (q, i) 상태·고정 step RK4·v = Q⁻¹(q) vs (v, i) 상태·DOP853 적응 step
- [PASS] (mid300) 한계 경우: 고정 C에서 ODE vs 해석적 LC 해 — DOP853 ODE vs v(t) = v_x − v_x cos ωt + I₀/(Cω) sin ωt
- [PASS] (vx0) 에너지 잔차 (source − ΔW_L − ΔW_C − rail − diode) — v_x·∫i, V_bus·(rail 전하), ½Li², E_node(v) — 상태식과 독립적으로 정의한 에너지 항
- [PASS] (vx0) 한계 경우: 고정 C에서 ODE vs 해석적 LC 해 — DOP853 ODE vs v(t) = v_x − v_x cos ωt + I₀/(Cω) sin ωt
- [회귀 NOT_RUN] (vx0) 독립 solver: 전하좌표 RK4

**deadtime_window** — 독립 0 · 회귀 0


## EX05 · CLLC — gain 곡선에서 실제 동역학으로


**state_identity** — 독립 6 · 회귀 0

- [PASS] (textbook) 에너지 항등식: 모델 도함수 vs 교재 포트 전력 — ∇W·(A x + b) (모델 행렬) vs v₁i₁ − v₂i₂ − R₁i₁² − R₂i₂² (교재 식)
- [PASS] (textbook) 항등식이 부호 실수를 검출 — v_C2 부호를 뒤집은 도함수는 항등식을 깨야 한다
- [PASS] (textbook) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (lossy) 에너지 항등식: 모델 도함수 vs 교재 포트 전력 — ∇W·(A x + b) (모델 행렬) vs v₁i₁ − v₂i₂ − R₁i₁² − R₂i₂² (교재 식)
- [PASS] (lossy) 항등식이 부호 실수를 검출 — v_C2 부호를 뒤집은 도함수는 항등식을 깨야 한다
- [PASS] (lossy) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

**operating_points** — 독립 3 · 회귀 0

- [PASS] (textbook) seed (n = 1) 실패 보존 — FL10 seed: inductive 최대 gain < 필요 gain
- [PASS] (textbook) 독립 경로: 교재 상태식 + 출력 C/R, DOP853 한 주기 — 상태식을 직접 적어 solve_ivp 이벤트로 적분
- [PASS] (textbook) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

**floquet** — 독립 2 · 회귀 2

- [PASS] (textbook) 섭동 감쇠율: 시뮬레이션 vs Floquet \|λ\|max — 비선형 사이클 시뮬레이션의 포락선 vs FD monodromy 고유값
- [PASS] (lossy) 섭동 감쇠율: 시뮬레이션 vs Floquet \|λ\|max — 비선형 사이클 시뮬레이션의 포락선 vs FD monodromy 고유값
- [회귀 PASS] (textbook) 주기해의 DC offset (i₁, i₂′, i_m 평균)
- [회귀 PASS] (lossy) 주기해의 DC offset (i₁, i₂′, i_m 평균)

**gvf** — 독립 5 · 회귀 0

- [PASS] (textbook) G_vf(0) vs 정상상태 기울기 (FHA 해 136.099 kHz) — 선형 map의 DC 이득 C(I−A)⁻¹B vs ±20 Hz 별도 주기해의 v_o 차분
- [PASS] (textbook) G_vf(0) vs 정상상태 기울기 (FHA 해 147.061 kHz) — 선형 map의 DC 이득 C(I−A)⁻¹B vs ±20 Hz 별도 주기해의 v_o 차분
- [PASS] (textbook) G_vf(0) vs 정상상태 기울기 (스위칭 운전점 148.005 kHz) — 선형 map의 DC 이득 C(I−A)⁻¹B vs ±20 Hz 별도 주기해의 v_o 차분
- [PASS] (textbook) FM 주입 300 Hz: 비선형 시뮬레이션 vs 선형 예측 — f_s에 ±0.002 % 사인 변조, 400주기 — 같은 입력열로 비선형 사이클 map과 선형 map 비교
- [PASS] (textbook) FM 주입 1000 Hz: 비선형 시뮬레이션 vs 선형 예측 — f_s에 ±0.002 % 사인 변조, 400주기 — 같은 입력열로 비선형 사이클 map과 선형 map 비교

**closed_loop** — 독립 3 · 회귀 0

- [PASS] (slow) 선형 폐루프(제어기 상태 포함) vs 비선형 폐루프: 0.5 V 기준 계단 — plant FD Jacobian + 적분기 + 1주기 지연의 선형 map vs 비선형 사이클 시뮬레이션 (300주기, 소신호)
- [PASS] (fast) 선형 폐루프(제어기 상태 포함) vs 비선형 폐루프: 0.5 V 기준 계단 — plant FD Jacobian + 적분기 + 1주기 지연의 선형 map vs 비선형 사이클 시뮬레이션 (300주기, 소신호)
- [PASS] (wrong_sign) 선형 폐루프(제어기 상태 포함) vs 비선형 폐루프: 0.5 V 기준 계단 — plant FD Jacobian + 적분기 + 1주기 지연의 선형 map vs 비선형 사이클 시뮬레이션 (300주기, 소신호)

**startup_reverse** — 독립 3 · 회귀 0

- [PASS] (textbook) 기동 전체 에너지 잔차 (E_in − E_load − E_R − ΔW) — 주기마다 포트·부하 에너지를 정확 적분해 합산, 시작/끝 저장에너지 차
- [PASS] (textbook) 독립 경로: 교재 상태식 DOP853로 기동 첫 30주기 연쇄 적분 — 주기마다 다른 T_k로 solve_ivp(이벤트) 연쇄 vs 엔진 사이클 map
- [PASS] (textbook) reverse: FHA 판정과 스위칭 판정의 일치 — FHA 최대 gain < 필요 gain ⇔ 스위칭 최대 전력 < 목표

**tolerance** — 독립 0 · 회귀 0


## EX06 · DAB: RMS 최소화와 ZVS의 상충관계


**general_modulation** — 독립 10 · 회귀 2

- [PASS] (textbook) SPS: 닫힌 식 vs 구간 적분 (I_rms) — I_rms 일반식 vs b(θ;π,φ) 정확 PWL 적분
- [PASS] (textbook) SPS: 닫힌 식 vs 구간 적분 (I_pk) — max(\|i₀\|,\|i_φ\|) vs PWL 꼭짓점
- [PASS] (textbook) candidate: 구간 적분 vs 스위칭 해 (P) — PWL vs expm·3-level bridge mode
- [PASS] (textbook) candidate: 구간 적분 vs 스위칭 해 (I_rms) — PWL vs expm
- [PASS] (textbook) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (sps) SPS: 닫힌 식 vs 구간 적분 (I_rms) — I_rms 일반식 vs b(θ;π,φ) 정확 PWL 적분
- [PASS] (sps) SPS: 닫힌 식 vs 구간 적분 (I_pk) — max(\|i₀\|,\|i_φ\|) vs PWL 꼭짓점
- [PASS] (sps) candidate: 구간 적분 vs 스위칭 해 (P) — PWL vs expm·3-level bridge mode
- [PASS] (sps) candidate: 구간 적분 vs 스위칭 해 (I_rms) — PWL vs expm
- [PASS] (sps) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [회귀 PASS] (textbook) candidate: 목표 전력 재현
- [회귀 PASS] (sps) candidate: 목표 전력 재현

**candidate_map** — 독립 0 · 회귀 0


**commutation_detail** — 독립 0 · 회귀 0


**implementation** — 독립 2 · 회귀 0

- [PASS] (textbook) 국소 감도: 해석 미분 vs 유한차분 (작은 Δφ) — ∂P/∂φ = V₁V₂(1−2φ/π)/(ωL) vs 중앙차분
- [PASS] (textbook) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

## EX07 · 제어·입력 상호작용 — 단일 루프가 안정해도 시스템은 불안정할 수 있다


**cpl_exact** — 독립 8 · 회귀 0

- [PASS] (c100u) 평형점: fsolve(비선형 우변 = 0) vs 이차식 해 — Newton 계열 수치해 (고전압 쪽 초기값) vs 해석식
- [PASS] (c100u) 극점: 비선형 모델의 수치 Jacobian vs 해석 A — 중앙차분 Jacobian의 고유값 vs [[−R/L, −1/L], [1/C, g/C]]의 고유값
- [PASS] (c100u) 성장·감쇠율: 비선형 ODE 봉우리 vs 선형 극점 — DOP853 비선형 적분(1 V 교란)의 봉우리 로그 기울기 vs 고유값 실수부
- [PASS] (c100u) 임계 C: max Re(λ(C)) = 0 수치 근 vs L·g/R — 고유값 실수부의 부호가 바뀌는 C를 brentq로 탐색 vs trace 조건 해석식
- [PASS] (c1m) 평형점: fsolve(비선형 우변 = 0) vs 이차식 해 — Newton 계열 수치해 (고전압 쪽 초기값) vs 해석식
- [PASS] (c1m) 극점: 비선형 모델의 수치 Jacobian vs 해석 A — 중앙차분 Jacobian의 고유값 vs [[−R/L, −1/L], [1/C, g/C]]의 고유값
- [PASS] (c1m) 성장·감쇠율: 비선형 ODE 봉우리 vs 선형 극점 — DOP853 비선형 적분(1 V 교란)의 봉우리 로그 기울기 vs 고유값 실수부
- [PASS] (c1m) 임계 C: max Re(λ(C)) = 0 수치 근 vs L·g/R — 고유값 실수부의 부호가 바뀌는 C를 brentq로 탐색 vs trace 조건 해석식

**converter_impedance** — 독립 15 · 회귀 4

- [PASS] (nominal) 컨버터 소신호 A_c: 손 유도 vs 수치 Jacobian — 편미분을 손으로 쓴 행렬 vs 비선형 우변의 중앙차분
- [PASS] (nominal) 컨버터 입력 B_c (∂/∂v): 손 유도 vs 수치 — 같은 두 경로
- [PASS] (nominal) Nyquist 영점 수 = 우반평면 고유값 수 — 소신호 임피던스(Z_s 해석식, Y_in 상태공간)의 위상 회전 vs 전체 시스템 Jacobian 고유값
- [PASS] (nominal) 시간영역: 비선형 평균모델 포락선 vs 지배 고유값 — DOP853 비선형 적분 vs 선형화 고유값 (다른 극점의 과도 성분 때문에 허용오차 10 %)
- [PASS] (slow) 컨버터 소신호 A_c: 손 유도 vs 수치 Jacobian — 편미분을 손으로 쓴 행렬 vs 비선형 우변의 중앙차분
- [PASS] (slow) 컨버터 입력 B_c (∂/∂v): 손 유도 vs 수치 — 같은 두 경로
- [PASS] (slow) Nyquist 영점 수 = 우반평면 고유값 수 — 소신호 임피던스(Z_s 해석식, Y_in 상태공간)의 위상 회전 vs 전체 시스템 Jacobian 고유값
- [PASS] (feedforward) 컨버터 소신호 A_c: 손 유도 vs 수치 Jacobian — 편미분을 손으로 쓴 행렬 vs 비선형 우변의 중앙차분
- [PASS] (feedforward) 컨버터 입력 B_c (∂/∂v): 손 유도 vs 수치 — 같은 두 경로
- [PASS] (feedforward) Nyquist 영점 수 = 우반평면 고유값 수 — 소신호 임피던스(Z_s 해석식, Y_in 상태공간)의 위상 회전 vs 전체 시스템 Jacobian 고유값
- [PASS] (feedforward) 시간영역: 비선형 평균모델 포락선 vs 지배 고유값 — DOP853 비선형 적분 vs 선형화 고유값 (다른 극점의 과도 성분 때문에 허용오차 10 %)
- [PASS] (big_c) 컨버터 소신호 A_c: 손 유도 vs 수치 Jacobian — 편미분을 손으로 쓴 행렬 vs 비선형 우변의 중앙차분
- [PASS] (big_c) 컨버터 입력 B_c (∂/∂v): 손 유도 vs 수치 — 같은 두 경로
- [PASS] (big_c) Nyquist 영점 수 = 우반평면 고유값 수 — 소신호 임피던스(Z_s 해석식, Y_in 상태공간)의 위상 회전 vs 전체 시스템 Jacobian 고유값
- [PASS] (big_c) 시간영역: 비선형 평균모델 포락선 vs 지배 고유값 — DOP853 비선형 적분 vs 선형화 고유값 (다른 극점의 과도 성분 때문에 허용오차 10 %)
- [회귀 PASS] (nominal) Y_in: 모드 분해(부분분수) vs 주파수별 직접 선형해
- [회귀 PASS] (slow) Y_in: 모드 분해(부분분수) vs 주파수별 직접 선형해
- [회귀 PASS] (feedforward) Y_in: 모드 분해(부분분수) vs 주파수별 직접 선형해
- [회귀 PASS] (big_c) Y_in: 모드 분해(부분분수) vs 주파수별 직접 선형해

**digital_delay** — 독립 8 · 회귀 0

- [PASS] (td15) 지연 위상 @ 1 kHz: 샘플·지연·hold 시뮬레이션 vs 360·f·T_d — 사인 명령을 T_c마다 샘플해 τ 뒤에 T_c 동안 유지한 계단 파형의 기본파 위상 (정확 구간 적분)
- [PASS] (td15) 지연 위상 @ 10 kHz: 샘플·지연·hold 시뮬레이션 vs 360·f·T_d — 사인 명령을 T_c마다 샘플해 τ 뒤에 T_c 동안 유지한 계단 파형의 기본파 위상 (정확 구간 적분)
- [PASS] (td15) 분수 지연 이산 map vs 실제 갱신 시각의 연속 plant (엔진) — b_old/b_new로 나눈 정확 이산화 vs 갱신 시각마다 전압을 바꾸는 RL 회로의 행렬지수 해
- [PASS] (td15) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (slip) 지연 위상 @ 1 kHz: 샘플·지연·hold 시뮬레이션 vs 360·f·T_d — 사인 명령을 T_c마다 샘플해 τ 뒤에 T_c 동안 유지한 계단 파형의 기본파 위상 (정확 구간 적분)
- [PASS] (slip) 지연 위상 @ 10 kHz: 샘플·지연·hold 시뮬레이션 vs 360·f·T_d — 사인 명령을 T_c마다 샘플해 τ 뒤에 T_c 동안 유지한 계단 파형의 기본파 위상 (정확 구간 적분)
- [PASS] (slip) 분수 지연 이산 map vs 실제 갱신 시각의 연속 plant (엔진) — b_old/b_new로 나눈 정확 이산화 vs 갱신 시각마다 전압을 바꾸는 RL 회로의 행렬지수 해
- [PASS] (slip) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

**saturation_transfer** — 독립 2 · 회귀 2

- [PASS] (axis) dq 포화 궤적: dq 모델 vs 같은 인가 전압의 abc 적분 — 포화된 dq 전압을 3상으로 되돌려 DOP853 적분 후 Park 변환 vs dq 정확 map
- [PASS] (circle) dq 포화 궤적: dq 모델 vs 같은 인가 전압의 abc 적분 — 포화된 dq 전압을 3상으로 되돌려 DOP853 적분 후 Park 변환 vs dq 정확 map
- [회귀 PASS] (axis) bumpless: 복귀 순간 명령 연속
- [회귀 PASS] (circle) bumpless: 복귀 순간 명령 연속
