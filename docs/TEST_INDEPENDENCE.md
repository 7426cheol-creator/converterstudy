# 테스트 독립성 지도 (Test independence map)

생성: `tools/gen_docs.py` · run-all 2026-09-30T23:59:39

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

## FL02 · Si·SiC·GaN과 데이터시트, Gate drive·DPT·보호


**loss_ab** — 독립 9 · 회귀 0

- [PASS] (textbook) A 손실: 식 vs PWL 파형 적분 + edge 계수 — 사각 스위치 전류(같은 RMS)의 ∫i²R dt 정확 적분과 on/off 쌍 계수 vs I²R + E·f_s
- [PASS] (textbook) B 손실: 식 vs PWL 파형 적분 + edge 계수 — 같은 독립 경로
- [PASS] (textbook) 교차 주파수: 닫힌 식 vs 파형 경로의 근 찾기 — PWL 파형 손실 차이의 brentq 근 vs I²ΔR/ΔE
- [PASS] (f20k) A 손실: 식 vs PWL 파형 적분 + edge 계수 — 사각 스위치 전류(같은 RMS)의 ∫i²R dt 정확 적분과 on/off 쌍 계수 vs I²R + E·f_s
- [PASS] (f20k) B 손실: 식 vs PWL 파형 적분 + edge 계수 — 같은 독립 경로
- [PASS] (f20k) 교차 주파수: 닫힌 식 vs 파형 경로의 근 찾기 — PWL 파형 손실 차이의 brentq 근 vs I²ΔR/ΔE
- [PASS] (coss_unknown_20k) A 손실: 식 vs PWL 파형 적분 + edge 계수 — 사각 스위치 전류(같은 RMS)의 ∫i²R dt 정확 적분과 on/off 쌍 계수 vs I²R + E·f_s
- [PASS] (coss_unknown_20k) B 손실: 식 vs PWL 파형 적분 + edge 계수 — 같은 독립 경로
- [PASS] (coss_unknown_20k) 교차 주파수: 닫힌 식 vs 파형 경로의 근 찾기 — PWL 파형 손실 차이의 brentq 근 vs I²ΔR/ΔE

**gate_protect** — 독립 4 · 회귀 2

- [PASS] (textbook) plateau 시간: event ODE vs Q_gd/I_g — piecewise Q–v 모델 ODE(DOP853)의 event 시각 차 vs 닫힌 식
- [PASS] (textbook) plateau 도달: event ODE vs R·C₁·ln(ΔV₀/ΔV₁) — ODE event vs RC 충전 닫힌 식
- [PASS] (blank_long) plateau 시간: event ODE vs Q_gd/I_g — piecewise Q–v 모델 ODE(DOP853)의 event 시각 차 vs 닫힌 식
- [PASS] (blank_long) plateau 도달: event ODE vs R·C₁·ln(ΔV₀/ΔV₁) — ODE event vs RC 충전 닫힌 식
- [회귀 PASS] (textbook) 보호 체인 산술 (회귀)
- [회귀 PASS] (blank_long) 보호 체인 산술 (회귀)

**dpt_cell** — 독립 11 · 회귀 0

- [PASS] (nominal) gate 전하 보존: ∫i_G dt vs Q_gate(v) 닫힌 식 — solver가 적분한 gate 전류 vs C_gs·v_gs − Q_gd(v_dg)의 해석 적분식 (Miller 구간)
- [PASS] (nominal) 에너지 잔차 (source + drivers − load − 소산 − ΔW) — V_bus·∫i, 드라이버 ∫v·i_G, 부하 ∫v_PM·I_L, 저항·채널·diode 소산, ½Li² + 비선형 C 에너지식 — 상태식과 독립적으로 정의한 항
- [PASS] (nominal) 표본 적분 vs solver 적분 (turn-on 단자 에너지) — dense output 표본의 사다리꼴 적분 vs 적분상태 ∫v_DS·i_D
- [PASS] (nominal) 허용오차 강화 수렴 (rtol 1e-6 → 1e-8) — 같은 셀을 더 엄격한 허용오차로 다시 적분: E_on·E_off·v_DS peak 변화
- [PASS] (fast) 에너지 잔차 (source + drivers − load − 소산 − ΔW) — V_bus·∫i, 드라이버 ∫v·i_G, 부하 ∫v_PM·I_L, 저항·채널·diode 소산, ½Li² + 비선형 C 에너지식 — 상태식과 독립적으로 정의한 항
- [PASS] (fast) 표본 적분 vs solver 적분 (turn-on 단자 에너지) — dense output 표본의 사다리꼴 적분 vs 적분상태 ∫v_DS·i_D
- [PASS] (fast) 허용오차 강화 수렴 (rtol 1e-6 → 1e-8) — 같은 셀을 더 엄격한 허용오차로 다시 적분: E_on·E_off·v_DS peak 변화
- [PASS] (no_kelvin) gate 전하 보존: ∫i_G dt vs Q_gate(v) 닫힌 식 — solver가 적분한 gate 전류 vs C_gs·v_gs − Q_gd(v_dg)의 해석 적분식 (Miller 구간)
- [PASS] (no_kelvin) 에너지 잔차 (source + drivers − load − 소산 − ΔW) — V_bus·∫i, 드라이버 ∫v·i_G, 부하 ∫v_PM·I_L, 저항·채널·diode 소산, ½Li² + 비선형 C 에너지식 — 상태식과 독립적으로 정의한 항
- [PASS] (no_kelvin) 표본 적분 vs solver 적분 (turn-on 단자 에너지) — dense output 표본의 사다리꼴 적분 vs 적분상태 ∫v_DS·i_D
- [PASS] (no_kelvin) 허용오차 강화 수렴 (rtol 1e-6 → 1e-8) — 같은 셀을 더 엄격한 허용오차로 다시 적분: E_on·E_off·v_DS peak 변화

**vgs_spike** — 독립 6 · 회귀 0

- [PASS] (artifact) 겉보기 전압 = L_sH·max(di_HS/dt) (측정 정의 일관성) — 출력식의 L_sH·(선형계에서 푼 di/dt) vs 표본 전류의 수치 미분
- [PASS] (artifact) 에너지 잔차 (turn-on 사건) — 포트·소산·저장에너지 원장 (상태식과 독립 정의)
- [PASS] (real) 겉보기 전압 = L_sH·max(di_HS/dt) (측정 정의 일관성) — 출력식의 L_sH·(선형계에서 푼 di/dt) vs 표본 전류의 수치 미분
- [PASS] (real) 에너지 잔차 (turn-on 사건) — 포트·소산·저장에너지 원장 (상태식과 독립 정의)
- [PASS] (clamp) 겉보기 전압 = L_sH·max(di_HS/dt) (측정 정의 일관성) — 출력식의 L_sH·(선형계에서 푼 di/dt) vs 표본 전류의 수치 미분
- [PASS] (clamp) 에너지 잔차 (turn-on 사건) — 포트·소산·저장에너지 원장 (상태식과 독립 정의)

## FL03 · 손실·온도·수명 — 숫자가 서로 맞아야 한다


**rc_step** — 독립 3 · 회귀 0

- [PASS] (textbook) 독립 경로: 손으로 쓴 RK4 (h 반감) — RK4 스칼라 적분(h = t/20, t/40, t/80) vs 행렬지수 정확 해
- [PASS] (textbook) 해석해 vs 정확 적분 — 닫힌 식 T_b + PR(1 − e^(−t/τ)) vs 엔진 expm
- [PASS] (textbook) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

**electrothermal** — 독립 10 · 회귀 0

- [PASS] (nominal) 고정점: 반복 수렴값 vs 닫힌 식 — 단순 대입 반복(40회) vs (T_b − T_ref + R·P₀)/(1 − g) + T_ref
- [PASS] (nominal) 고정점: 정확 과도의 장시간 값 vs 닫힌 식 — 행렬지수 과도를 60τ_eff 적분한 끝값 vs 닫힌 식
- [PASS] (nominal) 독립 경로: RK4(h = 0.02 s) vs 정확 과도 끝값 — 구간별 RK4 스칼라 적분 vs 구간별 행렬지수
- [PASS] (nominal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (sensitive) 고정점: 반복 수렴값 vs 닫힌 식 — 단순 대입 반복(40회) vs (T_b − T_ref + R·P₀)/(1 − g) + T_ref
- [PASS] (sensitive) 고정점: 정확 과도의 장시간 값 vs 닫힌 식 — 행렬지수 과도를 60τ_eff 적분한 끝값 vs 닫힌 식
- [PASS] (sensitive) 독립 경로: RK4(h = 0.02 s) vs 정확 과도 끝값 — 구간별 RK4 스칼라 적분 vs 구간별 행렬지수
- [PASS] (sensitive) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (runaway) 독립 경로: RK4(h = 0.02 s) vs 정확 과도 끝값 — 구간별 RK4 스칼라 적분 vs 구간별 행렬지수
- [PASS] (runaway) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

**loss_map** — 독립 6 · 회귀 2

- [PASS] (nominal) 한계 경우: 사건 합 전도(forward) vs 정현 PWM 닫힌 식 — N개 스위칭 주기 합(선형 E·상수 R·dead time 0) vs I²R(1/8 + m·cosφ/3π)
- [PASS] (nominal) 한계 경우: 사건 합 전도(reverse) vs 닫힌 식 — 같은 합 vs I²R(1/8 − m·cosφ/3π)
- [PASS] (nominal) 한계 경우: 사건 합 스위칭 vs f_s·k·I_pk/π — E = k\|i\| 사건 합 vs 닫힌 식
- [PASS] (over_I) 한계 경우: 사건 합 전도(forward) vs 정현 PWM 닫힌 식 — N개 스위칭 주기 합(선형 E·상수 R·dead time 0) vs I²R(1/8 + m·cosφ/3π)
- [PASS] (over_I) 한계 경우: 사건 합 전도(reverse) vs 닫힌 식 — 같은 합 vs I²R(1/8 − m·cosφ/3π)
- [PASS] (over_I) 한계 경우: 사건 합 스위칭 vs f_s·k·I_pk/π — E = k\|i\| 사건 합 vs 닫힌 식
- [회귀 PASS] (nominal) map 보간이 격자점을 그대로 재현 (회귀)
- [회귀 PASS] (over_I) map 보간이 격자점을 그대로 재현 (회귀)

**foster_cauer** — 독립 8 · 회귀 0

- [PASS] (nominal) Foster(고유값 분해) vs 사다리 과도 (Z_jc(t)) — 일반화 고유값 G·v = λ·C·v로 만든 Foster 합 vs 행렬지수로 푼 사다리 과도
- [PASS] (nominal) Cauer → Foster → Cauer 왕복 (연분수 전개) — 고유값 분해의 Foster를 연분수로 다시 Cauer로: 원래 R·C 복원
- [PASS] (nominal) 변환된 Cauer 연결 ≈ 물리 사다리 연결 — Foster 피팅 → 연분수 Cauer → 방열판 연결 vs 원래 물리 사다리 + 방열판
- [PASS] (nominal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (small_sink) Foster(고유값 분해) vs 사다리 과도 (Z_jc(t)) — 일반화 고유값 G·v = λ·C·v로 만든 Foster 합 vs 행렬지수로 푼 사다리 과도
- [PASS] (small_sink) Cauer → Foster → Cauer 왕복 (연분수 전개) — 고유값 분해의 Foster를 연분수로 다시 Cauer로: 원래 R·C 복원
- [PASS] (small_sink) 변환된 Cauer 연결 ≈ 물리 사다리 연결 — Foster 피팅 → 연분수 Cauer → 방열판 연결 vs 원래 물리 사다리 + 방열판
- [PASS] (small_sink) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

## FL04 · 인버터 — 알고 있는 것을 더 날카롭게 증명하기


**dq_point** — 독립 6 · 회귀 3

- [PASS] (nominal) 상전압 peak: abc 시간영역 vs dq 폐형식 — 명시적 돌극 L(θ) 행렬·ψ_abc의 복소 step 미분(Faraday) → v_a(θ) 최대값 vs √(v_d² + v_q²)
- [PASS] (nominal) 단자 전력: abc Σv·i 평균 vs shaft + 동손 — abc 순시전력 평균 (토크식을 쓰지 않음) vs P_shaft + 1.5R_s\|i\|² (토크식 → i_q)
- [PASS] (sag760) 상전압 peak: abc 시간영역 vs dq 폐형식 — 명시적 돌극 L(θ) 행렬·ψ_abc의 복소 step 미분(Faraday) → v_a(θ) 최대값 vs √(v_d² + v_q²)
- [PASS] (sag760) 단자 전력: abc Σv·i 평균 vs shaft + 동손 — abc 순시전력 평균 (토크식을 쓰지 않음) vs P_shaft + 1.5R_s\|i\|² (토크식 → i_q)
- [PASS] (mtpa) 상전압 peak: abc 시간영역 vs dq 폐형식 — 명시적 돌극 L(θ) 행렬·ψ_abc의 복소 step 미분(Faraday) → v_a(θ) 최대값 vs √(v_d² + v_q²)
- [PASS] (mtpa) 단자 전력: abc Σv·i 평균 vs shaft + 동손 — abc 순시전력 평균 (토크식을 쓰지 않음) vs P_shaft + 1.5R_s\|i\|² (토크식 → i_q)
- [회귀 PASS] (nominal) dq 단자전력 1.5(v_d i_d + v_q i_q) vs shaft + 동손
- [회귀 PASS] (sag760) dq 단자전력 1.5(v_d i_d + v_q i_q) vs shaft + 동손
- [회귀 PASS] (mtpa) dq 단자전력 1.5(v_d i_d + v_q i_q) vs shaft + 동손

**dq_plane** — 독립 6 · 회귀 0

- [PASS] (nominal) MTPA: 폐형식 vs 같은 토크 곡선 위 최소 \|i\| — ∂T/∂γ = 0 해석식(\|i\| 고정) + 토크 역산 vs 토크 곡선을 따라 \|i\|를 bounded 최소화 (다른 문제 설정)
- [PASS] (nominal) 토크 곡선 ∩ 전압 한계: 두 경로 — 토크 곡선을 따라 \|v\| = V 풀기 vs 전압 한계 곡선(상측 근)을 따라 T = T* 풀기
- [PASS] (nominal) 최대 토크 @ 800 V: 격자+정제 vs SLSQP — i_d 격자에서 가능한 최대 i_q → bounded 정제 vs 부등식 제약 SLSQP (다른 알고리즘)
- [PASS] (fast) MTPA: 폐형식 vs 같은 토크 곡선 위 최소 \|i\| — ∂T/∂γ = 0 해석식(\|i\| 고정) + 토크 역산 vs 토크 곡선을 따라 \|i\|를 bounded 최소화 (다른 문제 설정)
- [PASS] (fast) 토크 곡선 ∩ 전압 한계: 두 경로 — 토크 곡선을 따라 \|v\| = V 풀기 vs 전압 한계 곡선(상측 근)을 따라 T = T* 풀기
- [PASS] (fast) 최대 토크 @ 800 V: 격자+정제 vs SLSQP — i_d 격자에서 가능한 최대 i_q → bounded 정제 vs 부등식 제약 SLSQP (다른 알고리즘)

**inverter_switching** — 독립 8 · 회귀 12

- [PASS] (nominal) 독립 solver: 손으로 쓴 dq ODE (RK45, cos/sin(ω_e t) 직접) — 2상태 비자율 ODE + 스위칭 상태별 v_αβ를 RK45로 rtol 1e-7 → 1e-9 적분한 한 주기 끝 전류 vs 정확 해(4상태 자율 LTI, 행렬지수)
- [PASS] (nominal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (N27) 독립 solver: 손으로 쓴 dq ODE (RK45, cos/sin(ω_e t) 직접) — 2상태 비자율 ODE + 스위칭 상태별 v_αβ를 RK45로 rtol 1e-7 → 1e-9 적분한 한 주기 끝 전류 vs 정확 해(4상태 자율 LTI, 행렬지수)
- [PASS] (N27) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (spwm) 독립 solver: 손으로 쓴 dq ODE (RK45, cos/sin(ω_e t) 직접) — 2상태 비자율 ODE + 스위칭 상태별 v_αβ를 RK45로 rtol 1e-7 → 1e-9 적분한 한 주기 끝 전류 vs 정확 해(4상태 자율 LTI, 행렬지수)
- [PASS] (spwm) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (sag760) 독립 solver: 손으로 쓴 dq ODE (RK45, cos/sin(ω_e t) 직접) — 2상태 비자율 ODE + 스위칭 상태별 v_αβ를 RK45로 rtol 1e-7 → 1e-9 적분한 한 주기 끝 전류 vs 정확 해(4상태 자율 LTI, 행렬지수)
- [PASS] (sag760) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [회귀 PASS] (nominal) 정상 주기해 잔차 (한 전기 주기)
- [회귀 PASS] (nominal) 회전자각 발진기 보존 \|cos²+sin²−1\|
- [회귀 PASS] (nominal) PWM 기본파 정상분 (펄스 Fourier) vs 스위칭 해 평균 v_d, v_q
- [회귀 PASS] (N27) 정상 주기해 잔차 (한 전기 주기)
- [회귀 PASS] (N27) 회전자각 발진기 보존 \|cos²+sin²−1\|
- [회귀 PASS] (N27) PWM 기본파 정상분 (펄스 Fourier) vs 스위칭 해 평균 v_d, v_q
- [회귀 PASS] (spwm) 정상 주기해 잔차 (한 전기 주기)
- [회귀 PASS] (spwm) 회전자각 발진기 보존 \|cos²+sin²−1\|
- [회귀 PASS] (spwm) PWM 기본파 정상분 (펄스 Fourier) vs 스위칭 해 평균 v_d, v_q
- [회귀 PASS] (sag760) 정상 주기해 잔차 (한 전기 주기)
- [회귀 PASS] (sag760) 회전자각 발진기 보존 \|cos²+sin²−1\|
- [회귀 PASS] (sag760) PWM 기본파 정상분 (펄스 Fourier) vs 스위칭 해 평균 v_d, v_q

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

## FL07 · 자성체 — 컨버터 전문가로 가는 실제 관문


**winding_flux** — 독립 11 · 회귀 3

- [PASS] (textbook) L_m → ∞ 한계: 스위칭 해 I_rms vs FL08 폐형식 — L_m = 1 MH·무손실 T-모델 정확 해 vs I_pk√(1 − 2φ/3π)
- [PASS] (textbook) B(t): 권선 전압 사다리꼴 적분 vs 자화전류 상태 L_m i_m/(N A_e) — 출력 v_m 표본의 누적 사다리꼴 적분 (Faraday) vs 상태변수 i_m (서로 다른 양)
- [PASS] (textbook) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (textbook) 독립 solver: (i₁, i₂′) 상태·3×3 선형계 RK45 — 다른 상태 선택(i₁, i₂′)과 매 단계 3×3 해로 쓴 ODE를 RK45 rtol 1e-7 → 1e-9 vs 정확 해 (i₁, i_m)
- [PASS] (fl08) L_m → ∞ 한계: 스위칭 해 I_rms vs FL08 폐형식 — L_m = 1 MH·무손실 T-모델 정확 해 vs I_pk√(1 − 2φ/3π)
- [PASS] (fl08) B(t): 권선 전압 사다리꼴 적분 vs 자화전류 상태 L_m i_m/(N A_e) — 출력 v_m 표본의 누적 사다리꼴 적분 (Faraday) vs 상태변수 i_m (서로 다른 양)
- [PASS] (fl08) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (fl08) 독립 solver: (i₁, i₂′) 상태·3×3 선형계 RK45 — 다른 상태 선택(i₁, i₂′)과 매 단계 3×3 해로 쓴 ODE를 RK45 rtol 1e-7 → 1e-9 vs 정확 해 (i₁, i_m)
- [PASS] (mismatch) B(t): 권선 전압 사다리꼴 적분 vs 자화전류 상태 L_m i_m/(N A_e) — 출력 v_m 표본의 누적 사다리꼴 적분 (Faraday) vs 상태변수 i_m (서로 다른 양)
- [PASS] (mismatch) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (mismatch) 독립 solver: (i₁, i₂′) 상태·3×3 선형계 RK45 — 다른 상태 선택(i₁, i₂′)과 매 단계 3×3 해로 쓴 ODE를 RK45 rtol 1e-7 → 1e-9 vs 정확 해 (i₁, i_m)
- [회귀 PASS] (textbook) zero-DC 주기해 잔차 (한 주기)
- [회귀 PASS] (fl08) zero-DC 주기해 잔차 (한 주기)
- [회귀 PASS] (mismatch) zero-DC 주기해 잔차 (한 주기)

**flux_walk** — 독립 10 · 회귀 2

- [PASS] (textbook) 주기당 자속 이동: 정확 해 vs volt-second 식 — 행렬지수 적분의 B(T) − B(0) vs V·Δt/(N A_e)
- [PASS] (textbook) B: 권선 전압 적분 vs 자화전류 상태 — v_w 표본 누적 사다리꼴 적분 vs L_m i_m/(N A_e)
- [PASS] (textbook) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (textbook) 동기 표본 적분 drift: 수치 vs 사다리꼴 edge 오차식 — 표본 파형의 누적 사다리꼴 적분 기울기 vs 반 표본 어긋난 두 edge의 오차 합 V·Δt_s/(N A_e)
- [PASS] (R_only) B: 권선 전압 적분 vs 자화전류 상태 — v_w 표본 누적 사다리꼴 적분 vs L_m i_m/(N A_e)
- [PASS] (R_only) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (R_only) 동기 표본 적분 drift: 수치 vs 사다리꼴 edge 오차식 — 표본 파형의 누적 사다리꼴 적분 기울기 vs 반 표본 어긋난 두 edge의 오차 합 V·Δt_s/(N A_e)
- [PASS] (blocking) B: 권선 전압 적분 vs 자화전류 상태 — v_w 표본 누적 사다리꼴 적분 vs L_m i_m/(N A_e)
- [PASS] (blocking) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (blocking) 동기 표본 적분 drift: 수치 vs 사다리꼴 edge 오차식 — 표본 파형의 누적 사다리꼴 적분 기울기 vs 반 표본 어긋난 두 edge의 오차 합 V·Δt_s/(N A_e)
- [회귀 INFO] (R_only) 주기당 자속 이동 (R 있음)
- [회귀 INFO] (blocking) 주기당 자속 이동 (R 있음)

**loss_screen** — 독립 11 · 회귀 0

- [PASS] (nominal) 환산 규칙: 두 회로의 저장에너지 파형 — L′ = n²L, C′ = C/n², R′ = n²R로 만든 1차 환산 회로 vs 2차 실제 값 회로 (둘 다 정확 해) — 한 주기 9개 시각의 ½Li² + ½Cv²
- [PASS] (nominal) skin depth: 폐형식 vs 1-D 확산 유한차분 — d²H/dx² = jωμσH를 중앙차분(6000 격자)으로 풀고 ln\|H\| 기울기로 감쇠 길이 추정 vs √(ρ/(πfμ))
- [PASS] (nominal) 원형선 R_ac/R_dc: Bessel 정확식 vs 저주파 전개 1 + (a/δ)⁴/48 — 복소 Bessel J₀/J₁ vs 급수 전개 (a = 0.2δ)
- [PASS] (nominal) Dowell 1층: 정확식 vs 저주파 전개 1 + (5m²−1)Δ⁴/45 — 쌍곡·삼각 함수식 vs 급수 전개 (Δ = 0.2)
- [PASS] (nominal) Dowell 4층: 정확식 vs 저주파 전개 1 + (5m²−1)Δ⁴/45 — 쌍곡·삼각 함수식 vs 급수 전개 (Δ = 0.2)
- [PASS] (nominal) iGSE: 삼각 자속 폐형식 vs 실제 파형 수치 적분 — k_i(4B_pk f)^α(2B_pk)^{β−α} vs 권선 전압 표본의 ∫\|dB/dt\|^α dt (R₂′ 강하만큼 차이)
- [PASS] (split) 환산 규칙: 두 회로의 저장에너지 파형 — L′ = n²L, C′ = C/n², R′ = n²R로 만든 1차 환산 회로 vs 2차 실제 값 회로 (둘 다 정확 해) — 한 주기 9개 시각의 ½Li² + ½Cv²
- [PASS] (split) skin depth: 폐형식 vs 1-D 확산 유한차분 — d²H/dx² = jωμσH를 중앙차분(6000 격자)으로 풀고 ln\|H\| 기울기로 감쇠 길이 추정 vs √(ρ/(πfμ))
- [PASS] (split) 원형선 R_ac/R_dc: Bessel 정확식 vs 저주파 전개 1 + (a/δ)⁴/48 — 복소 Bessel J₀/J₁ vs 급수 전개 (a = 0.2δ)
- [PASS] (split) Dowell 1층: 정확식 vs 저주파 전개 1 + (5m²−1)Δ⁴/45 — 쌍곡·삼각 함수식 vs 급수 전개 (Δ = 0.2)
- [PASS] (split) Dowell 4층: 정확식 vs 저주파 전개 1 + (5m²−1)Δ⁴/45 — 쌍곡·삼각 함수식 vs 급수 전개 (Δ = 0.2)

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


**fha_gain** — 독립 10 · 회귀 0

- [PASS] (textbook) 교재 정규화식 vs 페이저 절점해석 — 1/H = 1 + (1−F⁻²)/k + jQ(F−1/F) (reference) vs 절점 전압 방정식 (Z_r, Z_m, R_ac)
- [PASS] (textbook) ∠Z_in 경계: 닫힌 식 vs 페이저 Im Z_in의 근 — k²Q²y² + (1+k−k²Q²)y − 1 = 0 (y = F²) vs brentq(Im Z_in(F)) on the nodal phasor
- [PASS] (textbook) FHA 회로의 시간영역 주기해 vs 페이저 — 사인파 전원(발진기 상태)·R_ac 부하 회로를 행렬지수로 풀어 v_load 진폭 비교
- [PASS] (textbook) 극한: F = 1에서 모든 Q의 \|H\| = 1 — 직렬 공진에서 Z_r = 0
- [PASS] (textbook) 극한: Q → 0 (무부하) \|H\| = 1/\|1 + (1−F⁻²)/k\| — L_r·C_r·L_m 분압만 남는다
- [PASS] (hb) 교재 정규화식 vs 페이저 절점해석 — 1/H = 1 + (1−F⁻²)/k + jQ(F−1/F) (reference) vs 절점 전압 방정식 (Z_r, Z_m, R_ac)
- [PASS] (hb) ∠Z_in 경계: 닫힌 식 vs 페이저 Im Z_in의 근 — k²Q²y² + (1+k−k²Q²)y − 1 = 0 (y = F²) vs brentq(Im Z_in(F)) on the nodal phasor
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

## FL11 · PSFB·HV-LV·12 V 확장 — 선택의 이유를 설명하기


**psfb_duty_loss** — 독립 30 · 회귀 11

- [PASS] (nominal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (nominal) PWL 독립 경로: V_o (C_o→∞ 상수 출력 가정) — 구간별 직선 대수 + load line (reference/psfb.py) vs 정확 스위칭 해
- [PASS] (nominal) PWL 독립 경로: commutation 시간 t_c — t_c = 2I_0/(nV_in/L_k + V_o/L_o) vs 사건 위치
- [PASS] (nominal) PWL 독립 경로: 1차 RMS — PWL 정확 적분 vs Kronecker 모멘트 적분
- [PASS] (nominal) PWL 독립 경로: SR RMS — PWL 정확 적분 vs Kronecker 모멘트 적분
- [PASS] (nominal) L_k volt-second 항등식: ΔD = 4L_k f_s I_0/(n V_in) — 반주기 L_k 전압 적분(손유도) vs 시뮬레이션 평균 V_o
- [PASS] (nominal) 독립 solver: 손으로 쓴 ODE + RK45 사건 (20주기 과도) — 스칼라 식·별도 전환 규칙·solve_ivp 사건 vs 행렬지수 정확 해 (주기해의 0.8배에서 시작한 20주기 과도)
- [PASS] (seed_ideal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (seed_ideal) PWL 독립 경로: V_o (C_o→∞ 상수 출력 가정) — 구간별 직선 대수 + load line (reference/psfb.py) vs 정확 스위칭 해
- [PASS] (seed_ideal) PWL 독립 경로: commutation 시간 t_c — t_c = 2I_0/(nV_in/L_k + V_o/L_o) vs 사건 위치
- [PASS] (seed_ideal) PWL 독립 경로: 1차 RMS — PWL 정확 적분 vs Kronecker 모멘트 적분
- [PASS] (seed_ideal) PWL 독립 경로: SR RMS — PWL 정확 적분 vs Kronecker 모멘트 적분
- [PASS] (seed_ideal) L_k volt-second 항등식: ΔD = 4L_k f_s I_0/(n V_in) — 반주기 L_k 전압 적분(손유도) vs 시뮬레이션 평균 V_o
- [PASS] (seed_ideal) 독립 solver: 손으로 쓴 ODE + RK45 사건 (20주기 과도) — 스칼라 식·별도 전환 규칙·solve_ivp 사건 vs 행렬지수 정확 해 (주기해의 0.8배에서 시작한 20주기 과도)
- [PASS] (regulate48) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (regulate48) PWL 독립 경로: V_o (C_o→∞ 상수 출력 가정) — 구간별 직선 대수 + load line (reference/psfb.py) vs 정확 스위칭 해
- [PASS] (regulate48) PWL 독립 경로: commutation 시간 t_c — t_c = 2I_0/(nV_in/L_k + V_o/L_o) vs 사건 위치
- [PASS] (regulate48) PWL 독립 경로: 1차 RMS — PWL 정확 적분 vs Kronecker 모멘트 적분
- [PASS] (regulate48) PWL 독립 경로: SR RMS — PWL 정확 적분 vs Kronecker 모멘트 적분
- [PASS] (regulate48) L_k volt-second 항등식: ΔD = 4L_k f_s I_0/(n V_in) — 반주기 L_k 전압 적분(손유도) vs 시뮬레이션 평균 V_o
- [PASS] (regulate48) 독립 solver: 손으로 쓴 ODE + RK45 사건 (20주기 과도) — 스칼라 식·별도 전환 규칙·solve_ivp 사건 vs 행렬지수 정확 해 (주기해의 0.8배에서 시작한 20주기 과도)
- [PASS] (lossy_sr) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (lossy_sr) 독립 solver: 손으로 쓴 ODE + RK45 사건 (20주기 과도) — 스칼라 식·별도 전환 규칙·solve_ivp 사건 vs 행렬지수 정확 해 (주기해의 0.8배에서 시작한 20주기 과도)
- [PASS] (low_line) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (low_line) PWL 독립 경로: V_o (C_o→∞ 상수 출력 가정) — 구간별 직선 대수 + load line (reference/psfb.py) vs 정확 스위칭 해
- [PASS] (low_line) PWL 독립 경로: commutation 시간 t_c — t_c = 2I_0/(nV_in/L_k + V_o/L_o) vs 사건 위치
- [PASS] (low_line) PWL 독립 경로: 1차 RMS — PWL 정확 적분 vs Kronecker 모멘트 적분
- [PASS] (low_line) PWL 독립 경로: SR RMS — PWL 정확 적분 vs Kronecker 모멘트 적분
- [PASS] (low_line) L_k volt-second 항등식: ΔD = 4L_k f_s I_0/(n V_in) — 반주기 L_k 전압 적분(손유도) vs 시뮬레이션 평균 V_o
- [PASS] (low_line) 독립 solver: 손으로 쓴 ODE + RK45 사건 (20주기 과도) — 스칼라 식·별도 전환 규칙·solve_ivp 사건 vs 행렬지수 정확 해 (주기해의 0.8배에서 시작한 20주기 과도)
- [회귀 PASS] (nominal) shooting 주기해 잔차
- [회귀 INFO] (nominal) 주기별 정상상태 기준 (상태변화 < 1e-6, \|ΔW\|/E_ref < 1e-6, 3주기 연속)
- [회귀 PASS] (seed_ideal) shooting 주기해 잔차
- [회귀 INFO] (seed_ideal) 주기별 정상상태 기준 (상태변화 < 1e-6, \|ΔW\|/E_ref < 1e-6, 3주기 연속)
- [회귀 PASS] (regulate48) shooting 주기해 잔차
- [회귀 INFO] (regulate48) 주기별 정상상태 기준 (상태변화 < 1e-6, \|ΔW\|/E_ref < 1e-6, 3주기 연속)
- [회귀 PASS] (lossy_sr) shooting 주기해 잔차
- [회귀 NOT_RUN] (lossy_sr) PWL 독립 경로 / L_k 항등식
- [회귀 INFO] (lossy_sr) 주기별 정상상태 기준 (상태변화 < 1e-6, \|ΔW\|/E_ref < 1e-6, 3주기 연속)
- [회귀 PASS] (low_line) shooting 주기해 잔차
- [회귀 INFO] (low_line) 주기별 정상상태 기준 (상태변화 < 1e-6, \|ΔW\|/E_ref < 1e-6, 3주기 연속)

**lk_tradeoff** — 독립 1 · 회귀 1

- [PASS] (nominal) 선형 외삽 검증: 예측 최소 부하에서 직접 스위칭 해 — I_min = n·i_req + (I_o − I_0) 외삽 vs 그 부하로 다시 푼 정확 스위칭 해의 lagging 전환 전류
- [회귀 INFO] (nominal) duty loss 선형성: ΔD/L_k 기울기 vs 4 f_s I_0/(n V_in)

**psfb_vs_dab** — 독립 6 · 회귀 0

- [PASS] (nominal) DAB φ: 닫힌 식 vs PWL 전력 적분 근 — P = V1V2φ(1−φ/π)/(ωL) 역산 vs 4구간 PWL의 ∫v2'·i 적분을 brentq로 푼 φ
- [PASS] (nominal) DAB RMS: 닫힌 식 vs PWL — i_0·i_φ 식의 두 직선 RMS vs PWL.rms()
- [PASS] (nominal) DAB 입력·출력 전력 일치 (무손실) — ∫v1·i vs ∫v2'·i (PWL)
- [PASS] (nominal) DAB zero-DC 기준해: 전류 평균 0 — 반주기 반대칭 초기조건 → 주기 평균
- [PASS] (nominal) PSFB: PWL 독립 경로 1차 RMS — PWL 정확 적분 vs 정확 스위칭 해
- [PASS] (nominal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

**lv12_extension** — 독립 4 · 회귀 0

- [PASS] (nominal) 병렬 최적 N: 연속 해 vs 정수 sweep — dP/dN = 0 해석해 vs 정수 N 전수 계산
- [PASS] (nominal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (stray_fail) 병렬 최적 N: 연속 해 vs 정수 sweep — dP/dN = 0 해석해 vs 정수 N 전수 계산
- [PASS] (stray_fail) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

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


## EX03 · 병렬 SiC·gate loop·DPT 계측


**static_sharing** — 독립 4 · 회귀 2

- [PASS] (textbook) KCL 절점 풀이 vs 전류분배식 — 공통 단자전압 V를 KCL로 풀고 V/R_k (절점법) vs I·G_k/ΣG
- [PASS] (textbook) 전열 해: 고정점 반복 vs Newton (hybr) — 감쇠 없는 대입 반복 vs 다변수 Newton 계열 해법
- [PASS] (no_tempco) KCL 절점 풀이 vs 전류분배식 — 공통 단자전압 V를 KCL로 풀고 V/R_k (절점법) vs I·G_k/ΣG
- [PASS] (no_tempco) 전열 해: 고정점 반복 vs Newton (hybr) — 감쇠 없는 대입 반복 vs 다변수 Newton 계열 해법
- [회귀 PASS] (textbook) 전류 합 = I_total (KCL)
- [회귀 PASS] (no_tempco) 전류 합 = I_total (KCL)

**dpt_deskew** — 독립 18 · 회귀 0

- [PASS] (textbook) skew -5 ns: PWL 정확 적분 vs 손계산 닫힌 식 — 구간별 선형 곱의 정확 적분 vs 손으로 유도한 다항식 (넓은 창)
- [PASS] (textbook) skew +0 ns: PWL 정확 적분 vs 손계산 닫힌 식 — 구간별 선형 곱의 정확 적분 vs 손으로 유도한 다항식 (넓은 창)
- [PASS] (textbook) skew +5 ns: PWL 정확 적분 vs 손계산 닫힌 식 — 구간별 선형 곱의 정확 적분 vs 손으로 유도한 다항식 (넓은 창)
- [PASS] (textbook) skew −5 ns: PWL 정확 적분 vs 촘촘한 표본 사다리꼴 — 20 001점 균일 표본의 사다리꼴 적분 vs PWL 정확 적분
- [PASS] (textbook) 대역 제한 모델: 정확 ramp 중첩 vs scipy lsim (FOH) — 1차 저역통과의 해석 ramp 응답 합 vs 상태공간 이산화 시뮬레이션 (끝값)
- [PASS] (textbook) 참조면 표의 셀 에너지 원장 — FL02 합성 셀의 포트·소산·저장에너지 원장
- [PASS] (skew_m5) skew -5 ns: PWL 정확 적분 vs 손계산 닫힌 식 — 구간별 선형 곱의 정확 적분 vs 손으로 유도한 다항식 (넓은 창)
- [PASS] (skew_m5) skew +0 ns: PWL 정확 적분 vs 손계산 닫힌 식 — 구간별 선형 곱의 정확 적분 vs 손으로 유도한 다항식 (넓은 창)
- [PASS] (skew_m5) skew +5 ns: PWL 정확 적분 vs 손계산 닫힌 식 — 구간별 선형 곱의 정확 적분 vs 손으로 유도한 다항식 (넓은 창)
- [PASS] (skew_m5) skew −5 ns: PWL 정확 적분 vs 촘촘한 표본 사다리꼴 — 20 001점 균일 표본의 사다리꼴 적분 vs PWL 정확 적분
- [PASS] (skew_m5) 대역 제한 모델: 정확 ramp 중첩 vs scipy lsim (FOH) — 1차 저역통과의 해석 ramp 응답 합 vs 상태공간 이산화 시뮬레이션 (끝값)
- [PASS] (skew_m5) 참조면 표의 셀 에너지 원장 — FL02 합성 셀의 포트·소산·저장에너지 원장
- [PASS] (poor_deskew) skew -5 ns: PWL 정확 적분 vs 손계산 닫힌 식 — 구간별 선형 곱의 정확 적분 vs 손으로 유도한 다항식 (넓은 창)
- [PASS] (poor_deskew) skew +0 ns: PWL 정확 적분 vs 손계산 닫힌 식 — 구간별 선형 곱의 정확 적분 vs 손으로 유도한 다항식 (넓은 창)
- [PASS] (poor_deskew) skew +5 ns: PWL 정확 적분 vs 손계산 닫힌 식 — 구간별 선형 곱의 정확 적분 vs 손으로 유도한 다항식 (넓은 창)
- [PASS] (poor_deskew) skew −5 ns: PWL 정확 적분 vs 촘촘한 표본 사다리꼴 — 20 001점 균일 표본의 사다리꼴 적분 vs PWL 정확 적분
- [PASS] (poor_deskew) 대역 제한 모델: 정확 ramp 중첩 vs scipy lsim (FOH) — 1차 저역통과의 해석 ramp 응답 합 vs 상태공간 이산화 시뮬레이션 (끝값)
- [PASS] (poor_deskew) 참조면 표의 셀 에너지 원장 — FL02 합성 셀의 포트·소산·저장에너지 원장

**dynamic_sharing** — 독립 9 · 회귀 0

- [PASS] (nominal) 에너지 잔차 (4 branch 셀 전체) — V_bus·∫i_dc + 드라이버 − 부하 − (채널·diode·저항 소산) − ΔW; 상태식과 독립 정의
- [PASS] (nominal) 허용오차 강화 수렴 (rtol 1e-6 → 1e-8) — 같은 셀 재적분: branch peak·branch E_on 변화
- [PASS] (nominal) 한계 경우: 동일·비결합 4 branch = 4배 die 단일 소자의 1/4 — 4-branch 상태식(20+ 상태) vs L·R을 1/4로 환산한 단일 소자 셀(다른 행렬·차원)
- [PASS] (skew10) 에너지 잔차 (4 branch 셀 전체) — V_bus·∫i_dc + 드라이버 − 부하 − (채널·diode·저항 소산) − ΔW; 상태식과 독립 정의
- [PASS] (skew10) 허용오차 강화 수렴 (rtol 1e-6 → 1e-8) — 같은 셀 재적분: branch peak·branch E_on 변화
- [PASS] (skew10) 한계 경우: 동일·비결합 4 branch = 4배 die 단일 소자의 1/4 — 4-branch 상태식(20+ 상태) vs L·R을 1/4로 환산한 단일 소자 셀(다른 행렬·차원)
- [PASS] (kelvin) 에너지 잔차 (4 branch 셀 전체) — V_bus·∫i_dc + 드라이버 − 부하 − (채널·diode·저항 소산) − ΔW; 상태식과 독립 정의
- [PASS] (kelvin) 허용오차 강화 수렴 (rtol 1e-6 → 1e-8) — 같은 셀 재적분: branch peak·branch E_on 변화
- [PASS] (kelvin) 한계 경우: 동일·비결합 4 branch = 4배 die 단일 소자의 1/4 — 4-branch 상태식(20+ 상태) vs L·R을 1/4로 환산한 단일 소자 셀(다른 행렬·차원)

## EX04 · transformer 설계 closure — 전기적으로 가능한 n·L을 실제 부품으로


**turns_window** — 독립 10 · 회귀 0

- [PASS] (textbook) I_p: 정확 PWL 적분 vs FL08 폐형식 — 구간 끝점 대수 ∫i²dt vs I_pk√(1 − 2φ/3π) (정합 n)
- [PASS] (textbook) I_p (50:3): 정확 스위칭 해 vs PWL 대수 — 행렬지수 전파(엔진, L_m = 1 GH 한계) vs 교재 4구간 끝점 식으로 만든 PWL
- [PASS] (textbook) I_p (49:3): 정확 스위칭 해 vs PWL 대수 — 행렬지수 전파(엔진, L_m = 1 GH 한계) vs 교재 4구간 끝점 식으로 만든 PWL
- [PASS] (textbook) B_pk: 권선 전압 적분(정확 해) vs n·V_L/(4 f N_p A_e) — 자화전류 상태 L_m·i_m/(N A_e)의 peak-to-peak/2 vs 사각파 폐형식 (R = 0, k = 1)
- [PASS] (textbook) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (insulation) I_p: 정확 PWL 적분 vs FL08 폐형식 — 구간 끝점 대수 ∫i²dt vs I_pk√(1 − 2φ/3π) (정합 n)
- [PASS] (insulation) I_p (50:3): 정확 스위칭 해 vs PWL 대수 — 행렬지수 전파(엔진, L_m = 1 GH 한계) vs 교재 4구간 끝점 식으로 만든 PWL
- [PASS] (insulation) I_p (49:3): 정확 스위칭 해 vs PWL 대수 — 행렬지수 전파(엔진, L_m = 1 GH 한계) vs 교재 4구간 끝점 식으로 만든 PWL
- [PASS] (insulation) B_pk: 권선 전압 적분(정확 해) vs n·V_L/(4 f N_p A_e) — 자화전류 상태 L_m·i_m/(N A_e)의 peak-to-peak/2 vs 사각파 폐형식 (R = 0, k = 1)
- [PASS] (insulation) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

**cllc_turns** — 독립 6 · 회귀 0

- [PASS] (nominal) FHA 해: 벡터화 이분법 vs brentq — 격자 부호변화 + 전 후보 동시 이분법 vs 조밀 격자 + scipy brentq (서로 다른 root solver)
- [PASS] (nominal) FHA 전력 항등식: \|I₁\|²Re Z_in = \|I₂′\|²R_ac′ = P — 해에서 입력 임피던스로 계산한 입력 전력과 전류분배로 계산한 R_ac′ 전력 vs 정격 P (무손실 tank)
- [PASS] (nominal) 대칭 tank의 부하 무관점 \|H(f_r)\| = 1 — 환산 대칭(L_r2′ = L_r1, C_r2′ = C_r1)이면 f_r에서 Z_r = 0 → H = 1 (n·부하와 무관) — FHA 구현의 물리 항등식
- [PASS] (retune) FHA 해: 벡터화 이분법 vs brentq — 격자 부호변화 + 전 후보 동시 이분법 vs 조밀 격자 + scipy brentq (서로 다른 root solver)
- [PASS] (retune) FHA 전력 항등식: \|I₁\|²Re Z_in = \|I₂′\|²R_ac′ = P — 해에서 입력 임피던스로 계산한 입력 전력과 전류분배로 계산한 R_ac′ 전력 vs 정격 P (무손실 tank)
- [PASS] (retune) 대칭 tank의 부하 무관점 \|H(f_r)\| = 1 — 환산 대칭(L_r2′ = L_r1, C_r2′ = C_r1)이면 f_r에서 Z_r = 0 → H = 1 (n·부하와 무관) — FHA 구현의 물리 항등식

**harmonic_copper** — 독립 6 · 회귀 0

- [PASS] (textbook) Parseval: 시간영역 RMS (합성 파형, 임의 위상) vs √ΣI_h² — 차수 1·3·5와 임의 위상으로 만든 파형의 수치 적분 vs 성분 합
- [PASS] (dab) Parseval: Σ\|I_h\|² (h ≤ 399) vs 정확 PWL ∫i²dt — 구간별 닫힌 형태 Fourier 계수의 제곱합 vs 시간영역 구간 끝점 대수 (서로 다른 경로)
- [PASS] (dab) FFT (2¹⁴ 표본) vs 닫힌 형태 계수 (h = 1…15) — 균일 표본의 이산 Fourier 변환 vs 구간 적분식
- [PASS] (dab) 기울기 점프 공식 vs 구간 적분식 (h = 1…399) — 두 번 부분적분한 c_h = ΣΔs_k e^{−jhωt_k}/(T(jhω)²) vs 구간별 ∫(y₀+sτ)e^{−jhωτ}dτ (서로 다른 유도)
- [PASS] (dab) I_rms: PWL vs FL08 폐형식 — 구간 대수 vs I_pk√(1 − 2φ/3π)
- [PASS] (dab) Dowell 저주파 한계 F_R(Δ → 0) → 1 — 쌍곡·삼각 함수식의 극한

**tolerance_map** — 독립 8 · 회귀 0

- [PASS] (textbook) FHA 해: 벡터화 이분법 (MC 경로) vs brentq — Monte Carlo에 쓰는 격자·동시 이분법 vs 조밀 격자 + brentq, corner 5개 × 운전점 3개 (해 개수 포함)
- [PASS] (textbook) FHA 전력 항등식 (모든 corner의 해) — \|I₁\|²Re Z_in = \|I₂′\|²R_ac′ = P (무손실 tank의 에너지 보존)
- [PASS] (textbook) 각 corner의 \|H(f_r)\| = 1 (대칭 tank, 부하 무관) — f_r에서 Z_r = 0이면 H = Z_p/Z_p·R_ac′/R_ac′ = 1 — corner의 f_r 폐형식과 FHA 이득식이 같은 점을 가리키는지
- [PASS] (textbook) MC σ(ln f_r) vs 해석식 — 난수 표본의 표준편차 vs ln f_r = −½(ln L + ln C) + 상수의 분산식 (4 표준오차 이내)
- [PASS] (correlated) FHA 해: 벡터화 이분법 (MC 경로) vs brentq — Monte Carlo에 쓰는 격자·동시 이분법 vs 조밀 격자 + brentq, corner 5개 × 운전점 3개 (해 개수 포함)
- [PASS] (correlated) FHA 전력 항등식 (모든 corner의 해) — \|I₁\|²Re Z_in = \|I₂′\|²R_ac′ = P (무손실 tank의 에너지 보존)
- [PASS] (correlated) 각 corner의 \|H(f_r)\| = 1 (대칭 tank, 부하 무관) — f_r에서 Z_r = 0이면 H = Z_p/Z_p·R_ac′/R_ac′ = 1 — corner의 f_r 폐형식과 FHA 이득식이 같은 점을 가리키는지
- [PASS] (correlated) MC σ(ln f_r) vs 해석식 — 난수 표본의 표준편차 vs ln f_r = −½(ln L + ln C) + 상수의 분산식 (4 표준오차 이내)

**identification** — 독립 5 · 회귀 2

- [PASS] (nominal) 임피던스: 정확 시간영역 해(Fourier 사영) vs phasor ladder — 사인 전원을 진동자 상태로 둔 상태방정식의 주기해(행렬지수)에서 전류의 cos·sin 성분을 정확 적분 vs 복소 ladder 식
- [PASS] (nominal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (nominal) SRF: 수치 해 vs 근사식 1/(2π√(L_eff·C)) — ladder의 Im Z = 0 근 vs 집중 LC 근사 (R·C 분배 무시 → 5 % 허용)
- [PASS] (sc_hv_short) 임피던스: 정확 시간영역 해(Fourier 사영) vs phasor ladder — 사인 전원을 진동자 상태로 둔 상태방정식의 주기해(행렬지수)에서 전류의 cos·sin 성분을 정확 적분 vs 복소 ladder 식
- [PASS] (sc_hv_short) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [회귀 PASS] (nominal) 주기해 잔차 (한 주기)
- [회귀 PASS] (sc_hv_short) 주기해 잔차 (한 주기)

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

## EX08 · 전열 연성·mission — 최고온도 한 점에서 mission으로


**thermal_matrix** — 독립 12 · 회귀 0

- [PASS] (textbook) 고정점 반복 수렴값 vs 선형 풀이 — ΔT_(k+1) = Z(P₀ + diag(αP₀)ΔT_k) 60회 vs (I − Z·diag(αP₀))⁻¹Z·P₀
- [PASS] (textbook) 물리 망의 port 행렬 = Z (정상상태) — 8노드 Cauer 망의 컨덕턴스 행렬 풀이 vs 입력 Z
- [PASS] (textbook) 연성 상승: 2×2 행렬 풀이 vs 8노드 물리 망 정상상태 — junction 손실 되먹임을 넣은 망의 평형점 −A⁻¹b vs (I − Z·D)⁻¹Z·P₀
- [PASS] (textbook) 연성 상승 ΔT₂: 행렬 vs 물리 망 — 같은 두 경로
- [PASS] (textbook) 물리 망 과도의 끝값 (30/\|λ_max\| 뒤) vs 연성 정상해 — 증강 행렬지수 정확 적분 (냉간 시작, 강성 망의 긴 구간) vs (I − Z·D)⁻¹Z·P₀
- [PASS] (near) 고정점 반복 수렴값 vs 선형 풀이 — ΔT_(k+1) = Z(P₀ + diag(αP₀)ΔT_k) 182회 vs (I − Z·diag(αP₀))⁻¹Z·P₀
- [PASS] (near) 물리 망의 port 행렬 = Z (정상상태) — 8노드 Cauer 망의 컨덕턴스 행렬 풀이 vs 입력 Z
- [PASS] (near) 연성 상승: 2×2 행렬 풀이 vs 8노드 물리 망 정상상태 — junction 손실 되먹임을 넣은 망의 평형점 −A⁻¹b vs (I − Z·D)⁻¹Z·P₀
- [PASS] (near) 연성 상승 ΔT₂: 행렬 vs 물리 망 — 같은 두 경로
- [PASS] (near) 물리 망 과도의 끝값 (30/\|λ_max\| 뒤) vs 연성 정상해 — 증강 행렬지수 정확 적분 (냉간 시작, 강성 망의 긴 구간) vs (I − Z·D)⁻¹Z·P₀
- [PASS] (diverge) 물리 망의 port 행렬 = Z (정상상태) — 8노드 Cauer 망의 컨덕턴스 행렬 풀이 vs 입력 Z
- [PASS] (diverge) 발산 속도: 과도 적분의 ln\|T − T_lin\| 기울기 vs 최대 고유값 — 정확 적분 궤적 (3/λ ~ 5/λ 구간) vs 고유값 분해

**dynamic_network** — 독립 8 · 회귀 0

- [PASS] (nominal) 상반성 (reciprocity): Z₁₂(t) = Z₂₁(t) — P₂ 계단 → T_j1과 P₁ 계단 → T_j2를 따로 적분해 비교 (수동 RC 망의 성질)
- [PASS] (nominal) 정상상태 port 행렬 = Z — 망 컨덕턴스 풀이 vs 입력 열 행렬
- [PASS] (nominal) 모달(고유벡터) 해석해 vs 엔진 행렬지수 (Z₁₁(t)) — 일반화 고유값 분해의 모달 합 vs 증강 행렬지수
- [PASS] (nominal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (light_sink) 상반성 (reciprocity): Z₁₂(t) = Z₂₁(t) — P₂ 계단 → T_j1과 P₁ 계단 → T_j2를 따로 적분해 비교 (수동 RC 망의 성질)
- [PASS] (light_sink) 정상상태 port 행렬 = Z — 망 컨덕턴스 풀이 vs 입력 열 행렬
- [PASS] (light_sink) 모달(고유벡터) 해석해 vs 엔진 행렬지수 (Z₁₁(t)) — 일반화 고유값 분해의 모달 합 vs 증강 행렬지수
- [PASS] (light_sink) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)

**mission** — 독립 4 · 회귀 2

- [PASS] (nominal) rainflow: ASTM E1049 예제 재현 — 표준 예제 [−2, 1, −3, 5, −1, 3, −4, 4, −2] → 범위 3(0.5)·4(1.5)·6(0.5)·8(1.0)·9(0.5)
- [PASS] (nominal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (hot_hill) rainflow: ASTM E1049 예제 재현 — 표준 예제 [−2, 1, −3, 5, −1, 3, −4, 4, −2] → 범위 3(0.5)·4(1.5)·6(0.5)·8(1.0)·9(0.5)
- [PASS] (hot_hill) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [회귀 PASS] (nominal) 사이클 합 = (반전점 수 − 1)/2 (연성)
- [회귀 PASS] (hot_hill) 사이클 합 = (반전점 수 − 1)/2 (연성)

## EX09 · EMI: noise source, 경로, victim을 같이 본다


**hand_screens** — 독립 3 · 회귀 0

- [PASS] (textbook) 변위전류: 정확 시간영역 (낮은 경로 임피던스) vs C·dv/dt — R–L–C 경로에 dv/dt 램프를 건 행렬지수 해의 램프 끝 전류 vs 곱셈 screen
- [PASS] (textbook) 링잉 주파수: 정확 RLC 해의 영점 교차 vs 닫힌 식 — 행렬지수 스텝 응답을 조밀 샘플해 선형 보간 영점 교차 → 주파수 vs f_0√(1−ζ²)
- [PASS] (textbook) line 전류: i = C dv/dt 수치 구적 RMS vs 2πfCV — √2·230 V 정현파의 C dv/dt를 한 주기 수치 적분 (quad) vs phasor 식

**source_path_victim** — 독립 5 · 회귀 1

- [PASS] (nominal) 에너지 잔차 (E_in − E_out − E_loss − ΔW) — 포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)
- [PASS] (nominal) FFT 정규화: 해석 사다리꼴 샘플 FFT vs 닫힌 식 2VD\|sinc(nD)\|\|sinc(n f t_r)\| — numpy FFT(균일 샘플) vs 교과서 Fourier 계수
- [PASS] (nominal) CM 경로 전달함수: 상태공간 (셀 행렬) vs 임피던스 대수 — C(jωI−A)⁻¹B (시뮬레이션이 쓰는 행렬) vs Z 직병렬 대수 (reference/emi.py)
- [PASS] (nominal) Parseval: proxy 전압 RMS (시간영역) vs Σ\|V_n H_cm(f_n)\|²/2 — 정확 모멘트 적분 RMS vs 노드 전압 FFT × 임피던스 대수 전달함수의 고조파 합
- [PASS] (nominal) DM 경로 전달함수: 2-node admittance 해 vs 전류 분배식 — Y 행렬 선형해 vs Z 분배 대수
- [회귀 PASS] (nominal) 주기별 정상상태 기준 (정규화 상태 변화 < 1e-9)

**mitigation** — 독립 6 · 회귀 0

- [PASS] (nominal) snubber 에너지: R_s 적분 + 채널 증가분 vs C_s V² f_sw — 정확 시간영역 에너지 적분 (R_s·i² + ∫v·i_ch 차이) vs 닫힌 식 (두 edge마다 ½C_sV²)
- [PASS] (nominal) 에너지 원장: 기준 — 부하 source·DC link·채널·R_loop·R_s·R_par·측정 R 에너지 정확 적분
- [PASS] (nominal) 에너지 원장: gate 느리게 (×2) — 부하 source·DC link·채널·R_loop·R_s·R_par·측정 R 에너지 정확 적분
- [PASS] (nominal) 에너지 원장: C_par ×0.5 (두꺼운 절연) — 부하 source·DC link·채널·R_loop·R_s·R_par·측정 R 에너지 정확 적분
- [PASS] (nominal) 에너지 원장: RC snubber (470 pF, 3.16 Ω) — 부하 source·DC link·채널·R_loop·R_s·R_par·측정 R 에너지 정확 적분
- [PASS] (nominal) 에너지 원장: Y 커패시터 2.2 nF (선당) — 부하 source·DC link·채널·R_loop·R_s·R_par·측정 R 에너지 정확 적분

## EX10 · 고장: gate를 꺼도 에너지는 남아 있다


**rlc_fault** — 독립 12 · 회귀 0

- [PASS] (textbook) 에너지 잔차 W₀ − W(t) − E_R − E_D — ½Li² + ½Cv² (상태) vs 저항·diode 전력의 정확 적분 (2차 모멘트) — 상태식과 따로 정의한 에너지 항
- [PASS] (textbook) peak 시각: DOP853 event vs 정확 엔진 — 손으로 쓴 ODE + event(di/dt = 0) vs 행렬지수 해
- [PASS] (textbook) 독립 solver: DOP853 전류 (여러 시각) — 손으로 쓴 RLC ODE (rtol 1e-12, clamp는 terminal event) vs 정확 엔진
- [PASS] (textbook) E_R: DOP853 적분 상태 vs 정확 2차 모멘트 — ODE에 ∫Ri² 상태를 추가해 적분 vs 행렬지수 2차 모멘트
- [PASS] (clamp) 에너지 잔차 W₀ − W(t) − E_R − E_D — ½Li² + ½Cv² (상태) vs 저항·diode 전력의 정확 적분 (2차 모멘트) — 상태식과 따로 정의한 에너지 항
- [PASS] (clamp) peak 시각: DOP853 event vs 정확 엔진 — 손으로 쓴 ODE + event(di/dt = 0) vs 행렬지수 해
- [PASS] (clamp) 독립 solver: DOP853 전류 (여러 시각) — 손으로 쓴 RLC ODE (rtol 1e-12, clamp는 terminal event) vs 정확 엔진
- [PASS] (clamp) E_R: DOP853 적분 상태 vs 정확 2차 모멘트 — ODE에 ∫Ri² 상태를 추가해 적분 vs 행렬지수 2차 모멘트
- [PASS] (over) 에너지 잔차 W₀ − W(t) − E_R − E_D — ½Li² + ½Cv² (상태) vs 저항·diode 전력의 정확 적분 (2차 모멘트) — 상태식과 따로 정의한 에너지 항
- [PASS] (over) peak 시각: DOP853 event vs 정확 엔진 — 손으로 쓴 ODE + event(di/dt = 0) vs 행렬지수 해
- [PASS] (over) 독립 solver: DOP853 전류 (여러 시각) — 손으로 쓴 RLC ODE (rtol 1e-12, clamp는 terminal event) vs 정확 엔진
- [PASS] (over) E_R: DOP853 적분 상태 vs 정확 2차 모멘트 — ODE에 ∫Ri² 상태를 추가해 적분 vs 행렬지수 2차 모멘트

**protection_timeline** — 독립 12 · 회귀 0

- [PASS] (textbook) 소거 시각 (typ): 이산 사건 시계 vs timing 대수 — 0.1 ns 틱으로 검출기 상태기계를 진행 vs max/합 대수식
- [PASS] (textbook) 소거 시각 (max): 이산 사건 시계 vs timing 대수 — 0.1 ns 틱으로 검출기 상태기계를 진행 vs max/합 대수식
- [PASS] (textbook) 소자 에너지: 폐형식 vs 수치 적분 — 구간별 폐형식 (plateau + 삼각 fall) vs v_DS·i 파형 사다리꼴 적분 (20001점)
- [PASS] (type2) 소거 시각 (typ): 이산 사건 시계 vs timing 대수 — 0.1 ns 틱으로 검출기 상태기계를 진행 vs max/합 대수식
- [PASS] (type2) 소거 시각 (max): 이산 사건 시계 vs timing 대수 — 0.1 ns 틱으로 검출기 상태기계를 진행 vs max/합 대수식
- [PASS] (type2) 소자 에너지: 폐형식 vs 수치 적분 — 구간별 폐형식 (plateau + 삼각 fall) vs v_DS·i 파형 사다리꼴 적분 (20001점)
- [PASS] (soft_short) 소거 시각 (typ): 이산 사건 시계 vs timing 대수 — 0.1 ns 틱으로 검출기 상태기계를 진행 vs max/합 대수식
- [PASS] (soft_short) 소거 시각 (max): 이산 사건 시계 vs timing 대수 — 0.1 ns 틱으로 검출기 상태기계를 진행 vs max/합 대수식
- [PASS] (soft_short) 소자 에너지: 폐형식 vs 수치 적분 — 구간별 폐형식 (plateau + 삼각 fall) vs v_DS·i 파형 사다리꼴 적분 (20001점)
- [PASS] (fast_off) 소거 시각 (typ): 이산 사건 시계 vs timing 대수 — 0.1 ns 틱으로 검출기 상태기계를 진행 vs max/합 대수식
- [PASS] (fast_off) 소거 시각 (max): 이산 사건 시계 vs timing 대수 — 0.1 ns 틱으로 검출기 상태기계를 진행 vs max/합 대수식
- [PASS] (fast_off) 소자 에너지: 폐형식 vs 수치 적분 — 구간별 폐형식 (plateau + 삼각 fall) vs v_DS·i 파형 사다리꼴 적분 (20001점)

**discharge_resistor** — 독립 6 · 회귀 2

- [PASS] (textbook) 에너지: ∫v²/R dt vs ½C(V₀² − V_f²) — 저항 전력의 정확 적분 (2차 모멘트) vs 저장에너지 차이
- [PASS] (textbook) 60 V 도달: 정확 엔진 근 vs 폐형식 RC·ln(V₀/V_f) — 행렬지수 해의 v(t) = V_f 근 (brentq) vs 폐형식
- [PASS] (textbook) 60 V 도달: DOP853 event vs 정확 엔진 — 손으로 쓴 RC ODE + event vs 행렬지수 해
- [PASS] (worst_case) 에너지: ∫v²/R dt vs ½C(V₀² − V_f²) — 저항 전력의 정확 적분 (2차 모멘트) vs 저장에너지 차이
- [PASS] (worst_case) 60 V 도달: 정확 엔진 근 vs 폐형식 RC·ln(V₀/V_f) — 행렬지수 해의 v(t) = V_f 근 (brentq) vs 폐형식
- [PASS] (worst_case) 60 V 도달: DOP853 event vs 정확 엔진 — 손으로 쓴 RC ODE + event vs 행렬지수 해
- [회귀 PASS] (textbook) 직렬 저항 에너지 합 = 전체
- [회귀 PASS] (worst_case) 직렬 저항 에너지 합 = 전체

**safe_state** — 독립 16 · 회귀 0

- [PASS] (nominal) ASC 에너지 잔차 (축 기계입력 − 동손 − ΔW) — 전자기 전력·동손의 정확 2차 모멘트 적분 vs 0.75(L_d i_d² + L_q i_q²) 변화
- [PASS] (nominal) ASC 독립 solver (DOP853) — 손으로 쓴 dq ODE (v = 0) vs 행렬지수 해
- [PASS] (nominal) freewheel 에너지 잔차 (기계 − DC − 동손 − ΔW) — 역기전력×전류, DC 포트, 동손의 정확 적분 vs 권선(+C) 저장에너지 변화 — diode guard 오프셋(1e-7 A)의 영향 포함
- [PASS] (nominal) 비제어 정류 임계속도: 폐형식 vs diode 브리지 시뮬레이션 — V_dc/(√3pψ_m) 식 vs 0 A에서 시작한 hybrid 시뮬레이션의 평균 DC 전류 (0.98·n_th / 1.02·n_th)
- [PASS] (open) ASC 에너지 잔차 (축 기계입력 − 동손 − ΔW) — 전자기 전력·동손의 정확 2차 모멘트 적분 vs 0.75(L_d i_d² + L_q i_q²) 변화
- [PASS] (open) ASC 독립 solver (DOP853) — 손으로 쓴 dq ODE (v = 0) vs 행렬지수 해
- [PASS] (open) freewheel 에너지 잔차 (기계 − DC − 동손 − ΔW) — 역기전력×전류, DC 포트, 동손의 정확 적분 vs 권선(+C) 저장에너지 변화 — diode guard 오프셋(1e-7 A)의 영향 포함
- [PASS] (open) 비제어 정류 임계속도: 폐형식 vs diode 브리지 시뮬레이션 — V_dc/(√3pψ_m) 식 vs 0 A에서 시작한 hybrid 시뮬레이션의 평균 DC 전류 (0.98·n_th / 1.02·n_th)
- [PASS] (overspeed) ASC 에너지 잔차 (축 기계입력 − 동손 − ΔW) — 전자기 전력·동손의 정확 2차 모멘트 적분 vs 0.75(L_d i_d² + L_q i_q²) 변화
- [PASS] (overspeed) ASC 독립 solver (DOP853) — 손으로 쓴 dq ODE (v = 0) vs 행렬지수 해
- [PASS] (overspeed) freewheel 에너지 잔차 (기계 − DC − 동손 − ΔW) — 역기전력×전류, DC 포트, 동손의 정확 적분 vs 권선(+C) 저장에너지 변화 — diode guard 오프셋(1e-7 A)의 영향 포함
- [PASS] (overspeed) 비제어 정류 임계속도: 폐형식 vs diode 브리지 시뮬레이션 — V_dc/(√3pψ_m) 식 vs 0 A에서 시작한 hybrid 시뮬레이션의 평균 DC 전류 (0.98·n_th / 1.02·n_th)
- [PASS] (overspeed_open) ASC 에너지 잔차 (축 기계입력 − 동손 − ΔW) — 전자기 전력·동손의 정확 2차 모멘트 적분 vs 0.75(L_d i_d² + L_q i_q²) 변화
- [PASS] (overspeed_open) ASC 독립 solver (DOP853) — 손으로 쓴 dq ODE (v = 0) vs 행렬지수 해
- [PASS] (overspeed_open) freewheel 에너지 잔차 (기계 − DC − 동손 − ΔW) — 역기전력×전류, DC 포트, 동손의 정확 적분 vs 권선(+C) 저장에너지 변화 — diode guard 오프셋(1e-7 A)의 영향 포함
- [PASS] (overspeed_open) 비제어 정류 임계속도: 폐형식 vs diode 브리지 시뮬레이션 — V_dc/(√3pψ_m) 식 vs 0 A에서 시작한 hybrid 시뮬레이션의 평균 DC 전류 (0.98·n_th / 1.02·n_th)

## EX11 · 모델을 믿을 수 있는 범위: 검증·식별·불확도


**loss_uncertainty** — 독립 6 · 회귀 0

- [PASS] (textbook) u(loss): Monte Carlo (N = 200000, seed 20260930) vs GUM — 상관 Gaussian 표본(Cholesky)의 표본 표준편차 vs √(u_in² + u_out² − 2ρu_in u_out)
- [PASS] (textbook) u(η): Monte Carlo vs 1차 GUM (선형화 오차 포함) — P_out/P_in 비율의 표본 표준편차 vs 감도계수 식
- [PASS] (textbook) η 감도계수: 중앙차분 vs 해석 편미분 — ∂η/∂P 중앙차분으로 다시 조립한 u(η) vs 해석식
- [PASS] (rho09_nobasis) u(loss): Monte Carlo (N = 200000, seed 20260930) vs GUM — 상관 Gaussian 표본(Cholesky)의 표본 표준편차 vs √(u_in² + u_out² − 2ρu_in u_out)
- [PASS] (rho09_nobasis) u(η): Monte Carlo vs 1차 GUM (선형화 오차 포함) — P_out/P_in 비율의 표본 표준편차 vs 감도계수 식
- [PASS] (rho09_nobasis) η 감도계수: 중앙차분 vs 해석 편미분 — ∂η/∂P 중앙차분으로 다시 조립한 u(η) vs 해석식

**binomial_bounds** — 독립 2 · 회귀 0

- [PASS] (textbook) 상한: Beta 분위수 vs 이항 누적합의 근 (brentq) — scipy Beta ppf vs Σ_j≤k C(n,j)p^j(1−p)^(n−j) = 1−conf를 직접 합산해 푼 p
- [PASS] (textbook) 상한의 의미: p = 상한에서 k 이하 실패가 나올 확률 = 1 − conf (시뮬레이션) — numpy 이항 표본 20000회 (seed 7)

**identifiability** — 독립 4 · 회귀 0

- [PASS] (nominal) 합성 데이터 생성: 정확 엔진 vs 닫힌 식 step 응답 — 행렬지수 RLC 해 vs V(1 − e^{−αt}(cos ω_d t + α/ω_d sin ω_d t))
- [PASS] (nominal) 식별 가능한 조합: L·C와 R/L은 시작점과 무관 — 서로 다른 시작점의 적합 결과 비교
- [PASS] (nominal) Jacobian 계수 결손 (한 방향 비식별) — 적합점의 J = ∂잔차/∂log θ 특이값 분해
- [PASS] (nominal) 공동 적합이 참값을 불확도 안에서 회복 — 합성 참값 vs 공동 적합 (J 공분산의 표준편차 단위)

**monte_carlo** — 독립 2 · 회귀 2

- [PASS] (nominal) MC 수율 vs 해석 수율 (99.9 % CI 포함 여부) — seed 고정 Cholesky 표본 vs ln f 정규 분포의 Φ 식
- [PASS] (corr_pos) MC 수율 vs 해석 수율 (99.9 % CI 포함 여부) — seed 고정 Cholesky 표본 vs ln f 정규 분포의 Φ 식
- [회귀 PASS] (nominal) seed 재현성: 같은 seed → 같은 표본
- [회귀 PASS] (corr_pos) seed 재현성: 같은 seed → 같은 표본

**test_independence** — 독립 0 · 회귀 0


## EX12 · 설계 리뷰를 통과하는 답변 — 세 개의 통합 사례


**capstone_a** — 독립 1 · 회귀 1

- [PASS] (textbook) 부족분 결론이 불확도보다 큰가 — 부족분 vs 2 × (η·PF·전류 불확도 합성)
- [회귀 PASS] (textbook) 메모의 모든 수치가 실제 실행에서 왔는가

**capstone_b** — 독립 0 · 회귀 1

- [회귀 PASS] (textbook) 메모의 모든 수치가 실제 실행에서 왔는가

**capstone_c** — 독립 1 · 회귀 1

- [INFO] (textbook) screen 값과 합성 동특성 값이 다르다는 것을 드러냄 — 교재 screen(L·di/dt) vs EX03 합성 branch 모델
- [회귀 PASS] (textbook) 메모의 모든 수치가 실제 실행에서 왔는가
