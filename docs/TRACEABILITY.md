# 추적표 (Traceability) — 교재 v4.0 ↔ 실습 ↔ 코드 ↔ 테스트 ↔ 실행 결과

생성: `tools/gen_docs.py` · run-all 2026-09-30T19:46:57 · 앱 4.0.0 · 계약 4.0-reconstructed

열 설명: **학습 내용**(교재 절) · **구현**(실험 key) · **코드/테스트** · **실제 실행**(`convlab run-all`의 기준 preset 실행 수, 실패 수, 나온 상태) · **검증**(기준값 metric 통과 수 / 독립 경로 check 수) · **범위 한계** · **하드웨어 검증**(모두 미수행). 상태는 모델 안에서의 판정이며 학습 완료나 면접 준비 완료를 뜻하지 않는다.

| Lab | 교재 | 제목 | 실험 | 코드 | 테스트 (개수) | 실제 실행 | 검증 | 범위 한계 | 하드웨어 검증 |
|---|---|---|---|---|---|---|---|---|---|
| FL01 | 02. 전력변환을 다시 배우는 네 개의 법칙; 03. Buck와 Boost — 컨버터의 문법 [FL01] | Buck와 Boost — 컨버터의 문법 | buck_ccm, buck_light_load, buck_ripple, boost_rhpz | `src/convlab/labs/fl01_buck_boost.py` | `tests/test_fl01.py` (13) | 7 runs, 실패 0 · PASS_WITHIN_MODEL | 기준값 31/31 · 독립 검산 9 | 이상 스위치(C) 파형과 평균모델(B); 스위칭 손실·ZVS·링잉 미포함; RHP 응답은 저항부하·CCM·duty 입력 조건에서만 | 미수행 (하드웨어·실측 없음) |
| FL02 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| FL03 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| FL04 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| FL05 | 08. OBC·PFC — 전력품질을 설계 변수로 바꾸기 [FL05]; 02. 전력변환을 다시 배우는 네 개의 법칙 | OBC·PFC — 전력품질을 설계 변수로 바꾸기 | grid_boundary, pf_thd, boost_pfc, dclink_power | `src/convlab/labs/fl05_pfc.py` | `tests/test_fl05.py` (13) | 11 runs, 실패 0 · CUSTOMER_DECISION_REQUIRED / FAIL_CONSTRAINT / NOT_EVALUABLE / PASS_WITHIN_MODEL | 기준값 48/48 · 독립 검산 36 | η·PF 고정의 해석 경계(A); THD/PF는 이상 소자·합성 제어기·정의된 창의 모델 결과 — 규격 적합성·EMI 판정 아님 | 미수행 (하드웨어·실측 없음) |
| FL06 | 09. 제어 — PI보다 plant와 부호가 먼저다 [FL06]; E07. 제어: 단일 루프가 안정해도 시스템은 불안정할 수 있다 [EX07] | 제어 — PI보다 plant와 부호가 먼저다 | pi_design, discrete_loop, dq_sign | `src/convlab/labs/fl06_control.py` | `tests/test_fl06.py` (15) | 11 runs, 실패 0 · FAIL_CONSTRAINT / PASS_WITHIN_MODEL / UNSTABLE | 기준값 40/40 · 독립 검산 21 | 이상 평균 변조기의 선형·샘플링 루프 결과(A/B). PWM 리플·센서 필터·양자화 미포함; 위상여유·overshoot 기준은 학습용이며 고객 사양이 아님 | 미수행 (하드웨어·실측 없음) |
| FL07 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| FL08 | 11. DAB — 식에서 파형, 파형에서 설계 판단으로 [FL08]; 10. 자성체 [FL07] | DAB — 식에서 파형, 파형에서 설계 판단으로 | sps_nominal, zero_power_mismatch, voltage_corners, offset_startup, two_modules | `src/convlab/labs/fl08_dab.py` | `tests/test_fl08.py` (11) | 11 runs, 실패 0 · NO_SOLUTION / PASS_WITHIN_MODEL / SCREEN_ONLY | 기준값 19/19 · 독립 검산 26 | 이상 스위치 모델: ZVS는 NOT_EVALUABLE, 부호·전하는 SCREEN_ONLY; 손실은 후처리 추정 또는 결합 R만 | 미수행 (하드웨어·실측 없음) |
| FL09 | 12. LLC — 공진을 말로 설명하고 식으로 확인하기 [FL09]; 19. 교재–실습 계약과 통과 기준 (FHA 대 switching 5 %) | LLC — 공진을 말로 설명하고 식으로 확인하기 | fha_gain, switching_vs_fha, gain_curve, zvs_lm | `src/convlab/labs/fl09_llc.py` | `tests/test_fl09.py` (12) | 9 runs, 실패 0 · MISSING_INPUT / OUT_OF_VALIDITY / PASS_WITHIN_MODEL | 기준값 6/6 · 독립 검산 21 | FHA 회로는 switching 검증이 아니다; 이상 스위치: ZVS는 NOT_EVALUABLE, 전하는 SCREEN_ONLY | 미수행 (하드웨어·실측 없음) |
| FL10 | 13. CLLC — 설계 실패와 다중 해를 직접 방어하기 [FL10]; E05 · CLLC: gain 곡선에서 실제 동역학으로 | CLLC — 설계 실패와 다중 해를 직접 방어하기 | seed_fail, fix_n093, branch_selection, reverse, time_domain | `src/convlab/labs/fl10_cllc.py` | `tests/test_fl10.py` (10) | 7 runs, 실패 0 · CANDIDATE_FHA_ONLY / FAIL_CONSTRAINT / UNRESOLVED_RANKING | 기준값 14/14 · 독립 검산 19 | seed FAIL은 결과로 보존; FHA gain PASS ≠ 스위칭·ZVS·reverse·기동 PASS | 미수행 (하드웨어·실측 없음) |
| FL11 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| FL12 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| EX01 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| EX02 | E02. 비선형 Coss·dead time·ZVS [EX02]; 04. Si·SiC·GaN과 데이터시트 [FL02] | 비선형 Coss·dead time·ZVS — 에너지식 하나로 판정하지 않기 | qe_integrals, hb_constant_current, resonant_transition, deadtime_window | `src/convlab/labs/ex02_coss_zvs.py` | `tests/test_ex02.py` (11) | 8 runs, 실패 0 · FAIL_CONSTRAINT / PASS_WITHIN_MODEL / SCREEN_ONLY | 기준값 8/8 · 독립 검산 12 | charge screen은 SCREEN_ONLY — 실제 converter ZVS PASS 아님; 합성 C(v)의 D 수준 경향 결과이며 vendor 소자 모델이 아님 | 미수행 (하드웨어·실측 없음) |
| EX03 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| EX04 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| EX05 | E05 · CLLC: gain 곡선에서 실제 동역학으로; 13. CLLC [FL10] | CLLC — gain 곡선에서 실제 동역학으로 | state_identity, operating_points, floquet, gvf, closed_loop, startup_reverse, tolerance | `src/convlab/labs/ex05_cllc_dynamics.py` | `tests/test_ex05.py` (8) | 11 runs, 실패 0 · FAIL_CONSTRAINT / MARGINAL / OUT_OF_VALIDITY / PASS_WITHIN_MODEL / SCREEN_ONLY / UNSTABLE | 기준값 3/3 · 독립 검산 22 | 이상 소자·합성 출력 C; plant-only Floquet ≠ 폐루프 안정성 | 미수행 (하드웨어·실측 없음) |
| EX06 | E06 · DAB: RMS 최소화와 ZVS의 상충관계; 11. DAB [FL08] | DAB: RMS 최소화와 ZVS의 상충관계 | general_modulation, candidate_map, commutation_detail, implementation | `src/convlab/labs/ex06_dab_modulation.py` | `tests/test_ex06.py` (9) | 6 runs, 실패 0 · MISSING_INPUT / PASS_WITHIN_MODEL / SCREEN_ONLY | 기준값 18/18 · 독립 검산 12 | global optimum 아님 (격자 + 첫 φ 해); ZVS는 SCREEN_ONLY, 이상 전류 단계는 NOT_EVALUABLE | 미수행 (하드웨어·실측 없음) |
| EX07 | E07. 제어: 단일 루프가 안정해도 시스템은 불안정할 수 있다 [EX07]; 09. 제어 — PI보다 plant와 부호가 먼저다 [FL06] | 제어·입력 상호작용 — 단일 루프가 안정해도 시스템은 불안정할 수 있다 | cpl_exact, converter_impedance, digital_delay, saturation_transfer | `src/convlab/labs/ex07_system_stability.py` | `tests/test_ex07.py` (13) | 11 runs, 실패 0 · MARGINAL / NO_SOLUTION / PASS_WITHIN_MODEL / UNSTABLE | 기준값 29/29 · 독립 검산 33 | 이상 CPL 결과는 무한대역·국소(선형) 결과이며 큰 진폭 붕괴는 예측하지 않음; 컨버터 임피던스는 평균모델(B) 소신호 — 스위칭·샘플링 포함 실측 임피던스 아님 | 미수행 (하드웨어·실측 없음) |
| EX08 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| EX09 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| EX10 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| EX11 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
| EX12 | - | 미구현 | - | - | - | - | - | - | 미수행 (하드웨어·실측 없음) |
