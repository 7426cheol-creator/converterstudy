# 모델 카드 (Model cards)

생성: `tools/gen_docs.py` · run-all 2026-09-30T19:46:57

모델 수준: A 해석/FHA · B 평균 동역학 · C 이상 스위칭 · D 비이상 commutation(합성). 가정·적용 불가 범위는 각 실험의 기준 preset 실행 결과에서 그대로 가져왔다.


## FL01 · Buck와 Boost — 컨버터의 문법

범위: Buck/Boost 상태식·평균·스위칭·DCM; 전류 slope·리플·에너지·RHP 응답 (교재 19장 표)

주장 한계: 이상 스위치(C) 파형과 평균모델(B); 스위칭 손실·ZVS·링잉 미포함; RHP 응답은 저항부하·CCM·duty 입력 조건에서만; Boost 출력 C는 교재에 값이 없어 ASSUMED


### buck_ccm — Buck 48→12 V: 두 상태를 평균하고 스위칭 파형으로 확인

- 모델 수준: **C (+B 평균모델, 기동 과도)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 이상 스위치(즉시 전환, dead time 없음), 선형 L·C, 저항부하 R = D·V_in/I_o; 입력 V_in은 이상 전압원(소스 임피던스 없음); ESR은 C 가지에 직렬; DCR·R_DS(on)는 입력값(기본 0)
- 적용 불가: ZVS/스위칭 손실·dv/dt·링잉 (EX02, FL02); 코어 포화·온도 (FL07, FL03); 폐루프 응답 (FL06)
- 주장 한계: 이상 스위치 수준(C)의 파형·RMS·리플. 스위칭 손실·ZVS·링잉·포화는 주장하지 않는다.

### buck_light_load — 경부하: diode-emulation DCM vs 강제 동기정류의 음전류

- 모델 수준: **C (+A DCM 해석식)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 이상 스위치; diode-emulation은 i_L = 0에서 이상적으로 차단; 개루프 duty 고정 (폐루프 duty 조정 없음)
- 적용 불가: DCM 링잉·EMI; 경부하 효율의 정확한 값(스위칭 손실 미포함)
- 주장 한계: DCM 변환비와 전류 파형(C). 링잉·경부하 효율 정밀값은 주장하지 않는다.

### buck_ripple — 출력 리플: 커패시턴스 성분과 ESR 성분은 위상이 다르다

- 모델 수준: **C (+A 손계산)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: ESR은 주파수 무관 상수, ESL 없음; C는 이상 선형
- 적용 불가: ESL spike·고주파 리플 측정값; 커패시터 수명·발열
- 주장 한계: 상수 ESR·ESL 없음 가정의 리플 파형.

### boost_rhpz — Boost: duty를 늘렸는데 출력이 먼저 떨어진다 (RHP zero)

- 모델 수준: **A+B+C** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 출력 C = 100 µF는 교재에 없는 ASSUMED 값 (영점 위치에는 무관, 과도 모양에는 영향); 저항부하, 이상 다이오드, 개루프 duty step
- 적용 불가: 폐루프 안정성 판정 (FL06/EX07); DCM·정전력부하에서의 영점 위치
- 주장 한계: 개루프 duty step의 역응답과 이상 CCM 소신호 영점. 폐루프 안정성은 주장하지 않는다.

## FL05 · OBC·PFC — 전력품질을 설계 변수로 바꾸기

범위: 단상/3상 PFC·DC-link sizing; 16.450 A·10.699 kW·PF/THD 정의·energy (교재 19장 표); 3상 상/선간전압과 단상 2ω 커패시터 문제 분리; 강제 정현파 THD = 0을 PFC 성능으로 주장하지 않음

주장 한계: η·PF 고정의 해석 경계(A); THD/PF는 이상 소자·합성 제어기·정의된 창의 모델 결과 — 규격 적합성·EMI 판정 아님; 강제 정현파 모델의 THD = 0은 구성의 결과이며 PFC 성능이 아님; 스위칭 파형에 임의 사인 리플을 더한 값을 측정 THD처럼 보고하지 않음; DC-link 수명·ESR 발열은 데이터 필요 (MISSING_INPUT)


### grid_boundary — 11 kW와 16 A가 동시에 안 되는 조건: 입력 경계와 변조지수

- 모델 수준: **A (+ 시간영역 3상 전력 경로)** · 기준 실행 상태: `FAIL_CONSTRAINT`
- 가정: η와 PF는 운전점과 무관한 상수 (교재의 합성 조건); 3상 평형, 선전류 = 상전류 RMS; 계통 임피던스 전압강하 없음; 변조지수 코너의 L_g = 1 mH는 ASSUMED (단위 역률 기본파 전압강하만)
- 적용 불가: 효율·PF의 부하/전압 의존성; 과변조 영역의 고조파·이득; 단상 운전의 입력 조건 (별도 계산 필요)
- 주장 한계: η·PF 고정의 해석 경계(A). 실제 효율·PF 곡선과 과변조 특성은 주장하지 않는다.

### pf_thd — PF·변위역률·THD: 정의와 측정 창

- 모델 수준: **A (합성 파형, DFT)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 합성 파형: 기본파 + 3·5·7차 전류 고조파 (+ 5차 전압, DC); 샘플링은 이상 (양자화·앨리어싱 필터 없음)
- 적용 불가: IEC 61000-3-2/3-12 적합성 판정; 스위칭 주파수 대역의 전도 EMI; 실측 파형의 THD
- 주장 한계: 합성 파형의 정의 계산(A). 규격 적합성·실측 THD는 주장하지 않는다.

### boost_pfc — 단상 boost PFC 스위칭 모델: THD는 폐루프 전류에서 계산한다

- 모델 수준: **C (정확 스위칭) + B 평균모델 + A 강제 정현파** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 이상 다이오드 bridge·이상 스위치(즉시 전환), 부스트 다이오드 이상; 계통은 이상 정현 전압원 (V_rms 230 V, 50 Hz), 계통 임피던스·EMI 필터 없음; 부하는 저항 R = V_ref²/P = 48.48 Ω (엔진은 선형 회로), DCR 50 mΩ만 손실; 샘플링: carrier valley(OFF 구간 중앙)에서 i_L·v_C·v_g, duty는 다음 주기 적용
- 적용 불가: IEC 고조파 규격 적합성·EMI; 소자 손실·효율 정밀값; bridge 역회복·접합용량이 만드는 영교차 현상; 후단 DC/DC(정전력 부하)와의 상호작용 (EX07)
- 주장 한계: 이상 소자·합성 제어기의 스위칭 모델(C) THD/PF. 규격 적합성·EMI·소자 손실은 주장하지 않는다.

### dclink_power — 단상 2ω 전력과 3상 순시전력, 그리고 DC-link를 정하는 서로 다른 이유

- 모델 수준: **A (+ B 에너지 ODE)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 단상 예: 230 V RMS 단위 역률 정현파; 후단 전력 일정 (정전력 부하), PFC 입력 전력 P(1 − cos2ωt); hold-up 동안 PFC 공급 0, ESR·제어 응답 없음
- 적용 불가: 커패시터 수명·ESR 발열 (데이터 필요); 제어 루프가 요구하는 최소 용량; 3상 고조파·불평형 전압 조건의 정밀 리플
- 주장 한계: 이상 정현파·정전력 에너지 모델(A/B). 커패시터 수명·ESR 발열은 주장하지 않는다.

## FL06 · 제어 — PI보다 plant와 부호가 먼저다

범위: RL/current loop·PI 단위(V vs duty)·delay·saturation·anti-windup·step·dq 부호 (교재 19장 표); Kp 5.026548 V/A, Ki 628.318531 V/(A·s), 37.5 µs → 13.5°

주장 한계: 이상 평균 변조기의 선형·샘플링 루프 결과(A/B). PWM 리플·센서 필터·양자화 미포함; 위상여유·overshoot 기준은 학습용이며 고객 사양이 아님; dq 결과는 평형 계통·이상 PLL 가정; 성공 기준과 모델의 물리 한계를 섞지 않음


### pi_design — 단위를 가진 PI: V 출력 vs duty 출력, 그리고 지연이 먹는 위상

- 모델 수준: **A (+ 샘플링 루프 B)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: plant: L di/dt = u − R i − e, e는 피드포워드로 상쇄된다고 보고 루프 분석에서 제외; 변조기는 이상 평균 전압원 u = K_out·y (PWM 리플·dead time·전압 오차 없음); 샘플링 루프: 순방향 Euler PI, 연산 지연 1 샘플, ZOH (T_d 연속 근사 = 37.5 µs)
- 적용 불가: 포화·anti-windup 동작 (실험 2); 센서 필터·PWM 리플·식별하지 않은 공진이 있는 실제 루프의 위상여유; 입력필터·CPL과의 전체 시스템 안정성 (EX07)
- 주장 한계: 선형 루프 해석(A)과 이상 변조기의 샘플링 루프(B). 실측 루프이득·센서 필터·공진은 주장하지 않는다.

### discrete_loop — 샘플링 전류 루프: 포화·anti-windup·교재의 다섯 가지 사건

- 모델 수준: **B (정확 ZOH plant + 샘플링 PI)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: plant: L di/dt = u − R i − e, e는 샘플 경계에서만 바뀐다 (ZOH 정확 map); PI: 순방향 Euler, 연산 지연 1 샘플, |u| ≤ V_max 포화, back-calculation K_aw = 1·Ki/Kp; 이상 평균 변조기(리플·dead time 없음), 센서는 이상(필요 시 offset만)
- 적용 불가: PWM 리플·센서 필터·양자화; 하드웨어 OCP 동작; dq 벡터 포화·축 간 결합 (EX07); 외부 전압 루프와의 상호작용
- 주장 한계: 이상 평균 변조기의 샘플링 루프(B). PWM 리플·센서 필터·하드웨어 OCP는 주장하지 않는다.

### dq_sign — dq PFC 전류 제어: 부호가 맞아야 PI가 의미가 있다

- 모델 수준: **B (dq 평균모델 + 독립 abc 모델)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 평형 3상 계통, 이상 PLL (θ = ωt), 계통 전압 피드포워드; 평균 변조기가 샘플 동안 dq 전압을 유지 (연속 회전 각도); 상별 R·L 동일, 영상분 없음
- 적용 불가: 불평형·역상분·약계통·PLL 동특성; PWM 리플·dead time 전압 오차; DC-link 동특성 (FL05, EX07)
- 주장 한계: 평형 계통·이상 PLL·평균 변조기의 dq 전류 루프(B). 불평형·약계통·PWM 리플은 주장하지 않는다.

## FL08 · DAB — 식에서 파형, 파형에서 설계 판단으로

범위: DAB analytical·piecewise·switching; 1.5 kW·2.01988 Arms·mismatch (교재 19장); zero-DC 기준해, L_m 분리, 0/90/180° 포트 전류, ZVS NOT_EVALUABLE/SCREEN_ONLY

주장 한계: 이상 스위치 모델: ZVS는 NOT_EVALUABLE, 부호·전하는 SCREEN_ONLY; 손실은 후처리 추정 또는 결합 R만; 모듈당/합계를 분리해 표시


### sps_nominal — SPS 4구간: 닫힌 식·구간 적분·스위칭 해로 같은 숫자 얻기

- 모델 수준: **A + C** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 이상 full bridge 두 개(SPS, 50 % duty), 이상 변압기 n = N_p/N_s, 직렬 L은 1차 환산; dead time·노드 용량 없음; R, L_m은 입력했을 때만 포함
- 적용 불가: ZVS 판정 (NOT_EVALUABLE); 스위칭 손실·효율; 변압기 포화 (FL07)
- 주장 한계: 이상 스위치(C)·선형 L의 전력·RMS·peak. ZVS·손실 주장 없음.

### zero_power_mismatch — 0 W인데도 왜 뜨거운가: 순환전류와 여자전류

- 모델 수준: **A + C** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 이상 bridge, 선형 L·L_m; 손실은 R_est로 후처리한 추정치 (전력단 미결합)
- 적용 불가: 실제 무부하 소비전력 (스위칭·코어 손실 미포함)
- 주장 한계: 이상 bridge의 순환·여자전류. 무부하 손실 정밀값 아님.

### voltage_corners — 9개 voltage corner: 해가 있는 것과 통과하는 것은 다르다

- 모델 수준: **A (+C 교차검증)** · 기준 실행 상태: `SCREEN_ONLY`
- 가정: V_L 최대 54 V는 ASSUMED (교재는 36/48 V만 명시); 합성 C_oss (HV/LV)와 dead time은 ASSUMED
- 적용 불가: ZVS 확정·손실·온도
- 주장 한계: 해 존재·RMS·부호 screen (SCREEN_ONLY).

### offset_startup — DC offset·기동·비대칭 volt-second: 이상 L은 offset을 없애지 않는다

- 모델 수준: **C** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 0 초기조건 (precharge·soft-start 없음); 비대칭은 1차 하강 edge 지연으로 모델
- 적용 불가: 자속 포화·코어 손실; 실제 기동 시퀀스
- 주장 한계: 이상 스위치·선형 L의 offset 동역학. 포화 없음.

### two_modules — 2개 모듈 interleaving: 0/90/180°의 실제 DC 포트 전류와 L±10 % 분담

- 모델 수준: **C** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 모듈 L = L(1∓δ), δ = 0.1; 공통 φ 명령 (모듈별 전류 제어 없음)
- 적용 불가: 모듈별 폐루프 분담 제어; 출력 필터 설계
- 주장 한계: 이상 파형의 포트 전류 합. 출력 필터·제어 미포함.

## FL09 · LLC — 공진을 말로 설명하고 식으로 확인하기

범위: LLC FHA·rectifier switching; R_ac·HB/FB·gain/phase·FHA mismatch (교재 19장)

주장 한계: FHA 회로는 switching 검증이 아니다; 이상 스위치: ZVS는 NOT_EVALUABLE, 전하는 SCREEN_ONLY; 손실·온도 주장 없음


### fha_gain — FHA gain 곡선: Q·k·bridge 계수를 식으로 확인

- 모델 수준: **A** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 기본파 근사(FHA): 정류기 입력 전압 기본파와 전류가 동상; R_ac = 8n²R_dc/π² (full-bridge 정류, 연속 도통); 이상 소자·무손실 tank
- 적용 불가: 경부하·f_r 아래의 불연속 정류(off 구간); ZVS 판정; 동특성(G_vf)
- 주장 한계: FHA 대수. 정류 switching·ZVS·동특성 주장 없음.

### switching_vs_fha — 정류기가 있는 스위칭 회로 vs FHA: 어디서 맞고 어디서 틀리나

- 모델 수준: **C + A** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 이상 bridge (dead time 0); 이상 다이오드 full-bridge 정류, 권선·다이오드 손실 없음; 출력 C_o = 200 µF (ASSUMED), 부하 R_L은 Q에서 유도; 주기해 = shooting (초기 과도 제외)
- 적용 불가: ZVS 보증; 효율·온도; SR 타이밍; 기동 과도
- 주장 한계: 이상 스위칭·다이오드 정류의 정상상태. ZVS·손실·기동 주장 없음.

### gain_curve — 스위칭 gain 곡선 전체: FHA peak와 실제 peak는 다른 곳에 있다

- 모델 수준: **C + A** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 각 점: 물리 12주기 + Newton shooting 주기해; 이상 스위치·다이오드, 합성 tank
- 적용 불가: 실제 gain peak 위치(기생 용량·dead time 영향); 제어 설계(동특성 없음)
- 주장 한계: 이상 스위칭 주기해의 gain·모드. 손실·기생 용량 없음.

### zvs_lm — L_m tradeoff: 경부하 ZVS 전하 vs 순환전류

- 모델 수준: **C + SCREEN** · 기준 실행 상태: `MISSING_INPUT`
- 가정: 이상 스위칭 주기해에서 edge 전류를 읽고 전하 screen 적용; 합성 C_oss·t_d (ASSUMED)
- 적용 불가: ZVS 보증; 효율 최적 L_m
- 주장 한계: edge 전류 부호·전하 screen. ZVS 보증·손실 없음.

## FL10 · CLLC — 설계 실패와 다중 해를 직접 방어하기

범위: CLLC network·full switching·branch; 초기 FAIL·수정안 2개 root·한계 (교재 19장)

주장 한계: seed FAIL은 결과로 보존; FHA gain PASS ≠ 스위칭·ZVS·reverse·기동 PASS; 이상 소자·강한 전압원 모델


### seed_fail — 초기안 FAIL을 먼저 재현하고 저장한다

- 모델 수준: **A** · 기준 실행 상태: `FAIL_CONSTRAINT`
- 가정: FHA (기본파), full-bridge 양측, R_ac = 8n²V_bat²/(π²P); 합성 학습 사양 (교재 13장); 이상 소자·무손실
- 적용 불가: 스위칭 파형·ZVS; reverse; 기동
- 주장 한계: FHA gain 가능성. 스위칭·ZVS·기동·reverse 주장 없음.

### fix_n093 — 수정안 n = 0.93: 모든 해를 찾고 두 해의 차이를 표로

- 모델 수준: **A** · 기준 실행 상태: `UNRESOLVED_RANKING`
- 가정: FHA (기본파); n은 연속 설계변수 — 13:14 ≈ 0.9286 같은 정수 권선은 회로·공차·창 면적·손실을 다시 계산; 합성 학습 사양
- 적용 불가: 최종 transformer 선정; 스위칭 동작점 (→ ‘시간영역 검증’ 실험); 공차 corner
- 주장 한계: FHA 해와 FHA 전류. 스위칭 동작점 아님.

### branch_selection — branch 선택: 두 해는 fold에서 만나고 기울기가 0이 된다

- 모델 수준: **A** · 기준 실행 상태: `UNRESOLVED_RANKING`
- 가정: FHA 해를 V_bat별로 독립 계산; link 850 V, 11 kW 고정
- 적용 불가: 폐루프 안정성; 스위칭 동작점
- 주장 한계: FHA 해 구조. 폐루프·스위칭 주장 없음.

### reverse — reverse는 forward의 역수가 아니다: 회로를 뒤집어 다시 유도

- 모델 수준: **A** · 기준 실행 상태: `FAIL_CONSTRAINT`
- 가정: reverse: 배터리측 bridge 구동, link측 다이오드(또는 body diode) 정류; 같은 T-모델에서 L₁↔L₂′, C₁↔C₂′ 교환; FHA
- 적용 불가: reverse 스위칭 검증; SR 동작
- 주장 한계: reverse FHA 가능성. reverse 스위칭 검증 아님.

### time_domain — 시간영역 검증: FHA 해 주파수에서 실제로 얼마의 전력이 흐르나

- 모델 수준: **C + A** · 기준 실행 상태: `FAIL_CONSTRAINT`
- 가정: 이상 bridge·다이오드, 직렬 저항 R₁ = R₂′ = 0 mΩ (그 외 손실 없음); link·배터리 모두 강한 전압원 (내부저항·출력 C 없음); 주기해 = 물리 12주기 + Newton shooting (event 시각 민감도 포함)
- 적용 불가: 실제 소자 ZVS·손실; 배터리 내부저항·케이블이 있는 실제 전력 민감도; 제어 안정성
- 주장 한계: 이상 스위칭·강한 전압원 모델의 전력·RMS. 손실·ZVS 보증·기동 없음.

## EX02 · 비선형 Coss·dead time·ZVS — 에너지식 하나로 판정하지 않기

범위: Q/E 적분, deadtime·current sign·partial ZVS (E13 표); 286.606 ns vs 174.574 ns; 100/200/300/500 ns; 실제 i(t) 변화 경로는 회로·state를 추가해 별도 검증

주장 한계: charge screen은 SCREEN_ONLY — 실제 converter ZVS PASS 아님; 합성 C(v)의 D 수준 경향 결과이며 vendor 소자 모델이 아님; 2E_oss를 모든 ZVS topology의 보편 최소에너지로 쓰지 않음; 역회복·gate 동역학·링잉 미모델


### qe_integrals — Coss·Qoss·Eoss는 세 가지 다른 함수다

- 모델 수준: **A** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 합성 C(v) = C₀/√(1+v/V₀); 소자 1개의 출력 용량(드레인-소스)만
- 적용 불가: 실제 소자 데이터시트 곡선(MISSING_INPUT); 온도·전류 의존성
- 주장 한계: 합성 C(v)의 적분값. 실제 소자 곡선 아님.

### hb_constant_current — 고정 rail half-bridge: 2Q_oss와 dead time (charge screen)

- 모델 수준: **A (charge screen)** · 기준 실행 상태: `SCREEN_ONLY`
- 가정: 고정 rail(이상 전압원), 두 소자 동일 C(v); dead time 동안 주입전류 일정 (정전류 가정); gate 지연·채널 전류 감소 시간 0
- 적용 불가: 실제 공진 전류가 변하는 converter의 ZVS 판정 (실험 3 필요); full bridge의 여러 노드·transformer 용량이 있는 회로에 2Q 계수를 그대로 적용; 역회복 손실
- 주장 한계: 정전류 charge screen (SCREEN_ONLY).

### resonant_transition — 실제로 변하는 전류로 commutation event 풀기 (합성 D 수준)

- 모델 수준: **D (합성)** · 기준 실행 상태: `FAIL_CONSTRAINT`
- 가정: 고정 rail, 두 소자 동일 합성 C(v), 선형 인덕터, 인덕터 반대편 전압 v_x 일정; 채널은 gate-off 순간 전류를 즉시 끊음 (turn-off 손실·gate 동역학 없음); body diode = 이상 diode + V_f (역회복 없음); hard turn-on 손실은 용량 에너지 수지만 (V·I overlap 제외)
- 적용 불가: 특정 소자의 ZVS 보증·turn-on 손실 정밀값; 링잉·dv/dt·EMI; 역회복; 온도·lot 편차
- 주장 한계: 합성 C(v)·이상 diode·즉시 채널 차단의 D 수준 경향. 실제 소자 ZVS 보증 아님.

### deadtime_window — dead time은 한 점이 아니라 운전점별 창이다

- 모델 수준: **D (합성)** · 기준 실행 상태: `SCREEN_ONLY`
- 가정: v_x·L 고정, 합성 C(v); gate 지연 산포는 입력 하나로 더함 (분포 미모델)
- 적용 불가: 실제 소자·온도에서의 dead time 확정
- 주장 한계: 합성 event 격자 (SCREEN_ONLY).

## EX05 · CLLC — gain 곡선에서 실제 동역학으로

범위: E05: 상태식·항등식, 두 root의 스위칭 운전점·G_vf, 폐루프 step, startup/reverse 실패 사례, seed 실패 보존

주장 한계: 이상 소자·합성 출력 C; plant-only Floquet ≠ 폐루프 안정성; ZVS/overshoot 수치는 합성 모델에만 유효


### state_identity — CLLC 상태식과 에너지 항등식: 환산과 부호를 먼저 검산

- 모델 수준: **C** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 1차 환산 T-모델 (교재 E05) + 출력 C_o·R_L; 이상 다이오드 3-상태 정류기 (off = i₂ = 0 제약, 떠 있는 노드)
- 적용 불가: SR 채널/바디다이오드 구분; 권선 간 용량·누설 분포
- 주장 한계: 모델 구조 검산. 설계 통과 주장 없음.

### operating_points — FHA 두 해 vs 스위칭 운전점: 기울기 부호가 같은가

- 모델 수준: **C + A** · 기준 실행 상태: `OUT_OF_VALIDITY`
- 가정: 출력 C_o = 100 µF, R_L = V²/P (전압 루프용 부하); 환산 대칭 CLLC (FL10 수정안); 이상 소자
- 적용 불가: 배터리(강한 전압원) 부하 — FL10 ‘시간영역 검증’; 실제 소자 ZVS·손실
- 주장 한계: 이상 스위칭 주기해. 손실·ZVS 없음.

### floquet — 주기해와 안정성은 따로: Floquet multiplier와 초기값 섭동

- 모델 수준: **C** · 기준 실행 상태: `MARGINAL`
- 가정: 외부 clock(고정 주파수)으로 구동되는 sampled map; FD 섭동 1e-6 × 상태 scale, event 재탐색 포함
- 적용 불가: 폐루프 안정성; 대신호(기동·포화)
- 주장 한계: plant 국소 수렴. 폐루프·대신호 주장 없음.

### gvf — branch별 G_vf: 정상 기울기는 DC 한 점일 뿐

- 모델 수준: **C** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: cycle-to-cycle map: 주기 시작 샘플 v_o, 주기마다 한 번 바뀌는 주파수 명령; FM 주입 진폭 ±0.002 % (소신호)
- 적용 불가: f_s/2 이상의 변조; 대신호 과도
- 주장 한계: event 포함 소신호 모델 (이상 소자). 실제 보드 G_vf 아님.

### closed_loop — 폐루프: 지연·포화 포함 PI, 부호가 맞아도 불안정할 수 있다

- 모델 수준: **C** · 기준 실행 상태: `MARGINAL`
- 가정: PI: f = f₀ + s·(K_p e + ∫K_i e dt), s = 제어기가 가정한 기울기 부호; 주기 시작에서 v_o 샘플, 다음 주기에 적용(1주기 계산 지연); 조건부 적분 anti-windup, 주파수 포화 120–210 kHz
- 적용 불가: 센서·ADC 필터; 전류 루프·charge control; 실제 배터리 충전 모드
- 주장 한계: 이상 스위칭 plant + 이산 PI. 실제 제어기 구현·센서 없음.

### startup_reverse — 기동과 reverse: 빈 출력 C, 연결된 배터리, 저전압 역방향

- 모델 수준: **C** · 기준 실행 상태: `FAIL_CONSTRAINT`
- 가정: 기동 A: 출력 C 0 V·tank 0 상태에서 주파수 램프 (개루프); 기동 B: 강한 배터리 연결, 개루프 하강 sweep; reverse: 배터리측 bridge 구동, link 강한 전압원, tank 교환
- 적용 불가: precharge 회로; burst 기동; SR 역전류
- 주장 한계: 개루프 기동·reverse의 이상 스위칭 과도. 보호 회로 없음.

### tolerance — 공차: f_r만 움직이는 것이 아니다

- 모델 수준: **A + C** · 기준 실행 상태: `SCREEN_ONLY`
- 가정: L₁·L₂′, C₁·C₂′가 같이 움직인다 (환산 대칭 유지); 명목 운전점 주파수 고정 — 제어가 보정하기 전의 편차
- 적용 불가: 통계적 수율; 온도 drift
- 주장 한계: 합성 공차 screen. 수율·온도 drift 없음.

## EX06 · DAB: RMS 최소화와 ZVS의 상충관계

범위: w₁, w₂, φ switching function; 정확 PWL P/Irms/Ipk; SPS 닫힌 식 독립 검증; 3.113563 → 2.893092 A; commutation sign/charge·zero-state·최소 펄스·dead time·timer; 전환 transient·offset

주장 한계: global optimum 아님 (격자 + 첫 φ 해); ZVS는 SCREEN_ONLY, 이상 전류 단계는 NOT_EVALUABLE; RMS 감소 ≠ 효율 이득; 손실 단계는 MISSING_INPUT (합성 손실은 설명용)


### general_modulation — switching function을 고정하고 정확 적분: SPS vs width candidate

- 모델 수준: **C + A** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 이상 bridge, 선형 L, zero-mean(반주기 반대칭) 전류 해; φ는 가장 작은 양의 해 (다른 해 가지는 따로 비교해야 함)
- 적용 불가: ZVS 판정; 효율 개선 수치; global optimum 주장
- 주장 한계: 이상 전류 파형의 RMS/peak. ZVS·효율 개선 주장 없음.

### candidate_map — candidate map과 단계별 권장점: 이상 전류 → commutation → 손실

- 모델 수준: **C + SCREEN** · 기준 실행 상태: `MISSING_INPUT`
- 가정: 격자 탐색(w 간격 입력값)과 각 셀의 가장 작은 φ 해; 합성 C_oss·dead time·최소 펄스·peak 한계 (ASSUMED)
- 적용 불가: global optimum; 효율·온도 개선 주장; 실제 소자 ZVS 보증
- 주장 한계: 격자 탐색 + SCREEN_ONLY. 손실 단계는 MISSING_INPUT (합성은 설명용).

### commutation_detail — switch node별 commutation: zero-state 선택이 전환 전류를 바꾼다

- 모델 수준: **C + SCREEN** · 기준 실행 상태: `SCREEN_ONLY`
- 가정: edge 전류 × dead time 전하 근사; 고정 rail half-bridge 2Q_oss 요구전하 (EX02 가정); 합성 C_oss (HV/LV 별도)
- 적용 불가: ZVS 보증; dead time 동안 전류 변화가 큰 경우
- 주장 한계: SCREEN_ONLY

### implementation — FPGA/MCU 구현: timer 분해능·최소 펄스·모드 전환 offset

- 모델 수준: **A + C** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 이상 1-step 감도 (dithering·폐루프 없음); 전환은 bridge 전압 패턴의 즉시 교체; R = 0 Ω
- 적용 불가: 실제 출력 리플 예측; scheduler 안정성 증명
- 주장 한계: 이상 1-step 감도와 이상 bridge 전환 offset.

## EX07 · 제어·입력 상호작용 — 단일 루프가 안정해도 시스템은 불안정할 수 있다

범위: CPL exact example(394.935887 V, 320.565519 µF, 100 µF 불안정·1 mF 안정)을 비선형 적분·local Jacobian으로 독립 확인; 유한 대역 converter 입력 임피던스 비교; 15 µs 지연 1 kHz 5.4°·10 kHz 54°; delay/saturation/antiwindup 실험; 최소 3개 전압/부하 코너 주파수응답; current-loop margin과 whole-system 안정성 각각 보고 (E07·E13 표)

주장 한계: 이상 CPL 결과는 무한대역·국소(선형) 결과이며 큰 진폭 붕괴는 예측하지 않음; 컨버터 임피던스는 평균모델(B) 소신호 — 스위칭·샘플링 포함 실측 임피던스 아님; |Z_s| < |Z_in|은 보수적 충분조건으로만 사용; 스위칭 파형에 임의 사인 리플을 더해 측정 THD처럼 보고하지 않음 (THD는 FL05의 시뮬레이션 전류로만); 루프 여유·overshoot 기준은 학습용이며 고객 사양이 아님


### cpl_exact — 소스 R–L + C + CPL: 교재 예제를 세 경로로 재현

- 모델 수준: **A (해석 극점) + 비선형 ODE + 수치 Jacobian** · 기준 실행 상태: `UNSTABLE`
- 가정: 이상 CPL i = P/v (제어 대역 무한), 선형 R·L·C, 소스 V_s 일정; 교란 1 V (교재 그림과 같은 크기), 비선형 적분은 |Δv| ≤ 20 % V_e에서만
- 적용 불가: 실제 컨버터의 유한 대역 입력 임피던스 (실험 2); 큰 진폭의 전압 붕괴·UVLO·전류 제한; 스위칭 리플
- 주장 한계: 이상 무한대역 CPL의 국소 선형 결과와 유효범위 안의 비선형 적분. 큰 진폭 붕괴·보호 동작은 주장하지 않는다.

### converter_impedance — 실제 converter는 유한 대역 CPL이다: Z_s/Z_in, 코너, 단일 루프 vs 전체 시스템

- 모델 수준: **B (평균 컨버터 소신호 + 비선형 시간영역)** · 기준 실행 상태: `UNSTABLE`
- 가정: 평균 buck(B), 무손실, 이상 변조기 d = u/V_ff, 스위칭 리플 없음; 제어: 전류 PI (f_ci 5000 Hz, 영점 f_ci/5) + 출력전압 피드포워드, 전압 PI (f_bw 200 Hz, 영점 f_bw/5); 입력전압 피드포워드 없음 (d = u/400 V); 필터: 소스 R–L + C
- 적용 불가: 샘플링 지연·PWM의 영향 (실험 3); duty·전류 포화 이후의 큰 신호 동작; 스위칭 주파수 부근의 임피던스
- 주장 한계: 평균모델(B)의 소신호 임피던스와 비선형 시간영역. 스위칭·샘플링·포화를 포함한 실측 임피던스는 주장하지 않는다.

### digital_delay — 디지털 지연: 샘플·연산·갱신 순서가 위상을 정한다

- 모델 수준: **B (샘플링 순서의 정확 이산 모델)** · 기준 실행 상태: `MARGINAL`
- 가정: 샘플은 캐리어 골(또는 정점)에서, 연산 시간 일정 (jitter 없음); 극점 소거 PI (Kp = Lω_c, Ki = Rω_c), 이상 평균 변조기; plant: FL06의 RL (0.8 mH, 0.1 Ω)
- 적용 불가: 연산 시간 변동·인터럽트 지연 분포; PWM 비교기·dead time의 전압 오차; 입력필터·CPL과 결합된 전체 시스템 (실험 2)
- 주장 한계: 샘플링 순서의 정확 이산 모델(B). jitter·하드웨어 PWM 세부는 주장하지 않는다.

### saturation_transfer — dq 벡터 포화·적용 벡터 anti-windup·bumpless 복귀

- 모델 수준: **B (샘플링 dq 루프 + 1축 루프)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: FL06 dq plant (0.8 mH, 0.1 Ω), 평형 계통·이상 PLL, 1 샘플 연산 지연; 회생 + 무효전류 step i_d -30 A, i_q 40 A; V_dc 600 V; PI 영점 = plant 극점; bumpless 실험: e 피드포워드 없는 1축 루프, sag −10 % 동안 PI 정지
- 적용 불가: SVPWM 육각형·과변조 영역; 보호 동작의 하드웨어 타이밍; 입력필터·CPL과의 결합 (실험 2)
- 주장 한계: 평형 계통·이상 PLL의 샘플링 dq 루프(B). 과변조·보호 하드웨어 타이밍은 주장하지 않는다.
