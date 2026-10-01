# 모델 카드 (Model cards)

생성: `tools/gen_docs.py` · run-all 2026-09-30T23:59:39

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

## FL02 · Si·SiC·GaN과 데이터시트, Gate drive·DPT·보호

범위: loss 비교·gate/DPT 제한된 commutation; 조건표·E 적분·L·di/dt·가설 분리 (교재 19장 표). 14/20.5 W, 6/6.1 W, 75 V, 30 A, 10 ns, 1.8 µs

주장 한계: 합성 A/B는 비교 논리 훈련 — 실제 부품 데이터셋은 별도(MISSING_INPUT), 제품 순위 아님; loss 비교는 postprocessed loss estimate (전력단에 결합된 손실 아님); DPT 셀은 합성 D 수준 경향 — vendor 소자 모델이 아니며 역회복·온도 없음; 이상 스위치로 gate ringing을 만들었다고 주장하지 않음 (ringing은 셀의 L·C에서만); 단순 모델로 SC survival을 보장하지 않음


### loss_ab — 합성 A/B 손실 비교: 낮은 R_DS(on)이 더 뜨거운 이유

- 모델 수준: **A (postprocessed loss estimate)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: R은 실제 운전온도의 값, E는 한 운전점(전류·전압·온도·R_g)의 사건당 E_on+E_off (교재 합성 소자); 주기마다 on/off 1쌍, 스위치 RMS 10 A — 파형 모양과 무관하게 RMS로 전도손실 계산; 병렬 sweep: 전도 ∝ 1/N, E의 k_cap = 0.3 비율(전하·용량 성분) ∝ N, 나머지는 N과 무관, 게이트 전하 ∝ N (합성 가정); 게이트 구동 전력은 소자 손실에 더하지 않고 별도 표시
- 적용 불가: 실제 부품의 손실 순위·선정; E(I, T)·R_DS(on)(T)가 변하는 운전영역 전체 (FL03, EX01); 병렬 소자의 전류 불균형·발진 (EX03)
- 주장 한계: 합성 A/B의 postprocessed 손실 비교(A). 실제 부품·제품 순위를 주장하지 않는다.

### gate_protect — Gate·Miller·loop 전압과 보호 타임라인 (1차 크기)

- 모델 수준: **A (1차 screen) + B (gate-charge ODE)** · 기준 실행 상태: `MISSING_INPUT`
- 가정: di/dt·dv/dt 일정 (1차 크기); gate-charge 곡선은 합성 piecewise 모델 (C₁ 충전 → Q_gd plateau → C₂ 충전); plateau 전류 = (V_drv − V_pl)/(R_drv + R_g + R_g,int); driver droop·gate loop L 없음; SC 생존시간 3 µs는 교재의 조건부 가정
- 적용 불가: 실제 overshoot peak·ringing (실험 3의 합성 셀, EX09); 실제 SC 생존·보호 인증; EMC 적합성
- 주장 한계: 1차 크기 screen과 합성 보호 타임라인. SC survival·EMC 적합성을 주장하지 않는다.

### dpt_cell — 합성 DPT 셀: Miller·L·di/dt·링잉과 E_on/E_off 적분의 정의

- 모델 수준: **D (합성 commutation cell)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 합성 SiC형 MOSFET: V_th·g_m 채널(tanh 선형영역), C_gs 일정, C_gd(v)·C_ds(v) = C₀/√(1+v/V₀), 고정 T_j, 역회복 없음; power loop: L_a 8 + L_b 3 (‖R_p 3 Ω) + L_s 2 + L_sH 2 nH, R_loop 2 mΩ; 부하 인덕터는 사건 동안 일정 전류원 I_L (두 번째 펄스 전 전류 증가 무시); 드라이버: 이상 전압원 + 2 ns 선형 slew, on/off 경로별 R_g, gate loop L_g; 상측 gate는 V_off,H로 유지; stiff solver(LSODA) + 이벤트 구간 분할; 에너지는 적분상태로 solver와 같은 정확도로 적분
- 적용 불가: 특정 vendor 소자의 E_on/E_off·overshoot 예측; 단락(SC) 생존·보호 인증; 역회복이 큰 Si diode 조합; EMI 스펙트럼 (EX09)
- 주장 한계: 합성 소자·회로의 D 수준 경향. vendor 소자 E·overshoot 예측, SC survival, EMI를 주장하지 않는다.

### vgs_spike — 블라인드 문제: V_GS spike — 실제 Miller turn-on인가, 측정 기준 오차인가

- 모델 수준: **D (합성 commutation cell)** · 기준 실행 상태: `UNRESOLVED_ROOT_CAUSE`
- 가정: 실험 3과 같은 합성 셀; DUT turn-on 한 사건 (상측은 body diode로 I_L 환류 중에서 시작); 상측 gate: V_off,H를 R_gH(+R_g,int)로 유지; Miller clamp는 off 동안 늘 켜진 저임피던스 경로(R_clamp)로 단순화; Kelvin 핀 = die source (내부 source 인덕턴스 없음)
- 적용 불가: 실제 소자의 false turn-on 여유 (온도별 V_th, C_rss/C_iss 데이터 필요); probe CM rejection·대역 오차; gate 산화막 신뢰성 판정
- 주장 한계: 합성 셀의 판별 논리(D). 실제 소자의 false turn-on 여유·산화막 신뢰성을 주장하지 않는다.

## FL03 · 손실·온도·수명 — 숫자가 서로 맞아야 한다

범위: thermal RC·loss/T 반복 (교재 19장 표): 77.642 °C 기준·전열 수렴·경계; coolant 65→85 °C·30 s 과부하; loss map 범위; Foster/Cauer

주장 한계: 단일 노드·합성 사다리의 열 모델 — 실제 모듈 열망·계면 값이 아님; 전열 발산(NO_STABLE_FIXED_POINT)은 단순 모델의 결과이며 실제 파손온도 예측이 아님; Foster 내부 노드를 물리 계면으로 연결하지 않음; loss map 결과는 postprocessed loss estimate, 범위 밖은 외삽하지 않음; 수명(년)을 계산하지 않음 (EX08: 상대 proxy만)


### rc_step — 100 W 계단: 10 s = 77.642 °C, 최종 85 °C — 과도와 정상상태

- 모델 수준: **A (해석해) + C (정확 선형 적분)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 단일 열노드 (R_th, C_th = τ/R_th), 경계 온도 T_b 일정; 손실 P는 온도와 무관한 계단 (전열 연성은 실험 2); 펄스 비교: 같은 평균손실을 duty로 인가
- 적용 불가: 여러 층의 열 경로 (실험 4); 인접 소자 열 결합 (EX08); 수명 예측
- 주장 한계: 단일 열노드의 과도/정상상태. 다층 열 경로·수명은 주장하지 않는다.

### electrothermal — R_DS(on)(T) 되먹임: 수렴·발산·과부하 (전열 반복)

- 모델 수준: **A (고정점) + C (정확 선형 과도)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 단일 열노드, 전체 손실에 선형 온도계수: P = P₀[1 + α(T_j − T_ref)] (합성); 손실모델 적용범위 T_j ≤ 175 °C (ASSUMED) — 넘으면 미지원; 냉각수 온도는 계단 변화, 과부하는 손실 배율 k로 인가
- 적용 불가: 실제 소자 파손온도·열폭주 판정; 적용범위 밖 온도의 손실; 다중 소자 열 결합 (EX08)
- 주장 한계: 단일 열노드·선형 온도계수의 전열 고정점과 과도. 실제 파손온도·열폭주 판정이 아니다.

### loss_map — 손실 포함 경계와 loss map 범위: 채널·dead time·E_sw를 사건별로

- 모델 수준: **A (postprocessed loss estimate)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 상측 소자, 이상 정현 전류 i = I_pk sin(ωt), SPWM duty 0.5(1 + m sin(ωt + φ)), 전류 리플 무시; dead time은 켜지는 edge에서 on 시간을 줄임; i < 0 구간의 상측 channel은 soft switching; 합성 loss map (격자 bilinear 보간, 범위 밖 미지원); 전열 동작점: 단일 R_th, 정상상태
- 적용 불가: 실제 부품 손실·효율 보증; map 범위 밖 전류·온도; 열 과도·mission (실험 2, EX08)
- 주장 한계: 합성 loss map의 postprocessed 손실 추정. 실제 부품 손실·효율을 주장하지 않는다.

### foster_cauer — Foster vs Cauer: 내부 노드는 물리 계면이 아니다

- 모델 수준: **A (Z_th 표현) + C (정확 선형 과도)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: junction→case 4층 합성 Cauer 사다리 (ASSUMED 층값); TIM은 열용량 없는 R, 방열판은 한 노드 R·C, 냉각 경계 T_b 일정; 데이터시트 Z_jc는 case 온도 고정 경계에서 정의
- 적용 불가: 실제 모듈 층 구조·계면 열저항; 측정 Z_th의 경계 조건이 다른 경우; 다중 소자 cross heating (EX08)
- 주장 한계: 합성 사다리의 표현·연결 방식 비교. 실제 모듈 층 구조·계면 값을 주장하지 않는다.

## FL04 · 인버터 — 알고 있는 것을 더 날카롭게 증명하기

범위: dq solver·loss 경계·짧은 inverter switching; 438.545 V·전압여유·출력경계 (교재 19장 표); 760 V sag에서 같은 점 불가

주장 한계: 선형 L_d·L_q·ψ_m (포화·교차결합·온도 의존 없음); 반도체 손실은 채널 전도분만; diode·dead time·스위칭 손실은 별도; 스위칭 모델은 이상 스위치·개루프 전압 인가 (C 수준); 모터 파라미터·공차·전류 한계는 합성 학습 값


### dq_point — 250 kW 운전점: 전압여유 0.241 V와 shaft·AC·DC 경계

- 모델 수준: **A (+ 독립 abc 재구성)** · 기준 실행 상태: `MARGINAL`
- 가정: 선형 L_d·L_q (포화·교차결합 없음), 정현 역기전력, 정상상태 기본파만; 철손·기계손실 제외 (교재 기준점), 고정자 저항은 온도 미지정 15 mΩ; 가용 전압 = V_dc/√3 × 0.95 (SVPWM 선형 범위 + 제어 여유); 강건 판정 요구 여유 2 %와 공차 크기는 학습용 가정(ASSUMED)
- 적용 불가: 포화·온도에 따른 L_d/L_q/ψ_m 변화 (선형 모델 밖); 전류제어 과도·dead time 전압오차; 인버터 효율·차량 주행거리 (반도체 손실 일부만 포함)
- 주장 한계: 선형 dq 정상상태(A). 포화·철손·전류제어 과도·반도체 손실 전체는 주장하지 않는다.

### dq_plane — i_d–i_q 평면: 토크 곡선·전압 한계·전류 원·MTPA

- 모델 수준: **A** · 기준 실행 상태: `MARGINAL`
- 가정: 선형 L_d·L_q·ψ_m, 정상상태, R_s 포함 전압식; 전류 한계 550 A는 합성 가정(ASSUMED); 가용 전압 = V_dc/√3 × 여유계수 (추가 요구 여유 없음)
- 적용 불가: 포화·교차결합이 있는 실제 모터의 MTPA/약계자 표; 열 한계·연속/순간 정격 구분; 과변조 영역
- 주장 한계: 선형 정상상태 평면(A). 포화·열 한계·과변조는 주장하지 않는다.

### inverter_switching — 짧은 인버터 스위칭: PWM이 만든 평균 전압·리플·DC-link 전류

- 모델 수준: **C (이상 스위치)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 이상 스위치 (즉시 전환, dead time·전압강하·스위칭 손실 없음), 이상 DC 전압원; 선형 dq 모터 (포화·철손 없음), 일정 속도 (기계 동역학 없음); 해석 운전점의 dq 전압을 개루프로 인가 (전류제어기 없음), 동기 PWM (f_sw = N·f_e)
- 적용 불가: 반도체 손실·온도 (FL02/FL03); dead time 전압오차·저속 왜곡; EMI·고주파 공진; 폐루프 전류제어 성능 (FL06)
- 주장 한계: 이상 스위치(C) 파형·평균·리플. dead time·스위칭 손실·EMI·폐루프 성능은 주장하지 않는다.

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

## FL07 · 자성체 — 컨버터 전문가로 가는 실제 관문

범위: flux integrator·L_m/leakage·손실 screen; winding B·0.20 T·pulse bias (교재 19장 표); 0.0064 T/주기, 0.72 µH, 0.209 mm

주장 한계: 선형 코어 (포화·히스테리시스 없음); Steinmetz 계수는 SYNTHETIC — 정밀 core loss는 MISSING_INPUT; Dowell 1-D 근사·고립 원형선 screen; B_sat·L_m·R·C_b는 합성 가정


### winding_flux — 코어는 버스 전압이 아니라 권선 전압을 적분한다 (DAB T-모델)

- 모델 수준: **C (정확 스위칭)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 선형 코어 (L_m 일정, 포화·히스테리시스 없음), 이상 bridge (dead time·Coss 없음); 직렬 L 분배 k = 1, L_m·R은 합성 가정; SPS 위상은 무손실·L_m 없는 폐형식으로 정함
- 적용 불가: 코어 손실 정밀값 (MISSING_INPUT, 실험 3); 포화 판정 (재료 자료 필요); ZVS 판정 (NOT_EVALUABLE)
- 주장 한계: 선형 코어·이상 bridge의 자속·전류 파형 (C). 포화·코어 손실은 주장하지 않는다.

### flux_walk — 펄스 비대칭: flux walking, 저항·blocking C, 그리고 측정 오차와 구분하기

- 모델 수준: **C (정확 스위칭)** · 기준 실행 상태: `FAIL_CONSTRAINT`
- 가정: 선형 L_m (포화 없음), 이상 bridge, 비대칭은 매 주기 같은 Δt; B_sat·R·C_b·probe offset·표본화율은 학습용 가정
- 적용 불가: 포화 이후 전류 (포화 모델 없음); 능동 flux balance 제어 성능; 실측 probe 오차 보정
- 주장 한계: 선형 코어의 자속 이동과 측정 오차 신호 (C). 포화 후 전류·제어 성능은 주장하지 않는다.

### loss_screen — L_m·누설·환산과 손실 screen: 합성 계수로는 정밀 core loss를 만들지 않는다

- 모델 수준: **A (+ C 환산 검증)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: Steinmetz k·α·β·V_e는 SYNTHETIC (특정 재료 fit 아님), 온도 의존 없음; Dowell 1-D 평판 근사, 원형선은 고립 도선; 환산 검증 회로의 직렬 R_s·C_s는 합성 값
- 적용 불가: 특정 코어·재료의 손실 예측; minor loop·DC bias·온도에 따른 손실; litz·foil 권선 최적화 확정
- 주장 한계: 환산 규칙·skin/Dowell·합성 core-loss screen (A). 정밀 core loss·R_ac는 MISSING_INPUT.

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

## FL11 · PSFB·HV-LV·12 V 확장 — 선택의 이유를 설명하기

범위: PSFB waveforms·SR·duty loss; D_eff·LV RMS·DAB 비교 (교재 19장 표); 12 V 대전류 손실·배선 (지침 5절)

주장 한계: PSFB는 이상 스위치(C) 모델: ZVS는 NOT_EVALUABLE, 전환 전류 표는 SCREEN_ONLY; 자화전류·2차 링잉·SR 타이밍 오차·스위칭 손실은 계산하지 않음; n = 12·V_in 800 V·D_eff 0.72의 48 V는 합성 seed; L_k·L_o·C_o·경로 저항은 ASSUMED; DAB 비교는 교재 11장 모듈의 닫힌 식·PWL (FL08의 전체 실험 대체 아님); 12 V budget은 합성 부품값의 A 수준; 특정 소자·layout 손실 아님


### psfb_duty_loss — PSFB: 명령 phase가 그대로 출력 duty가 되지 않는다 (duty loss와 SR 전류)

- 모델 수준: **C (+A 손계산·PWL)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 이상 스위치(즉시 전환, dead time 없음), 이상 transformer(자화전류 없음, L_m → ∞), 누설은 1차 환산 L_k 하나; 정류: SR은 diode-emulation 이상 타이밍(전류 0에서 차단) + R_DS(on); 다이오드는 V_f + R; 부하는 저항 R = V_nom²/P, 출력 C는 ESR 없음; 입력은 이상 전압원; 손실은 상태방정식 안의 저항·V_f만 (스위칭·Coss·gate·core 손실 없음)
- 적용 불가: ZVS 판정 (NOT_EVALUABLE; 표는 SCREEN_ONLY); 2차 정류 소자 링잉·overshoot·역회복; SR 타이밍 오차·역전류; 자화전류가 있는 1차 전류 파형; 폐루프 동특성 (regulate는 느린 루프의 정상점만)
- 주장 한계: 이상 스위치 수준(C)의 duty loss·SR 전류·RMS. ZVS·링잉·스위칭 손실·자화전류는 주장하지 않는다.

### lk_tradeoff — L_k를 키우면 lagging leg ZVS 범위가 넓어지지만 무엇을 내주나

- 모델 수준: **C + A (screen)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 정격 부하 저항 고정, V_o는 느린 루프가 48 V로 유지(regulate); lagging leg screen: ½L_k i² ≥ ½C_node V_in² (선형 C, 자화전류 없음); 저전압 corner V_in = 700 V, φ 상한 170° (ASSUMED)
- 적용 불가: ZVS 보증·turn-on 손실; 정수 권수 transformer 설계; 경부하 DCM에서의 ZVS
- 주장 한계: duty loss·전류는 C, ZVS는 선형 C_node screen (SCREEN_ONLY). ZVS 보증 아님.

### psfb_vs_dab — 같은 800→48 V·1.5 kW: PSFB와 DAB는 무엇이 다른가

- 모델 수준: **C (PSFB) + A/PWL (DAB)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: PSFB: 무손실 이상 스위치, 느린 루프가 48 V 유지; DAB: 교재 11장 모듈 (n = 50/3, L = 200 µH 1차 환산), SPS, zero-DC 반대칭 기준해; 두 converter 모두 자화전류 없음
- 적용 불가: 효율·손실 순위 (소자·자성체 손실 모델 없음); ZVS 판정 (둘 다 NOT_EVALUABLE; screen만); DAB의 다른 전압 corner (FL08)
- 주장 한계: 같은 전력의 전류·전압 파형 비교. 효율 순위·ZVS 판정은 주장하지 않는다.

### lv12_extension — 12 V 확장: 전류 4배, 같은 저항 손실 16배 — 소자만 병렬로 늘리면 되나

- 모델 수준: **A (budget) + C (검산)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 모든 경로 저항·Q_g·E_oss·분산은 ASSUMED 합성값 (특정 부품 아님); budget: 위치당 전류는 반주기 도통 I_o, 리플·commutation 중복 무시; 권선 AC 저항은 상수 계수 F_R (정밀값은 MISSING_INPUT: 권선 구조·주파수 필요); 3000 W는 총합 (모듈 분할 없음)
- 적용 불가: 특정 MOSFET·busbar 설계의 손실 정밀값; 병렬 소자 동적 분담·layout (EX03); SR 소자 온도·수명 (FL03/EX08); current doubler 정류 (모델 없음)
- 주장 한계: 합성 저항·전하값의 A 수준 budget + 결합 스위칭 검산. 특정 부품·layout의 손실·온도는 주장하지 않는다.

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

## EX03 · 병렬 SiC·gate loop·DPT 계측

범위: static sharing, deskew sensitivity, dynamic model 범위 (E13 표); 110.553/99.497/99.497/90.452 A; 533.333 µJ, +41.99 %/−33.01 %; L_s·di/dt 4 V, 10 ns × 2 kA/µs = 20 A

주장 한계: 정적 분담은 저항 모델 — 동적 분담과 따로 검증; DPT 민감도는 교재 합성 선형 파형 — 실제 probe·scope 오차 아님; 동적 분담은 실제 상태·inductance·gate 결합을 가진 bounded synthetic equivalent (D) — vendor fidelity 아님; 식별 불가 상호결합은 범위로만 보고; branch 전류 예측·제조 수율을 주장하지 않음


### static_sharing — 정적 4-branch 분담: 평균 100 A가 가리는 110.55 A

- 모델 수준: **A (정적 저항 + 전열)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 공통 단자전압의 정적 저항 모델 (inductance·gate 동역학 없음); R_k(T) = R_k[1 + α(T_j − T_c)], α = 0.004/K (합성), branch별 독립 R_th (branch 간 열 결합 없음)
- 적용 불가: turn-on/off 동적 분담·peak 전류 (실험 3); 실제 소자 분산·수율 추정; 모듈 내부 열 결합 (EX08)
- 주장 한계: 정적 저항·branch별 열 경로의 분담. 동적 분담·수율 추정은 주장하지 않는다.

### dpt_deskew — DPT 계측: 5 ns 어긋남이 E를 −33 %~+42 % 바꾼다

- 모델 수준: **A (합성 선형 overlap) + D (FL02 셀 참조면)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 교재 합성 turn-on: 40 ns 동안 v 선형 800→0 V, i 선형 0→100 A (실제 SiC 파형 아님); 넓은 고정창 [−10 ns, t_r + 10 ns]: 앞당긴 전류의 전이 전 구간 포함; probe = 1차 저역통과 (지연 τ = 1/2πf); 참조면 표는 FL02 합성 셀(D) 결과
- 적용 불가: 특정 probe·scope 조합의 실제 오차 (교정 필요); 실제 소자 E 순위; 링잉이 큰 파형의 창 정의 (창 길이를 따로 검토)
- 주장 한계: 합성 선형 파형의 측정 민감도와 합성 셀의 참조면. 실제 probe·scope 오차를 주장하지 않는다.

### dynamic_sharing — 동적 분담: 4 branch gate loop·common-source·skew·결합 (합성 D)

- 모델 수준: **D (합성 4-branch commutation cell)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 4 branch 모두 같은 합성 die 모델(분산은 R_on·V_th·지연·layout 입력으로만); 부하 인덕터 = 사건 동안 일정 전류원 I_total, 상측은 4 die 등가 하나; branch drain inductance 상호결합은 양의 인접 결합 k_M^|i−j| (양의 정부호) — 식별 불가 값은 범위로; power loop 감쇠는 R_p‖L_b 합성 등가, 고정 T_j, 역회복 없음
- 적용 불가: 특정 모듈·소자의 branch 전류 예측; 제조 분산으로부터의 수율 추정 (분포 입력 없음); 기생 발진의 정확한 한계 (감쇠 등가 의존); EMI
- 주장 한계: bounded synthetic equivalent (D). vendor 소자·모듈의 branch 전류 예측·수율을 주장하지 않는다.

## EX04 · transformer 설계 closure — 전기적으로 가능한 n·L을 실제 부품으로

범위: n 후보 3개 이상, core/geometry 가정, B/window screening, loss breakdown, L_m/L_σ/C_ps/R_ac source status, 공차 민감도, 공급사 질문 (E04 산출물); window 168.329 mm² (정확 168.3235), 2.60 W vs 2.26 W, 142.857/157.895/150.188 kHz

주장 한계: 코어·권선·절연 값은 합성 가정 — 최종 transformer 선정 아님; 정밀 core loss는 MISSING_INPUT; CLLC 결과는 FHA(A) 후보 — 스위칭 검증은 EX05; R_ac는 합성 Dowell, 식별 회로는 집중 capacitance 근사


### turns_window — 정수 권선·B·window·구리: DAB 50:3을 부품으로 옮기기

- 모델 수준: **A (screen) + C (검산)** · 기준 실행 상태: `MISSING_INPUT`
- 가정: window = bare copper 하한 (N_pI_p/J + N_sI_s/J)/k_u — 교재 E04 식; B screen은 최대 권선 전압 max(V₁,max, n·V_L,max)의 사각파: 직렬 L 배치를 확정하기 전의 보수적 값 (FL07); 운전 전류는 이상 SPS (L_m → ∞, 무손실) — 교재 window 예와 같은 정의; 합성 코어: A_e 250 mm², A_w 200 mm², MLT 60 mm, B_allow 0.2 T, 권선 온도 100 °C; L_m ∝ N_p² (같은 코어·gap)
- 적용 불가: 최종 transformer 선정; AC 구리 손실 (실험 3); core loss (MISSING_INPUT); 절연 거리·내전압 판정; 열 판정
- 주장 한계: B·window·DC 구리 screen (A)과 이상 SPS 전류 (C). 최종 transformer 선정·절연·열·core loss는 주장하지 않는다.

### cllc_turns — CLLC n = 0.93을 정수 권선으로: 운전점 지도가 n의 허용 범위를 정한다

- 모델 수준: **A (FHA)** · 기준 실행 상태: `CANDIDATE_FHA_ONLY`
- 가정: FHA (기본파 등가, R_ac′ = 8n²V_o²/(π²P)): A 수준 정상 이득; link 전압은 교재 세 점(650/700, 800/800, 920/850 V)에서만 평가 (그 사이 조정은 가정); 합성 코어: A_e 800 mm², A_w 600 mm², MLT 90 mm, B_allow 0.2 T; B는 f_min과 max(V_i, nV_o)의 사각파; 2차 tank: n = 0.93 설계의 물리 L_r2 = 46.25 µH, C_r2 = 24.34 nF 유지
- 적용 불가: 스위칭 운전점·ZVS·SR (EX05); 경부하·역방향·기동; 최종 권선 선정; core loss·열
- 주장 한계: FHA 정상 이득(A)과 합성 코어 screen. 스위칭 운전점·ZVS·최종 권선 선정은 주장하지 않는다.

### harmonic_copper — 비정현파 권선 손실: 고조파별 R_ac vs R_dc만

- 모델 수준: **A (정확 Fourier)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 권선이 선형(저항이 전류에 무관)이라 성분별 손실을 더할 수 있다; 두 권선 전류 비가 일정해 단일 R_ac(f)로 근접효과를 대표한다
- 적용 불가: core loss; 측정 R_ac(f) 없는 정밀 권선 손실; 자화전류로 두 권선 MMF가 불균형한 경우의 근접효과
- 주장 한계: 선형 R_ac 모델의 권선 손실 합 (A). R_ac는 합성, core loss는 MISSING_INPUT.

### tolerance_map — L·C ±5 %와 상관관계: f_r 이동과 운전점 지도

- 모델 수준: **A (FHA + MC)** · 기준 실행 상태: `CANDIDATE_FHA_ONLY`
- 가정: 공차: L ±5 %, C ±5 %, L_m ±15 % = 3σ 로그정규; 두 tank의 L은 같은 인자, C는 같은 인자로 움직인다 (교재 corner 방식); L_m은 L·C와 독립; FHA (A 수준)
- 적용 불가: 스위칭 운전점·ZVS; 온도에 따른 L·C 드리프트의 실제 상관 (자료 필요); 양산 수율 주장
- 주장 한계: FHA 운전점과 가정 분포의 Monte Carlo (A). 양산 수율·스위칭 운전점은 주장하지 않는다.

### identification — L_m·누설·capacitance·R_ac 식별 계획: 측정 조건이 값을 바꾼다

- 모델 수준: **C + A** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: T-등가회로 + 집중 권선 capacitance (C_p는 1차 포트, C_s는 2차 포트); C_ps는 다른 권선을 계측 접지에 묶으면 포트에 병렬로 더해지는 lumped 근사; 선형 코어 (μ가 B·f와 무관), 코어 손실 저항 없음; LCR 미터 = 이상 사인 전압원 + 이상 전류 측정 (fixture는 보정되었다고 가정)
- 적용 불가: 실제 transformer 값; 분포 capacitance의 고차 공진; 측정기 정확도 사양
- 주장 한계: 합성 등가회로의 측정 조건 영향 (C·A). 실제 transformer 값·측정기 정확도는 주장하지 않는다.

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

## EX08 · 전열 연성·mission — 최고온도 한 점에서 mission으로

범위: coupled temperature와 calibrated lifetime 한계 (E13 표): fixed 24/25 K, coupled 26.569592/27.751513 K, spectral radius 0.09789; 동적 self/cross physical boundary; mission/coolant/parameter spread; loss-only·uncoupled·coupled 비교

주장 한계: 열 행렬·α는 합성 예 — 실제 모듈 값이 아님; Foster 내부 노드를 실제 package 층으로 취급하지 않음; 발산(ρ ≥ 1)은 단순 모델의 결과이며 실물 열폭주 확정이 아님; lifetime 데이터가 없으므로 온도 사이클 histogram과 상대 proxy만 — calibrated 수명(년) 생성하지 않음; MC 분포·상관은 가정 (seed 명시)


### thermal_matrix — 열 행렬: 고정 24/25 K → 연성 26.57/27.75 K, spectral radius 0.098

- 모델 수준: **A (정상상태 열 행렬)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 정상상태 열 행렬 Z (같은 경계 온도 기준의 self/cross 항) — 다른 경계 기준의 R_th를 섞지 않음; 손실 P = P₀ + diag(αP₀)ΔT (전체 손실에 한 선형 온도계수, 합성); 물리 망: 두 die Cauer stack + 공유 baseplate·방열판 (port 행렬 = Z가 되도록 구성)
- 적용 불가: 실제 모듈의 열 행렬 (측정·해석 필요); 비선형 손실 곡선의 큰 온도 범위; 과도·mission (실험 2·3)
- 주장 한계: 정상상태 열 행렬·선형 온도계수. 실제 모듈 열 행렬·파손온도를 주장하지 않는다.

### dynamic_network — 동적 self/cross 열망: 물리 경계와 Foster 함정

- 모델 수준: **B (물리 multiport 열망)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: die별 3층 Cauer stack(합성 층값) + 공유 baseplate·방열판 → 냉각수 (port 행렬 = Z); 손실 온도 의존 없음 (이 실험은 경로·시간 규모만); 냉각수는 이상적인 경계 온도
- 적용 불가: 실제 모듈의 cross 열 경로; 측정 Z_th의 경계가 다른 경우; 수명 (실험 3: 상대 proxy만)
- 주장 한계: 합성 층값의 물리 multiport 열망. 실제 모듈의 cross 경로·계면 값을 주장하지 않는다.

### mission — mission·냉각수·분산: 온도 사이클 histogram과 상대 손상 proxy

- 모델 수준: **B (정확 선형 열망) + 사이클 계수 (상대 proxy)** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 합성 mission (도심 가속/순항/회생/정지, 고속, 등판, 주차) — 실제 차량 부하 분포 아님; 실험 2의 물리 multiport 망, 손실 P = P₀·f(t)·[1 + α(T_j − T_ref)]; 손실 모델 적용범위 T_j ≤ 175 °C (ASSUMED) — 넘으면 궤적을 끊고 proxy를 만들지 않음; rainflow: ASTM E1049 3점법, 반전점 히스테리시스 0.2 K; 상대 손상 proxy: Coffin–Manson–Arrhenius 모양, m = 5, E_a = 0.8 eV, ΔT_ref = 10 K, T_ref = 100 °C (모두 ASSUMED) — 같은 식끼리의 비만 의미; MC: 독립 정규/로그정규 분산, seed 고정 (상관·모델 오차 미포함)
- 적용 불가: 수명(년)·보증 기간; IGBT wire-bond 계수를 SiC package에 적용; DC-link capacitor·solder 등 다른 고장 기구; 실제 mission 빈도·불확도
- 주장 한계: 합성 mission의 온도 사이클과 상대 손상 proxy. 수명(년)·보증을 주장하지 않는다.

## EX09 · EMI: noise source, 경로, victim을 같이 본다

범위: CM/DM source–path–victim lumped network와 spectrum proxy; 5 A·50.329 MHz·0.158965 mA; gate 속도·경로 C·snubber 변화와 손실 (지침 10절)

주장 한계: relative conducted-noise proxy — receiver·규격 LISN이 없어 EMI compliance는 NOT_EVALUABLE, PASS를 만들지 않음; 합성 lumped 경로와 합성 채널 전류 모델 (vendor 소자·실제 PCB 아님); 누설전류 계산은 단일 이상 경로 (규격 시험 아님); 3-level NPC midpoint 확장은 구현하지 않음 (지침상 선택 사항)


### hand_screens — 손계산 screen: C·dv/dt = 5 A, 50.329 MHz, 0.158965 mA — 그리고 단위 함정

- 모델 수준: **A (screen) + C (검산)** · 기준 실행 상태: `SCREEN_ONLY`
- 가정: 이상 램프 전원, lumped C_par·L·R (분포 layout 없음); 링잉 검산은 직렬 RLC 스텝 응답; line 전류는 순수 정현파 1개 경로
- 적용 불가: EMI 규격 판정 (NOT_EVALUABLE); 누설·접촉전류 규격 판정; 실제 PCB·케이블의 분포 정수 효과
- 주장 한계: screen 수치와 그 시간영역 검산. EMI·누설전류 합격 판정이 아니다.

### source_path_victim — Source → Path → Victim: 스위칭 edge에서 측정 port까지의 CM/DM proxy spectrum

- 모델 수준: **C + A** · 기준 실행 상태: `NOT_EVALUABLE`
- 가정: hard-switched cell: 채널 전류는 gate 속도로 정한 PWL, diode는 이상 (역회복 없음), C_node 선형; CM 경로: 노드→C_par(+R_par)→chassis→(2C_Y+기생 ∥ 케이블 L/2 → 측정망 R_m/2 ∥ L_m/2)→DC bus; DC bus 두 선은 HF에서 묶임; DM 경로: DC-link 전류(diode 전류)가 C_dc(ESR·ESL)와 선로(2L_cab + 두 측정 branch)로 나뉨 — 주파수영역 계산; 측정망은 50 Ω ∥ 50 µH 형태의 합성 proxy (규격 LISN·receiver 아님)
- 적용 불가: EMI 규격 합격/불합격; 방사 EMI; 실제 PCB·케이블 분포정수; 소자 데이터 기반 스위칭 손실
- 주장 한계: 합성 lumped 경로의 relative conducted-noise proxy. EMI compliance PASS를 만들지 않는다.

### mitigation — gate 속도·경로 C·snubber·Y를 하나씩 바꾸고 잡음과 손실을 같이 본다

- 모델 수준: **C + A** · 기준 실행 상태: `NOT_EVALUABLE`
- 가정: 기준 cell·경로는 실험 2와 같음 (snubber·Y 없음에서 출발); C_par 축소 = 같은 면적에서 절연 두께 증가 (R_th,TIM ∝ 1/C_par; P_dev 60 W, R_th,TIM 0.15 K/W는 ASSUMED); Y 커패시터 누설은 AC 측 단일 이상 경로 계산
- 적용 불가: EMI 규격 합격/불합격; 필터 설계 확정; 실제 소자 gate 저항–손실 곡선
- 주장 한계: relative proxy의 대역별 변화와 합성 모델의 손실·열 비용. 필터·대책의 규격 합격을 주장하지 않는다.

## EX10 · 고장: gate를 꺼도 에너지는 남아 있다

범위: fault 종류별 source/path/stored-energy/timing 표, detection uncertainty, turn-off overshoot tradeoff, discharge dimensioning, energy residual (E10); 320 J·4370.344 A @ 120.919958 µs·772.121 Ω·828.885 W·318.2 J

주장 한계: SC survival·fuse clearing·안전 qualification이 아니다; 보호 단계 시간·SC 내량·포화 전류는 합성 학습 값; freewheel은 비돌극 근사 (임계속도는 정확); ISO 26262·ASIL 적합성을 주장하지 않는다


### rlc_fault — DC-link 단락: 320 J이 수동 RLC로 방전된다

- 모델 수준: **A + C + 독립 ODE** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 외부 전원 분리, 초기 전류 0, 이상 단락 스위치 (t = 0에 닫힘); 선형 L·R·C (온도·주파수 무관), 반도체 비선형 제한 없음; clamp 선택 시 브리지 diode = 이상 diode + 일정 전압 V_cl
- 적용 불가: SiC/IGBT 실제 단락 전류·생존 시간; fuse 용단·아크; ISO 26262·ASIL 판단; 배터리 기여가 있는 고장
- 주장 한계: 수동 RLC 사고 실험 (A+C). SC survival·fuse clearing·안전 적합성은 주장하지 않는다.

### protection_timeline — 보호 timeline: 더하지 말고 그린다 (DESAT·과전류·시스템 경로)

- 모델 수준: **A** · 기준 실행 상태: `SCREEN_ONLY`
- 가정: 고장 전류 = V_dc/L_f 선형 상승 + 합성 포화 한계 I_sat (소자 물리 아님); desaturation 전 V_DS ≈ 0, 이후 V_dc, turn-off 중 V_dc + L_σ·I/t_f (삼각 전류); 단계 시간 산포 -20 % / +40 %는 가정 (지터·온도 분포 없음)
- 적용 불가: 실제 SC 생존·파괴 판정 (vendor 조건: V_dc, V_GS, T_j, R_g, test circuit); ISO 26262·ASIL; 링잉·EMI에 의한 오검출의 실제 확률
- 주장 한계: 합성 timing budget screen (SCREEN_ONLY). 소자 qualification·SC survival·ISO 26262는 주장하지 않는다.

### discharge_resistor — 방전저항: 평균 전력이 아니라 펄스 에너지로 고른다

- 모델 수준: **A + C** · 기준 실행 상태: `FAIL_CONSTRAINT`
- 가정: 이상 RC (스위치 저항·C의 ESR·전압 의존 없음); 저항 정격(연속 전력·펄스 에너지·작동전압)은 합성 가정; 직렬 분배 worst-case: 1개 +공차, 나머지 −공차
- 적용 불가: 부품 선정 확정 (MISSING_INPUT); 안전 요구(방전 시간) 적합성 판정; 스위치 고장 모드·진단
- 주장 한계: 이상 RC 치수화와 공차 corner (A+C). 부품 선정 확정·안전 요구 적합성은 주장하지 않는다.

### safe_state — ASC vs freewheel: 속도·역기전력·DC bus·드라이버 전원이 정한다

- 모델 수준: **B/C** · 기준 실행 상태: `CUSTOMER_DECISION_REQUIRED`
- 가정: 속도 일정 (관성 무한), 선형 L_d·L_q·ψ_m (포화·감자 없음); ASC: 이상 스위치로 v_d = v_q = 0; freewheel: 이상 diode, 비돌극 근사 L = (L_d + L_q)/2 (임계속도는 인덕턴스와 무관해 정확); 전류 한계·과전압 한계는 합성 가정
- 적용 불가: 감자(demagnetization) 판정 (MISSING_INPUT); 차량 안전 상태 결정·ISO 26262·ASIL; 배터리 충전 수용·BMS 반응; diode 역회복·스위칭 과도
- 주장 한계: 조건부 비교 (B/C). 차량 안전상태 결정·감자 판정·ISO 26262/ASIL은 주장하지 않는다.

## EX11 · 모델을 믿을 수 있는 범위: 검증·식별·불확도

범위: uncertainty budget, calibration/holdout 구분, identifiability 예제, test independence map, model card (교재 E11); 15.40157 W·4.87487 W·0.2991 %

주장 한계: 모든 입력은 합성·가정 (seed 명시); 실제 계측기·부품 분포 자료는 MISSING_INPUT; 통계 상한은 IID 가정 조건부 — 결정적 grid에는 붙이지 않음 (NOT_EVALUABLE); test-independence 표는 verification 경로의 지도이며 hardware validation이 아님


### loss_uncertainty — 11 kW에서 손실 220 W의 불확도: 15.40 W, 상관 0.9면 4.87 W — 그 상관의 근거는?

- 모델 수준: **A + MC** · 기준 실행 상태: `UNRESOLVED_RANKING`
- 가정: 입력 오차는 정규분포(Monte Carlo), 1차 전파는 선형화; 두 후보는 같은 setup으로 따로 측정 (ρ_AB는 후보 간 공통 오차); 측정 경계(보조전원·냉각수 전력·열평형)는 같다고 가정
- 적용 불가: 실제 계측기 사양의 해석 (MISSING_INPUT: 사양서·교정성적서 필요); 비정상상태·고조파가 큰 전력 측정
- 주장 한계: 1차 GUM 전파와 MC의 합성 예. 실제 계측기 사양 해석·교정 자료는 MISSING_INPUT.

### binomial_bounds — 0 fail / 1000 IID의 95 % 상한 0.2991 % — 그리고 결정적 grid에는 bound를 붙이지 않는다

- 모델 수준: **A** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 시행은 IID Bernoulli (같은 분포, 서로 독립); 합성 실패 문제: g = θ1+θ2+θ3 > 임계, θ ~ N(0,1)
- 적용 불가: 양산 불량률 보증; 상관된 공정·systematic bias가 있는 시험; 결정적 grid·corner sweep 결과의 확률 해석
- 주장 한계: IID Bernoulli 가정의 통계 상한. 양산 불량률 보증이 아니다.

### identifiability — ringing 하나로는 L과 C를 따로 알 수 없다 — 식별 가능성과 calibration/holdout

- 모델 수준: **A + C** · 기준 실행 상태: `OUT_OF_VALIDITY`
- 가정: 합성 ringing: 직렬 RLC step 응답 + 백색 잡음 (seed 명시); 두 번째 excitation의 추가 C는 정확히 안다고 가정; 합성 손실면에 고온 모델 형식 항 포함; calibration 모델은 그 항이 없다
- 적용 불가: 실측 파형의 parameter 추출 결과 (probe·layout·비선형 C 미포함); calibration 데이터 밖의 온도·전류 예측
- 주장 한계: 합성 데이터로 보인 식별 가능성·holdout 원리. 실측 parameter 추출이나 validation이 아니다.

### monte_carlo — Monte Carlo 보고: 분포·상관·seed·N·수렴, 모델 편향은 따로

- 모델 수준: **A + MC** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: ln L, ln C ~ 정규 (σ_L 2.5%, σ_C 2.5%, 상관 0); 사양: |f_r/f_0 − 1| ≤ 4%; 모델 형식 편향은 ± 고정값 (표본 아님)
- 적용 불가: 실제 부품 분포·lot 상관이 없는 양산 수율 주장; 희귀 실패(ppm)의 확률
- 주장 한계: 가정 분포 조건부 수율. 양산 수율 보증이 아니다.

### test_independence — 이 저장소의 검증은 어떤 경로로 독립인가: test-independence 표와 모델 카드

- 모델 수준: **메타** · 기준 실행 상태: `PASS_WITHIN_MODEL`
- 가정: 기준 preset만 실행 (fast 실험 먼저, 시간 예산 안에서); 소스 스캔은 Check(independent=상수)만 셈 — 계산된 flag는 ‘동적’
- 적용 불가: hardware validation; test가 실제로 얼마나 독립인지의 의미 판정 (선언과 구조만 셈)
- 주장 한계: 저장소 자체의 verification 경로 지도. 실물 validation·의미 판정은 포함하지 않는다.

## EX12 · 설계 리뷰를 통과하는 답변 — 세 개의 통합 사례

범위: E12: 3개 customer case decision memo를 실제 simulation run으로 생성 (요구/경계·지배제약·후보·수치증거·측정불확도·권고조건·다음 분별시험); root cause UNRESOLVED 허용; 답변은 rubric

주장 한계: 메모의 수치는 합성 모델의 실제 실행 결과; model PASS ≠ 사용자 숙련; root cause를 시뮬레이션으로 확정하지 않음


### capstone_a — Capstone A · 11 kW OBC의 high-battery 충전 불가

- 모델 수준: **통합** · 기준 실행 상태: `FAIL_CONSTRAINT`
- 가정: FL05·FL10·EX05의 기준 preset 실행 결과를 그대로 사용; 불확도는 독립 성분의 제곱합 (상관 무시)
- 적용 불가: 실제 제품 정격; 경쟁사 비교 (조건 정렬 전)
- 주장 한계: 합성 교육 사양의 결정 메모. 실제 제품·경쟁사 정격 비교 아님.

### capstone_b — Capstone B · 900 V ↔ 저전압 DAB의 순환전류와 ZVS

- 모델 수준: **통합** · 기준 실행 상태: `UNRESOLVED_RANKING`
- 가정: FL08·EX06의 기준 preset 실행 결과; 모듈당 1.5 kW 기준 (합계 3 kW와 구분)
- 적용 불가: 효율 개선 주장; topology 최종 선정
- 주장 한계: 합성 교육 사양의 결정 메모. 효율 개선 주장 없음.

### capstone_c — Capstone C · 병렬 SiC 인버터의 한 branch 과열과 보호 동작

- 모델 수준: **통합** · 기준 실행 상태: `UNRESOLVED_ROOT_CAUSE`
- 가정: EX03 기준 preset 실행 결과; 합성 소자·layout 값
- 적용 불가: 소자 불량 판정; 보호 설정 승인
- 주장 한계: 합성 교육 사양의 결정 메모. root cause 확정 없음.
