# MATLAB/Octave 독립 교차검증

Python 앱이 낸 숫자를 **교재 식에서 다시 계산해** 비교한다. 코드는 MATLAB 문법(`.m`)으로 썼고,
이 환경과 CI에서는 **GNU Octave 8.4.0에서 실제로 실행**한다. Python 코드를 옮겨 쓰지 않았다. 닫힌 식,
`ode45`, `integral`, `fzero`, `eig`처럼 앱과 다른 경로로 값을 구한다.

| 실행 환경 | 상태 |
|---|---|
| GNU Octave 8.4.0 | 실행함. 결과는 `results/octave_crosscheck.json`. CI(`.github/workflows/ci.yml`의 `octave` job)도 같은 스크립트를 돌려 JSON을 artifact로 올린다 |
| MATLAB | NOT_RUN_ENVIRONMENT (설치되어 있지 않음). 문법은 MATLAB 호환(기본 함수만, Optimization Toolbox 없이)으로 썼지만 MATLAB에서 돌려 본 적은 없다 |
| Simulink / Simscape | NOT_RUN_ENVIRONMENT. `simscape/README.md` 참고 |

## 실행

```bash
PYTHON=<앱의 venv python> verification/run_octave_crosscheck.sh
```

1. `python -m convlab run LAB EXP --preset P --out matlab/results/python_export`로 필요한 preset을 내보낸다
   (이 폴더는 매번 새로 만들고 git에 올리지 않는다).
2. Octave에서 `run_all.m`이 JSON을 읽고 교재 식으로 다시 계산한다 (약 4분; 대부분 CLLC 스위칭 주기해).
3. `results/octave_crosscheck.json`을 쓴다. 한 줄에 한 항목(item, expected, octave_value, rel/abs error, tol,
   PASS/FAIL, runtime)이고, 머리말에 Octave 버전, 플랫폼, 내보낸 파일마다 앱 버전·request hash·CLI 명령이 있다.
4. FAIL이 하나라도 있으면 종료 코드 1. Octave나 Python이 없으면 종료 코드 2 (NOT_RUN_ENVIRONMENT).

Octave/MATLAB 안에서 직접: `addpath matlab; ok = run_all('matlab/results/python_export')`.

## 무엇을 어떻게 비교하나

| 파일 | 대상 (preset) | 독립 계산 |
|---|---|---|
| `xc_dab.m` | FL08 `sps_nominal/nominal`, `zero_power_mismatch/mismatch` | 교재 11장 닫힌 식(φ, i₀, Irms, Pmax); `ode45`를 edge에서 edge까지 적분하고 zero-mean 정상해를 골라 `fzero`로 P₂(φ)=1.5 kW의 φ를 다시 찾음; mismatch는 삼각파 식과 Lm 가지 포함 `ode45` |
| `xc_ex06.m` | EX06 `general_modulation/textbook` | E06 switching function b(θ;w,φ)로 SPS와 w₁=0.7π 후보를 `ode45` + 가장 작은 φ 근(격자 + `fzero`); 같은 점을 Fourier 급수(홀수 고조파 200 000개)로 한 번 더 |
| `xc_ex02.m` | EX02 `qe_integrals`, `hb_constant_current` (textbook) | Qoss·Eoss를 `integral`과 닫힌 식으로; C_node(v)dv/dt = I를 `ode45` + event로 적분해 전환시간 286.606 ns, dead time 끝의 node 전압 |
| `xc_ex07.m` | EX07 `cpl_exact/c100u`, `c1m` | E07 평형점·C_crit·`eig(A)` 극점; 비선형 모델 `ode45`의 극값에서 성장률·ω_d, 유효범위 이탈 시각 |
| `xc_llc.m` | FL09 `fha_gain/textbook` | 12장 정의값; Q 0.2/0.8/1.5 × F 7점 \|H\| 표를 페이저 회로로 (표는 소수 넷째 자리까지라 ±5e-5); 정규화 식과 회로식 일치; 표의 경계 문구 `F < 0.4388 capacitive …`를 `fzero`(Im Z_in = 0)로 확인(인쇄 자릿수의 반), "범위 안 전부 inductive" 문구는 F_b < F_min과 대조 |
| `xc_cllc.m` | FL10 `seed_fail/seed`, `fix_n093/textbook` + 교재 13장 인쇄값 (`FL10.tb.*`) | seed(n=1) 세 corner의 inductive 최대 gain과 그 주파수 (920/850 V: 1.016401 < 1.082353, 해 없음); n=0.93 세 corner의 inductive FHA 해, 920/850 V 두 해의 기울기·1차 RMS |
| `xc_cllc_td.m` | FL10 `time_domain/textbook`, `lossy`; EX05 `operating_points/textbook` | **CLLC 스위칭 모델을 따로 만든 hybrid `ode45`로** (아래 절): FHA 해에서의 전력·I₁ RMS·정류 off·Floquet, 11 kW 점의 C 전압 peak·dP/df, 11 kW 주파수; EX05 출력 C·R 부하의 V_o와 920 V 주파수 |
| `xc_fl05.m` | FL05 `grid_boundary`, `dclink_power` (nominal) | 8장 선전류·변조지수 식; 3상 순시전력의 한 주기 평균으로 `fzero`; C v dv/dt = p(t) − P를 `ode45`로 적분한 ripple p-p와 hold-up 시간 |
| `xc_fl01.m` | FL01 `buck_ccm/nominal` | 스위칭 R-L-C(ESR는 C 가지, R = D·V_in/I_o)를 on/off 구간별 `ode45`로 한 주기 적분; 구간이 모두 선형이라 주기 map이 affine이므로 (I − M)x₀ = c로 주기해; 인덕터·스위치 평균/rms, 출력 ripple(극값은 event로) |

허용오차: 닫힌 식 1e-9~1e-12 (상대), `ode45`/`fzero` 1e-8 (상대), 인쇄된 표·교재 값은 마지막 자리의 반,
비선형 성장률 fit 1e-3 (상대), CLLC 스위칭 모델은 아래 표. 모두 비교 전에 정했다.

## CLLC 스위칭 모델 (`xc_cllc_period.m`, `xc_cllc_orbit.m`, `xc_cllc_td.m`)

교재 E05의 1차 환산 T 회로를 그대로 적었다: v₁ → R₁·L₁·C₁ → node m (L_m) → L₂′·C₂′·R₂′ → 다이오드 bridge.
도통 중 v_m = [(v₁ − v_C1 − R₁i₁)/L₁ + (v_C2 + v₂ + R₂i₂)/L₂] / (1/L₁ + 1/L_m + 1/L₂).

- **bridge**: 이상 full bridge, v₁ = +V_link (0 ≤ t < T/2), −V_link (나머지). dead time 없음.
- **정류기**: 세 상태 P (i₂ > 0, v₂ = +n·V_o), N (i₂ < 0, v₂ = −n·V_o), off (i₂ = 0, 정류기 입력이 떠서
  v_r = L_m/(L₁+L_m)·(v₁ − v_C1 − R₁i₁) − v_C2). P/N은 i₂가 0을 지날 때 끝나고, 그때 v_r이 반대쪽 clamp를 넘으면
  반대 도통, 아니면 off. off는 v_r이 ±n·V_o에 닿으면 끝난다. bridge edge에서 v_r이 뛰므로 off를 다시 판정한다.
  이 논리는 앱 코드를 보지 않고 회로에서 다시 적었다.
- **출력**: 강한 배터리(FL10) 또는 C_o·R_L (EX05, C_o dv_o/dt = n|i₂| − v_o/R_L, 실제값).
- **적분**: 구간마다 `ode45` (RelTol 1e-10, AbsTol 1e-9, MaxStep T/100), event로 모드 전환 시각을 찾는다. Octave의
  `ode45`는 호출의 첫 step에서 찾은 event를 멈추지 않으므로(MATLAB 호환 동작) 첫 terminal event 뒤의 sample은 버린다.
- **주기해**: 주기 map x₀ → Φ_T(x₀)에 Newton shooting (전진차분 Jacobian, chord 재사용), 잔차 1e-10 (상태 scale 기준).
  Floquet 승수는 수렴한 주기해에서 중앙차분 monodromy의 고유값.
- **계속법**: FHA 해에서 출발해 주파수를 조금씩 옮기며(이전 주기해로 warm start) 11 kW 점과 V_o = 920 V 점까지 가고,
  regula falsi (Illinois)로 근을 1e-4 Hz(EX05는 1e-5 Hz)까지 좁힌다.

| 비교 | 허용오차 | 근거 |
|---|---|---|
| 스위칭 전력, C 전압 peak, V_o | 1e-5 (상대) | RelTol 1e-10의 결과가 1e-11과 1e-9 이내로 같다 (136.099 kHz 점 1.4e-10, 11 kW 점 2e-11). 앱은 shooting 잔차 1e-8을 받아들이고, 가장 느린 Floquet 승수 0.989가 이를 약 100배 키울 수 있다 → 1e-6. 1e-5는 그 10배 |
| 표에 인쇄된 값 (kW 2자리, A 2자리, % 1자리) | 마지막 자리의 반 (+1e-5 상대) | 인쇄 반올림 |
| Floquet \|λ\|max | 6e-5 | 인쇄 4자리의 반 5e-5 + 중앙차분 monodromy 1e-5 |
| 11 kW 주파수 | 0.5 Hz | 앱은 1 Hz 구간까지 이분해 중간점을 낸다 |
| 앱의 11 kW 주파수에서의 전력 vs 11 kW | 0.5 Hz × \|dP/df\| | 같은 이분 폭을 기울기로 환산 |
| dP/df (±2 Hz), dV_o/df (±50 Hz) | 두 값의 허용오차 1e-5에서 유도 | 차분 정의는 앱과 같다 |
| EX05 V_o = 920 V 주파수 | 1.1e-3 Hz | 앱 brentq xtol 1e-3 Hz + 이쪽 1e-4 Hz |

## 현재 결과 (Octave 8.4.0)

220개 항목: **220 PASS, 0 FAIL, 0 TODO** (9개 그룹, 약 210초; 그중 202초가 CLLC 스위칭 주기해).
`verification/run_octave_crosscheck.sh`는 종료 코드 0을 낸다.

- **FL09 경계 문구.** 이전 실행에서 FAIL 2개였던 `t_gain` 표의 경계 열은 앱에서 고쳐졌다 (4ee1a30, 닫힌 식, 소수 넷째
  자리). 지금 문구 "F < 0.4388 / 0.8442 / 0.9554 capacitive"는 Octave의 Im Z_in 근 0.438842 / 0.844213 / 0.955411과
  인쇄 자릿수의 반(5e-5) 안에서 같고, "범위 안 전부 inductive"는 F_b < F_min인 Q = 0.2에만 붙어 있다.
- **CLLC 스위칭 모델 (53개, 모두 PASS).** 주요 값:

| 항목 | 앱 | Octave | 차이 | 허용 |
|---|---|---|---|---|
| 136.099 kHz FHA 해의 배터리 전력 (무손실) | 17 341.4455 W | 17 341.4455 W | 1.2e-11 | 1e-5 |
| 147.061 kHz FHA 해의 배터리 전력 (무손실) | 22 242.0046 W | 22 242.0046 W | 3.1e-9 | 1e-5 |
| 같은 두 점의 Floquet \|λ\|max | 0.9224 / 0.9891 | 0.92241 / 0.98914 | < 4e-5 | 6e-5 |
| 137.183 / 145.905 kHz (R = 50 mΩ) | 17 401.5743 / 19 075.7718 W | 같음 | 2e-10 / 1.9e-9 | 1e-5 |
| 앱의 11 kW 주파수에서 v_C1 / v_C2′ peak (무손실) | 860.1104 / 815.9248 V | 860.1089 / 815.9232 V | 1.8e-6 / 2.0e-6 | 1e-5 |
| 그 점의 dP/df (±2 Hz) | −1 630 030 W/kHz | −1 630 034 W/kHz | 2.3e-6 | 45 W/kHz |
| 위 branch 11 kW 주파수 (무손실 / 손실) | 148 004.741 / 147 441.894 Hz | 148 004.862 / 147 442.165 Hz | 0.12 / 0.27 Hz | 0.5 Hz |
| EX05 V_o (C·R 부하) at 136.099 / 147.061 kHz | 961.5181 / 922.9362 V | 같음 | 3e-11 | 1e-5 |
| EX05 V_o = 920 V 주파수 | 148 005.23039 Hz | 148 005.23041 Hz | 1.8e-5 Hz | 1.1e-3 Hz |

다른 corner(650/700 V, 800/800 V)의 전력·I₁ RMS·정류 off·Floquet도 표 `t_td`의 인쇄값과 같다. 11 kW 점에서는 가장
느린 Floquet 승수가 0.99999라 앱의 shooting 잔차 1e-8이 이론상 크게 커질 수 있지만, 실제 차이는 2e-6이었다.

**앱에서 확인이 필요해 보이는 것** (FAIL은 아니다; 허용오차 안이지만 해석에 영향):

1. **FL10 `time_domain textbook`의 11 kW 운전점은 11 kW가 아니다.** `f_sw_hi` = 148 004.741 Hz는 1 Hz 구간 이분의
   중간점이다. 그 점의 dP/df가 −1630 W/Hz라서 이 주파수의 전력은 이 모델에서 **11 632 W (+5.7 %)**이고, 앱이 그 점에
   보고하는 I₁ RMS 15.87 A, v_C1 860.11 V, v_C2′ 815.92 V도 11.63 kW의 값이다 (Octave가 같은 주파수에서 2e-6 안으로
   재현). 정확히 11 kW인 148 004.862 Hz에서는 15.08 A, 817.5 V, 771.6 V다. 이분을 1 mHz까지 하거나 그 점의 전력을
   함께 보고하면 해결된다. 손실 preset에서는 같은 효과가 +5 W (+0.05 %)라 문제없다.
2. **그 운전점은 거의 중립 안정이다.** 이 모델의 Floquet 승수 최대값은 앱의 11 kW 주파수에서 0.9999853, 정확한 11 kW
   점에서 0.99998이다 (교란이 e배 줄어드는 데 약 6.8만 주기). 정류기가 주기의 1.3 % 동안 꺼지므로 승수 하나는 0이다.
   앱의 표는 Floquet을 FHA 해에서만 보여 준다. 손실 preset의 11 kW 점은 0.9958이다.
3. **P(f)가 그 근처에서 꺾인다.** 148.004 kHz에서는 정류 off가 0 %이고 12.20 kW, 148.0047 kHz에서는 off 1.33 %에
   11.63 kW다 (그 사이에서 off 구간이 생긴다). 148.006 kHz 7.15 kW, 148.010 kHz 2.79 kW. 148.05와 148.1 kHz에서는
   148.010 kHz 주기해에서 출발한 Newton이 수렴하지 않았고, 148.3 kHz에서는 다시 수렴했다 (575 W, off 14 %).
   무손실 강한 배터리 모델에서 이 근처의 운전점 숫자는 1 Hz 단위로 크게 달라진다 (탐색 기록은 검증 스크립트 밖에서
   같은 함수로 얻은 값).

## 이 검증이 말하지 않는 것

- 같은 이상 모델(이상 스위치·다이오드, FHA, 이상 CPL)을 다른 수치 경로로 재현했을 뿐이다. 실측, Simulink/Simscape,
  PSIM, 소자 모델(ZVS·Coss·dead time) 검증이 아니다.
- CLLC 스위칭 모델의 극단적 민감도(무손실 11 kW 점)는 이상 모델의 성질이다. 실제 회로의 저항·배터리 임피던스·dead time이
  이를 바꾼다 (lossy preset 50 mΩ만으로 dP/df가 84배 작아진다).
- MATLAB에서는 실행하지 않았다 (NOT_RUN_ENVIRONMENT).
