# MATLAB/Octave 독립 교차검증

Python 앱이 낸 숫자를 **교재 식에서 다시 계산해** 비교한다. 코드는 MATLAB 문법(`.m`)으로 썼고,
이 환경에서는 **GNU Octave 8.4.0에서 실제로 실행**했다. Python 코드를 옮겨 쓰지 않았다. 닫힌 식,
`ode45`, `integral`, `fzero`, `fsolve`, `eig`처럼 앱과 다른 경로로 값을 구한다.

| 실행 환경 | 상태 |
|---|---|
| GNU Octave 8.4.0 | 실행함. 결과는 `results/octave_crosscheck.json` |
| MATLAB | NOT_RUN_ENVIRONMENT (설치되어 있지 않음). 문법은 MATLAB 호환으로 썼지만 MATLAB에서 돌려 본 적은 없다 |
| Simulink / Simscape | NOT_RUN_ENVIRONMENT. `simscape/README.md` 참고 |

## 실행

```bash
PYTHON=<앱의 venv python> verification/run_octave_crosscheck.sh
```

1. `python -m convlab run LAB EXP --preset P --out matlab/results/python_export`로 필요한 preset을 내보낸다
   (이 폴더는 매번 새로 만들고 git에 올리지 않는다).
2. Octave에서 `run_all.m`이 JSON을 읽고 교재 식으로 다시 계산한다.
3. `results/octave_crosscheck.json`을 쓴다. 한 줄에 한 항목(item, expected, octave_value, rel/abs error, tol,
   PASS/FAIL, runtime)이고, 머리말에 Octave 버전, 플랫폼, 내보낸 파일마다 앱 버전·request hash·CLI 명령이 있다.
4. FAIL이 하나라도 있으면 종료 코드 1. TODO 항목은 실패로 세지 않는다. Octave나 Python이 없으면 종료 코드 2
   (NOT_RUN_ENVIRONMENT).

Octave/MATLAB 안에서 직접: `addpath matlab; ok = run_all('matlab/results/python_export')`.

## 무엇을 어떻게 비교하나

| 파일 | 대상 (preset) | 독립 계산 |
|---|---|---|
| `xc_dab.m` | FL08 `sps_nominal/nominal`, `zero_power_mismatch/mismatch` | 교재 11장 닫힌 식(φ, i₀, Irms, Pmax); `ode45`를 edge에서 edge까지 적분하고 zero-mean 정상해를 골라 `fzero`로 P₂(φ)=1.5 kW의 φ를 다시 찾음; mismatch는 삼각파 식과 Lm 가지 포함 `ode45` |
| `xc_ex06.m` | EX06 `general_modulation/textbook` | E06 switching function b(θ;w,φ)로 SPS와 w₁=0.7π 후보를 `ode45` + 가장 작은 φ 근(격자 + `fzero`); 같은 점을 Fourier 급수(홀수 고조파 200 000개)로 한 번 더 |
| `xc_ex02.m` | EX02 `qe_integrals`, `hb_constant_current` (textbook) | Qoss·Eoss를 `integral`과 닫힌 식으로; C_node(v)dv/dt = I를 `ode45` + event로 적분해 전환시간 286.606 ns, dead time 끝의 node 전압 |
| `xc_ex07.m` | EX07 `cpl_exact/c100u`, `c1m` | E07 평형점·C_crit·`eig(A)` 극점; 비선형 모델 `ode45`의 극값에서 성장률·ω_d, 유효범위 이탈 시각 |
| `xc_llc.m` | FL09 `fha_gain/textbook` | 12장 정의값; Q 0.2/0.8/1.5 × F 7점 |H| 표를 페이저 회로로 (표는 소수 넷째 자리까지라 ±5e-5); 정규화 식과 회로식 일치; 표의 inductive/capacitive 경계 문구를 `fzero`(Im Z_in = 0)로 확인 |
| `xc_cllc.m` | FL10 CLLC FHA (앱 미병합 → 교재 13장 인쇄값) | seed(n=1) 920/850 V 필요 gain 1.082353 vs inductive 최대 gain 1.016401 (해 없음); n=0.93 세 corner의 inductive 해, 920/850 V 두 해의 기울기·1차 rms; task에 적힌 FL10 draft 해 136099.47 / 147060.86 Hz (C_r1을 f_r = 150 kHz로 둔 값) |
| `xc_fl05.m` | FL05 `grid_boundary`, `dclink_power` (nominal) | 8장 선전류·변조지수 식; 3상 순시전력의 한 주기 평균으로 `fzero`; C v dv/dt = p(t) − P를 `ode45`로 적분한 ripple p-p와 hold-up 시간 |
| `xc_fl01.m` | FL01 `buck_ccm/nominal` | 스위칭 R-L-C(ESR는 C 가지, R = D·V_in/I_o)를 on/off 구간별 `ode45`로 한 주기 적분; 구간이 모두 선형이라 주기 map이 affine이므로 (I − M)x₀ = c로 주기해; 인덕터·스위치 평균/rms, 출력 ripple(극값은 event로) |

허용오차: 닫힌 식 1e-9~1e-12 (상대), `ode45`/`fzero` 1e-8 (상대), 인쇄된 표·교재 값은 마지막 자리의 반,
비선형 성장률 fit 1e-3 (상대). 기대값은 모두 Python export에서 읽고, FL10만 교재 인쇄값을 쓴다.

## 현재 결과 (Octave 8.4.0)

145개 항목: **142 PASS, 2 FAIL, 1 TODO**. `verification/run_octave_crosscheck.sh`는 지금 종료 코드 1을 낸다.

**FAIL 2개 — FL09 `t_gain` 표의 "∠Z_in 경계" 열.** 세 행 모두 `F < 0.955 capacitive`라고 쓰여 있지만,
독립 계산한 경계는 Q마다 다르다.

| Q | 표 문구 | Octave (Im Z_in = 0) | 올바른 문구 |
|---|---|---|---|
| 0.2 | F < 0.955 capacitive | F_b = 0.43884 (표 범위 0.7–1.5 밖) | 범위 안 전부 inductive |
| 0.8 | F < 0.955 capacitive | F_b = 0.84421 | F < 0.844 capacitive |
| 1.5 | F < 0.955 capacitive | F_b = 0.95541 | F < 0.955 capacitive (PASS) |

원인: `src/convlab/labs/fl09_llc.py`의 표 loop(`for q in qs:`)가 앞선 loop에서 남은 변수 `b`(마지막 Q = 1.5의
경계)를 모든 행에 쓴다. 그래프 marker는 `bounds`를 써서 맞다. 이 검증 작업에서는 다른 lab을 고치지 않았다.
`for q, b in bounds:`로 바꾼 사본(작업 트리 밖)에서 다시 내보내면 세 행 모두 PASS였다. FL09를 고친 뒤
스크립트를 다시 돌려 JSON을 갱신한다. |H| 값 21개는 모두 PASS다.

**TODO 1개 — CLLC 시간영역(FL10/EX05).** FL10이 병합되지 않아 FHA 숫자만 비교했다. `xc_cllc.m` 끝과
`run_octave_crosscheck.sh`의 export 목록에 `TODO(FL10/EX05 time domain)` hook이 있다. 병합되면 E05 상태식
(x = [i₁, i₂, v_C1, v_C2])을 `ode45` + shooting으로 풀어 두 분기(136.099 / 147.061 kHz)의 export와 비교한다.

## 이 검증이 말하지 않는 것

- 같은 이상 모델(이상 스위치, FHA, 이상 CPL)을 다른 수치 경로로 재현했을 뿐이다. 실측, Simulink/Simscape,
  PSIM, 소자 모델 검증이 아니다.
- CLLC 기대값은 교재 인쇄값이다. 교재는 C_r = 28.1448 nF로 적고, f_r = 150 kHz 정확값은 28.144773 nF다.
  이 차이로 근이 0.1 Hz 안에서 움직인다 (교재 허용 ±0.5 Hz 안).
- MATLAB에서는 실행하지 않았다 (NOT_RUN_ENVIRONMENT).
