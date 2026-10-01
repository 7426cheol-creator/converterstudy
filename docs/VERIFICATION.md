# 검증 범위 — 실제로 실행한 것과 실행하지 않은 것

이 문서는 이 저장소에서 **실제로 실행한 검증**과 **실행하지 않은 것**을 구분해 적습니다.

- 모든 검증은 합성 모델 안에서 방정식을 올바르게 풀었는지 보는 verification입니다.
- 실물 측정으로 모델의 목적 적합성을 보이는 validation(하드웨어)은 하나도 하지 않았습니다.
- 수치는 아래 날짜의 실행 결과이며, 같은 명령으로 다시 만들 수 있습니다(§5).

## 1. 한눈에

| 경로 | 무엇을 확인하나 | 상태 | 결과 |
|---|---|---|---|
| `pytest` | 실습별 교재 기준값(test에 숫자로 기재), 독립 경로 비교, 실패 사례 재현, 입력 거부, 회귀 | 실행 | §6 |
| `convlab run-all` | 모든 기준 preset 실행, 결과의 검증 check와 기준값 metric, 그래프 문단 계약 | 실행 | §6 |
| `convlab run-all --all-presets` | 기준이 아닌 preset까지 전부 실행(오류·계약 점검) | 실행 | §6 |
| `verification/independent_reference_check.py` | 교재·지침 기준값을 표준 라이브러리만으로 재계산. 앱 코드를 import하지 않음 | 실행 | §6 |
| `verification/run_octave_crosscheck.sh` | 교재 식을 MATLAB 언어로 다시 쓴 검산을 GNU Octave에서 실행 | 실행 (Octave 8.4.0) | §6, `matlab/results/octave_crosscheck.json` |
| `e2e/smoke.mjs` | 실제 브라우저(Chromium)에서 모든 실험의 실행, 제안 변경, 예측, 답 저장, export, 기록 화면 | 실행 | §6 |
| GitHub Actions CI | 위 Python 검증과 브라우저 스모크를 PR마다 실행 | workflow 작성. 마지막 push 이후 결과는 확인하지 못함(§4) | — |
| MATLAB / Simulink / Simscape | 대표점의 독립 회로 시뮬레이션 | **NOT_RUN_ENVIRONMENT** | builder만 있음(§4) |
| PSIM 12.0.2 | 수동 재구성 회로 | **NOT_RUN_ENVIRONMENT** | build sheet만 있음(§4) |
| 하드웨어 측정 | 실물 validation | **없음** | — |

## 2. 독립 경로의 종류

같은 함수를 두 번 부른 비교는 ‘회귀’로, 공식을 공유하지 않는 두 계산의 비교만 ‘독립’으로 셉니다(`Check.independent`). 실습별 목록은 [TEST_INDEPENDENCE.md](TEST_INDEPENDENCE.md)에 있습니다.

- **닫힌 식 vs 구간 선형 적분.** DAB SPS의 P, I_rms, I_pk(FL08, EX06), Buck 리플, 열 RC 응답 등.
- **행렬지수 엔진 vs 다른 적분기.** 공진 주기해는 scipy DOP853와 사건 탐지로 따로 적분합니다. 인버터는 손으로 쓴 dq ODE(RK45)와 비교하고, EX05의 사이클 map은 주기마다 solve_ivp를 연쇄해 비교합니다.
- **에너지 항등식.** 한 주기 E_in − E_out − E_loss − ΔW 잔차와 실험별 에너지 장부를 봅니다.
- **표현이 다른 같은 물리량.** CLLC 전력을 홀수 고조파 중첩과 시간영역 적분으로, FHA |H|를 교재 정규화식, 절점 페이저, 시간영역 FHA 회로로 각각 계산합니다.
- **소신호 vs 비선형.** EX05 G_vf는 사이클 map 선형화와 비선형 FM 주입을 비교합니다. 큰 진폭에서는 오차가 진폭에 비례해 주는지로 비선형과 결함을 구분합니다.
- **통계.** Monte Carlo 결과를 해석적 lognormal 수율과 비교합니다(EX11, EX04). 0 실패 상한은 Beta 분위수와 닫힌 식으로 비교합니다.
- **도구가 다른 재구현.** 표준 라이브러리 기준값 검산과 Octave 검산(fzero, ode45, integral, eig, Fourier 급수)이 있습니다.

## 3. 검증이 찾아낸 결함과 조치

검증 경로는 실제 결함을 찾았습니다. 조치한 것은 모두 test로 고정했습니다.

| 찾은 경로 | 결함 | 조치 |
|---|---|---|
| Octave 교차검증 | FL09 `t_gain` 표의 ∠Z_in 경계가 모든 Q 행에 마지막 Q의 값으로 들어감(루프 변수 재사용). 그래프 표식은 맞았음 | 경계를 닫힌 식(F²의 2차식)으로 계산하고 페이저 Im Z_in의 근과 독립 비교한다. Octave 값 0.43884 / 0.84421 / 0.95541로 test를 고정했다 |
| Octave 교차검증 (CLLC 스위칭) | FL10 ‘스위칭 11 kW 주파수’가 1 Hz 이분 구간의 중점이었다. dP/df ≈ −1.6 kW/Hz 경사라 그 점의 실제 전력은 11.63 kW(+5.7 %)였고, 함께 보고한 I₁·C 전압 peak도 11.63 kW 기준이었다 | 이분 뒤 Brent법으로 |P − 11 kW| ≤ 0.1 %까지 좁히고 그 점의 전력과 Floquet |λ|max를 함께 보고한다. 148004.862 Hz, C peak 817.5 / 771.6 V로 Octave와 일치하며, 거의 중립 안정(0.99998) 판정도 추가했다 |
| `run-all --all-presets` | EX05 G_vf `big_fm`(소신호의 100배 진폭)에서 1 kHz 비교가 2 %를 넘어 ‘검증 실패’로 집계됨 | 진폭 1/10 재실행으로 오차가 진폭에 비례해 주는지 확인한다(3.5 % → 0.007 %). 그렇다면 OUT_OF_VALIDITY(모델 범위)로, 아니면 FAIL로 둔다 |
| 브라우저 스모크 | EX11 `test_independence`의 제안 변경(예산 600 s)이 브라우저 요청 120 s를 넘김 | 제안 예산을 45 s로 바꾸고, 전체 실행은 별도 preset으로 뺐다 |
| 그래프 계약 점검 | 그래프 59개(16개 실습)에 ‘무엇을 입증했고 무엇은 아직인가’ 문단이 비어 있었음 | 모두 채우고 run-all이 이를 계약 위반으로 실패시키게 했다 |
| 화면 점검 | 추적표 화면의 실행·검증 열이 비어 있었음(생성 JSON 형식 불일치). 회로 제목이 보이지 않았음 | 생성기가 화면 형식의 rows를 쓰게 했고 회로 제목과 메모를 표시한다 |
| 공진 모델 shooting | 섭동된 상태에서 정류기 사건 처리 예외 | off 상태의 i₂ ≠ 0을 도통 상태로 보내는 규칙을 추가했다 |
| 성능 측정 | 작은 행렬 행렬지수가 BLAS 스레드 경쟁으로 약 290배 느려짐 | 패키지 import 시 BLAS 스레드를 1로 고정한다 |
| EX12 메모 | 다른 실습을 부를 때 표시 단위(kW) 숫자를 SI로 넘김. 전력 여유가 남는 경우를 부족으로 표기 | 단위를 명시해 넘기고 여유와 부족을 구분한다 |

## 4. 실행하지 않은 것

- **MATLAB / Simulink / Simscape: NOT_RUN_ENVIRONMENT.** 이 환경에 MATLAB이 없습니다.
  - `matlab/simscape/`에 builder 스크립트가 있습니다: DAB nominal·mismatch, CLLC 두 branch(기본파와 스위칭), EX02 비선형 C_oss 전환, EX07 CPL.
  - 블록은 경로를 추측하지 않고 이름으로 찾으며, 찾지 못하면 이유를 말하고 멈춥니다.
  - `.slx`나 실행 결과는 만들지 않았습니다. `status.json`이 모든 모델을 NOT_RUN_ENVIRONMENT로 기록합니다.
  - 첫 MATLAB 실행에서 확인할 것: Simscape 언어 세부, `_specify` 파라미터 이름, 열기 → compile → 시뮬레이션 → 저장·재열기.
- **PSIM 12.0.2: NOT_RUN_ENVIRONMENT.** PSIM이 없고 이 버전의 자동화 API를 가정하지 않았습니다.
  - `psim/`의 build sheet 7개를 손으로 회로를 만드는 용도로 둡니다: FL01, FL08, EX06, FL09, FL10, EX02, EX07.
  - 각 sheet에는 접속, 소자, gate 식, solver, 초기조건, probe, 앱 실행에서 가져온 기대값, 한계가 들어 있습니다.
- **GitHub Actions 결과.** workflow(`.github/workflows/ci.yml`)는 저장소에 있습니다.
  - 사용자의 GitHub 계정이 바뀌어 저장소를 `7426cheol-creator/converterstudy`로 옮겼고, 작업 브랜치 `ccr-947b03d9-ahvguh`로 push했습니다.
  - CI는 main push와 PR에서 돕니다. 이 문서를 쓰는 시점에는 아직 PR이 없어 GitHub의 CI 결과는 없습니다. 같은 명령의 로컬 결과는 §6에 있습니다.
- **하드웨어.** 측정, 열화상, 규격 시험, 수명 시험은 없습니다. ‘통과’는 모두 합성 모델 안의 판정입니다.

## 5. 재현

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m convlab run-all --out results               # 기준 preset
python -m convlab run-all --all-presets --out results_all
python verification/independent_reference_check.py
verification/run_octave_crosscheck.sh                  # GNU Octave 필요 (없으면 exit 2 = NOT_RUN_ENVIRONMENT)
(cd e2e && npm ci && node smoke.mjs --spawn)            # Node 22, Playwright Chromium
python tools/gen_docs.py --results results             # 추적표·모델 카드 등 재생성
```

## 6. 이번 실행 결과

2026-10-01 01:30–01:50 UTC, 커밋 42ce428(실습 24개 모두 병합) 기준입니다.

환경: Python 3.11.15, numpy 2.4.6, scipy 1.17.1, Linux x86_64(4코어), GNU Octave 8.4.0, Node 22.22.2, Playwright 1.56.1(Chromium).

| 경로 | 결과 |
|---|---|
| `pytest` | **378 passed**, 0 failed (316 s) |
| `convlab run-all` (기준 preset) | **223 runs**, 검증 실패 0, 그래프 문단 계약 위반 0. 기준값 metric 628개 모두 PASS(`docs/contract/reference_results_reconstructed.json`). 187 s(pytest와 동시 실행) |
| `convlab run-all --all-presets` | **398 runs**, 검증 실패 0, 계약 위반 0 (530 s) |
| `independent_reference_check.py` | **99 checks, 0 failed** (표준 라이브러리만, convlab 미 import) |
| Octave 교차검증 | **220 items: 220 PASS, 0 FAIL, 0 TODO** (Octave 8.4.0). 커밋 204298b에서 실행했고, 그 뒤 바뀐 코드는 FL12 추가와 문서뿐이며 Octave가 대조하는 실습은 바뀌지 않았다 |
| 브라우저 스모크 | **101 experiments, 0 failures** (355 s). 실험마다 기준 실행, 제안 변경 적용, 예측 선택, 재실행, 결과 영역 점검, 레이아웃 점검을 하고, 답 저장·export 3종·기록 화면도 확인 |
| GitHub Actions | 이 세션에서 결과 확인 못 함(§4) |

실습별 실행 수, 상태, 기준값, 독립·회귀 check 수는 [TRACEABILITY.md](TRACEABILITY.md)에 있고, 실행 기록과 결과 파일 sha256은 [run_manifest.json](run_manifest.json)에 있습니다.
