# Converter FAE Lab

교재 **「Infineon FAE · 기초 + 현업 전문가 심화 통합 v4.0」**과 맞춘 로컬 학습 시뮬레이터입니다.

- 기초 12개(FL01–FL12)와 심화 12개(EX01–EX12) 실습을 교재와 같은 ID로 연결합니다.
- 매 실험은 다음 순서로 진행합니다: **배울 것 → 바꿀 파라미터 → 먼저 예측 → 실행 → 파형에서 확인 → 왜 그런가 → 고객에게 어떻게 말할까**.

처음이면 **[START_HERE.md](START_HERE.md)** 부터 읽으세요. 설치, 첫 실습, 오늘 30분 과제가 있습니다.

> 구현 상태는 [docs/TRACEABILITY.md](docs/TRACEABILITY.md)(실습별 코드·테스트·실제 실행·검증·하드웨어 검증)와 [progress.json](progress.json)에 있습니다.
> 시뮬레이션 통과, 내 답, 학습 완료는 서로 다른 필드입니다. ‘면접 준비 완료’는 어디에도 자동으로 표시되지 않습니다.

## 설치와 실행

필요한 것은 Python 3.10 이상, numpy 1.24 이상, scipy 1.10 이상입니다(`pyproject.toml`, `requirements.txt`). 시험한 버전은 [docs/run_manifest.json](docs/run_manifest.json)에 적혀 있습니다.
로그인, 클라우드 DB, LLM API는 없고 서버는 127.0.0.1에서만 동작합니다.

```bash
./run.sh            # macOS/Linux: 첫 실행 때 .venv를 만들고 설치한 뒤 http://127.0.0.1:8765 를 엽니다
run.bat             # Windows
```

직접 설치하려면 `python -m pip install -e ".[test]"` 후 `python -m convlab serve`를 실행합니다.

교재 HTML은 이 공개 저장소에 커밋하지 않습니다. 갖고 있는 파일을 `textbook/`에 두면 실습 화면에서 해당 장이 열립니다([textbook/README.md](textbook/README.md)).

## 다시 실행하기

```bash
python -m convlab list                                    # 실습·실험·preset 목록
python -m convlab run FL01 buck_ccm --preset nominal --set L=50u --out out/   # 실험 하나 → json/csv/html
python -m convlab run-all --out results                   # 모든 기준 preset 실행 + results/run_manifest.json
python tools/gen_docs.py --results results                # 추적표·모델 카드·테스트 독립성·계약 재구성본·ERRATA.md 재생성
python -m pytest -q                                       # 단위·회귀·독립 경로 테스트
python verification/independent_reference_check.py       # 표준 라이브러리만으로 교재 기준값 재계산 (앱 코드를 import하지 않음)
verification/run_octave_crosscheck.sh                     # GNU Octave가 있으면: 교재 식으로 다시 쓴 MATLAB 언어 검산
(cd e2e && npm ci && node smoke.mjs --spawn)              # Node 22 + Playwright: 모든 실험을 실제 브라우저로 실행
```

CI(`.github/workflows/ci.yml`)도 같은 명령을 실행합니다. Python job은 reference check, pytest, run-all을, e2e job은 브라우저 스모크를 돌리며 비밀값은 쓰지 않습니다.

## 구조

| 위치 | 내용 |
|---|---|
| `src/convlab/engine/` | 정확한 구간 해석 엔진: 행렬지수 전파, guard 사건 탐지, 주기해(shooting·chord-Newton), 구간 선형 적분 |
| `src/convlab/labs/` | 실습 24개. 실험마다 회로, 파라미터, preset, 예측 질문, 실행, 판정, 해설, 면접 질문을 둔다. 공통 공진 모델은 `_resonant.py` |
| `src/convlab/reference/` | 교재 식을 그대로 옮긴 닫힌 식. 실험의 독립 비교 경로로 쓴다 |
| `src/convlab/model/` | 단위·파라미터 검증(범위 밖은 잘라내지 않고 거부), 결과·상태 스키마 |
| `src/convlab/web/` | 학습 화면: 회로 SVG, 파형 커서, 예측·답 기록, export |
| `tests/` | 실습별 테스트. 교재 기준값은 test에 숫자로 직접 적는다(앱이 낸 값을 기준값으로 되쓰지 않는다). 독립/회귀 구분은 [docs/TEST_INDEPENDENCE.md](docs/TEST_INDEPENDENCE.md) |
| `verification/` | 표준 라이브러리 기준값 검산, Octave 교차검증 실행 스크립트 |
| `matlab/` | Octave에서 실행한 교재 식 검산(`xc_*.m`)과 Simscape builder(`simscape/`, MATLAB 미실행) |
| `psim/` | PSIM 12.0.2 수동 재구성용 build sheet (미실행) |
| `e2e/` | 브라우저 스모크 테스트 |
| `docs/` | 추적표, 모델 카드, 테스트 독립성, run manifest, errata, 검증 범위, 알려진 한계 |

## 무엇을 실행했고 무엇을 실행하지 않았나

| 경로 | 상태 |
|---|---|
| Python 실습, pytest, run-all, 브라우저 스모크, 표준 라이브러리 기준값 검산 | 이 저장소에서 실행 ([docs/VERIFICATION.md](docs/VERIFICATION.md)) |
| GNU Octave 8.4 교차검증 (`matlab/xc_*.m`) | 실행, 결과는 `matlab/results/octave_crosscheck.json` |
| MATLAB / Simulink / Simscape | **NOT_RUN_ENVIRONMENT**: builder 스크립트만 있고 .slx나 결과는 없음 |
| PSIM 12.0.2 | **NOT_RUN_ENVIRONMENT**: build sheet만 있고 회로 파일은 없음 |
| 하드웨어 측정 | 없음. 모든 판정은 합성 모델 안의 verification이며 validation이 아님 |

## 원칙

- 모든 값에 단위, 출처(교재·가정·데이터시트), 모델 수준을 함께 표시합니다. 모델 수준은 A 해석/FHA, B 평균, C 이상 스위칭, D 비이상 commutation 합성입니다.
- 입력이 적용범위를 벗어나면 값을 잘라내지 않고 이유와 함께 거부합니다.
- 숫자를 기준값에 맞추려고 비물리적으로 보정하지 않습니다. 교재 수치와 다르면 [docs/ERRATA.md](docs/ERRATA.md)에 근거를 남깁니다.
- 의도된 실패(FL10 seed 등)는 지우지 않고 재현합니다.
- 합성 모델입니다. 실제 Infineon 제품 성능, 하드웨어 검증, EMI·수명·안전 적합성을 주장하지 않습니다. 자세한 내용은 [docs/KNOWN_LIMITS.md](docs/KNOWN_LIMITS.md)에 있습니다.
- 회사나 고객의 실제 자료는 필요 없고, 입력하지 않습니다.
