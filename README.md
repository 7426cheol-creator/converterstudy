# Converter FAE Lab

교재 **「Infineon FAE · 기초 + 현업 전문가 심화 통합 v4.0」** 과 정합된 로컬 학습 시뮬레이터입니다.
기초 12개(FL01–FL12)와 심화 12개(EX01–EX12) 실습을 같은 ID로 연결하고, 매 실험을
**배울 것 → 바꿀 파라미터 → 먼저 예측 → 실행 → 파형에서 확인 → 왜 그런가 → 고객에게 어떻게 말할까** 순서로 진행합니다.

> 진행 중인 작업입니다. 구현 상태는 앱의 ‘추적표’와 `docs/`에 실습별로 기록됩니다.
> 시뮬레이션 통과, 나의 답, 학습 완료 표시는 서로 다른 필드이며 ‘면접 준비 완료’는 자동으로 표시되지 않습니다.

## 실행

```bash
./run.sh            # macOS/Linux: 첫 실행 시 .venv 생성·설치 후 http://127.0.0.1:8765 열기
run.bat             # Windows
```

직접 설치: `python -m pip install -e ".[test]"` 후 `python -m convlab serve`.
필요한 것: Python 3.10+, numpy, scipy (로그인·클라우드·LLM API 없음, 127.0.0.1에서만 동작).

```bash
python -m convlab list                                   # 실습·실험·preset 목록
python -m convlab run FL01 buck_ccm --preset nominal --set L=50u --out out/
python -m convlab run-all --out results                 # 모든 기준 preset 실행 + run manifest
python -m pytest -q                                      # 테스트
```

## 첫 실습 (신규 학습자)

FL01 → ‘Buck 48→12 V’ → 기준 실행 전에 ΔI·peak·RMS를 손으로 적고 실행 → 파형 위에서 커서를 움직여 회로의
ON/OFF 도통 경로 확인 → **L 100 → 50 µH** 변경 전에 리플·평균·peak 예측 → 실행해 비교.

## 원칙

- 모든 값은 단위·출처(교재/가정)·모델 수준(A 해석/FHA, B 평균, C 이상 스위칭, D 비이상 commutation 합성)과 함께 표시합니다.
- 입력이 적용범위 밖이면 값을 잘라내지 않고 이유와 함께 거부합니다.
- 숫자를 기준값에 맞추는 비물리적 보정을 하지 않습니다. 교재 수치와 다르면 `docs/ERRATA.md`에 근거를 남깁니다.
- 합성 모델입니다. 실제 Infineon 제품 성능, 하드웨어 검증, EMI/수명/안전 적합성을 주장하지 않습니다.
