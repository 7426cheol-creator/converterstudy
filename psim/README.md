# PSIM 12.0.2 이식표

**상태: NOT_RUN_ENVIRONMENT.** 이 환경에는 PSIM이 없다. 여기 있는 것은 사람이 PSIM에서 회로를 **손으로**
다시 그리기 위한 표이고, PSIM으로 만든 파일(.psimsch)도, PSIM에서 얻은 결과도 없다. 기대값은 모두 Python 앱을
실제로 실행해 얻은 값이며, 각 시트에 CLI 명령·preset·앱 버전·request hash를 적었다.

교재 18장의 방침대로 PSIM 12.0.2에는 핵심 회로를 수동으로 재구성할 수 있는 이식표만 준다. 최신 PSIM에는
Python으로 schematic을 만드는 사례가 있지만 12.0.2에서 같은 기능을 확인하지 못했으므로 **그런 API를 가정하지
않는다.** 설치본의 도움말·samples에서 기능을 확인하면 그때 활용한다.

| 시트 | 회로 | 기준 실행 | 기대값 |
|---|---|---|---|
| [fl01_buck.md](fl01_buck.md) | 동기 Buck 48 → 12 V, 5 A | `FL01 buck_ccm --preset nominal` | 있음 |
| [fl08_dab.md](fl08_dab.md) | DAB 800 V ↔ 48 V, 1.5 kW / ratio mismatch 900 V ↔ 36 V | `FL08 sps_nominal --preset nominal`, `FL08 zero_power_mismatch --preset mismatch` | 있음 |
| [ex06_dab_width.md](ex06_dab_width.md) | DAB width modulation (w₁ = 0.7π) 900 V ↔ 36 V | `EX06 general_modulation --preset textbook` | 있음 |
| [fl09_llc.md](fl09_llc.md) | LLC 400 → 48 V, 다이오드 정류 | `FL09 switching_vs_fha --preset at_fr`, `--preset mid` | 있음 |
| [fl10_cllc.md](fl10_cllc.md) | CLLC n = 0.93, 850 V → 920 V, 두 분기 | FL10 미병합 | **자리만** (FHA 참고값만 교재에서) |
| [ex02_commutation.md](ex02_commutation.md) | half-bridge 비선형 Coss 전환 | `EX02 hb_constant_current --preset textbook` | 있음 |
| [ex07_cpl_filter.md](ex07_cpl_filter.md) | 입력 R-L-C + 이상 CPL 10 kW | `EX07 cpl_exact --preset c100u`, `--preset c1m` | 있음 |

기계가 읽는 상태는 [status.json](status.json)에 있다.

## 공통 약속

- **모델 수준.** 모든 시트는 앱과 같은 수준(대부분 C: 이상 스위치, dead time 0, 소자 모델 없음)으로 그린다.
  PSIM 스위치의 on-resistance, 다이오드 전압강하, snubber는 0 또는 가장 이상적인 값으로 둔다. 여기서 벗어나면
  기대값과 비교하지 않는다.
- **소자 이름.** 표의 "종류"는 일반 이름이다 (DC 전압원, MOSFET, Gating Block, 이상 변압기, 전압/전류 probe,
  제어 전원, 적분기, 수식 블록 등). PSIM 12.0.2 라이브러리의 정확한 이름과 메뉴 위치는 설치본에서 확인한다.
  Gating Block의 switching point 입력 형식(첫 점에서 on인지)도 설치본 도움말로 확인한다. 시트에는 on 구간을
  각도와 µs로 함께 적었다.
- **시간 원점.** 각 시트의 t = 0은 앱 series의 t = 0과 같게 잡았다 (시트마다 적음). PSIM 파형을 앱 CSV
  (`python -m convlab run ... --out DIR`의 CSV)와 겹쳐 보려면 같은 원점에서 저장한다.
- **Solver.** PSIM은 고정 time step이다. 스위칭 주기당 수천 step 이상을 권한다. 초기값을 주기해에 가깝게
  주면 짧게 돌려도 된다. 이상 인덕터(저항 0)는 기동 때 생긴 DC offset이 사라지지 않는다. 앱은 교재의
  zero-mean 정상해를 쓰므로, 초기 전류를 시트의 값으로 주거나 비교 전에 한 주기 평균을 뺀다.
- **비교.** 평균·RMS는 0.5 %, peak는 1 %, 시간·극점은 2 % 안이면 같은 모델로 본다 (고정 step·보간 오차 여유).
  벗어나면 회로 극성·gate 원점·초기값을 먼저 확인한다.
- **기록.** PSIM에서 실행하면 결과를 시트의 표에 적고 `status.json`의 해당 항목을 바꾼다
  (PSIM 버전, 날짜, PASS/FAIL, 벗어난 항목). 실행하지 않은 시트는 NOT_RUN_ENVIRONMENT로 남긴다.
