# PSIM 이식표 — FL08 DAB (800 V ↔ 48 V, 1.5 kW / ratio mismatch)

**상태: NOT_RUN_ENVIRONMENT** — PSIM 12.0.2가 이 환경에 없다. 이 시트는 손으로 회로를 그리기 위한 자료이고 PSIM 결과가 아니다. PSIM 12.0.2의 schematic 생성 API는 가정하지 않는다.

기준 실행:
- nominal: `python -m convlab run FL08 sps_nominal --preset nominal` (app 4.0.0, request_hash `2efac54d4fabc0e3`, 실행 2026-09-30T19:53:53, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 A (닫힌 식) + C (구간 적분·스위칭 해))
- mismatch: `python -m convlab run FL08 zero_power_mismatch --preset mismatch` (app 4.0.0, request_hash `f6c228f7feabaeea`, 실행 2026-09-30T19:53:53, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 A + C)

교재 11장 worked example(모듈 1개)과 "0 W인데도 왜 뜨거운가"의 mismatch 예제다. 앱은 1차 환산 회로를 푼다.
PSIM에서는 실제 변압기(50:3)와 LV bridge를 그려 같은 1차 환산값이 나오는지 본다.

## 1. 회로

| 소자 | 종류 | 노드 (+ → −) | nominal | mismatch | 값의 출처 |
|---|---|---|---|---|---|
| V_H | DC 전압원 | HV+ → HV− | 800 V | 900 V | 입력 `VH` |
| S1, S2 | MOSFET leg A (이상, 역병렬 diode 이상) | S1: HV+ → A, S2: A → HV− | | | |
| S3, S4 | MOSFET leg B | S3: HV+ → B, S4: B → HV− | | | |
| L | 인덕터 (1차 직렬, 외부 L + leakage) | A → T1 | 200 µH | 200 µH | 입력 `L` |
| TR | 이상 변압기 1상 2권선, N_p:N_s = 50:3 | 1차 T1(점) – B, 2차 T2(점) – T3 | n = 16.6667 | n = 16.6667 | 입력 `Np`, `Ns` |
| L_m | 인덕터 (자화) | T1 → B (변압기 1차 단자, v₂′ 쪽) | 없음 | 2 mH | 입력 `Lm` (nominal preset은 0 = 없음) |
| S5, S6 | MOSFET leg C (2차) | S5: LV+ → T2, S6: T2 → LV− | | | |
| S7, S8 | MOSFET leg D (2차) | S7: LV+ → T3, S8: T3 → LV− | | | |
| V_L | DC 전압원 (배터리, 이상) | LV+ → LV− | 48 V | 36 V | 입력 `VL` |

1차와 2차는 절연이다. PSIM이 요구하면 각 쪽에 ground를 따로 둔다. 권선 저항 `R` = 0, dead time 없음
(`td_hv` 150 ns, `td_lv` 50 ns는 앱의 commutation screen 표에만 쓰인다). 1차 환산 V₂ = n·V_L = 800 V (nominal),
600 V (mismatch).

## 2. Gate timing (f_s = 100 kHz, T = 10 µs, dead time 0)

v₁ = V_H(a − b), v₂′ = n·V_L(c − d), a…d = 각 leg 상측 on (1) / off (0). 하측은 상측의 보수다.
교재 11장 표와 같이 v₁은 [0°, 180°)에서 +V₁, v₂′는 [φ, φ + 180°)에서 +V₂이다 (primary가 φ만큼 선행 → HV→LV 양의 전력).

| leg | 상측 on [deg] | 상측 on [µs] (nominal) | mismatch (φ = 0) |
|---|---|---|---|
| A (S1; S2 보수) | 0 → 180 | 0 → 5 | 0 → 5 |
| B (S3; S4 보수) | 180 → 360 | 5 → 10 | 5 → 10 |
| C (S5; S6 보수) | φ → φ + 180 = 18.84875 → 198.84875 | 0.5235765 → 5.5235765 | 0 → 5 |
| D (S7; S8 보수) | φ + 180 → φ + 360 = 198.84875 → 378.84875 (= 18.84875) | 5.5235765 → 10.5235765 (= 0.5235765) | 5 → 10 |

φ = 0.3289728 rad = 18.84875° (export `phi`, `dt_T` = φ/2π = 0.05235765). d = φ/π = 0.1047153과 Δt/T를 섞지 않는다.

## 3. 시뮬레이션 설정

| 항목 | 값 |
|---|---|
| time step | 1 ns (주기당 10 000 step) |
| total time | 100 µs (10주기) |
| 저장 시작 | 80 µs (마지막 2주기) |
| 초기값 nominal | i_L(0) = −2.094306 A (export `i0`, v₁ 상승 edge의 전류) |
| 초기값 mismatch | i_L(0) = −3.75 A, i_Lm(0) = −0.75 A (series `iL`, `im`의 t = 0 값) |

L과 L_m에 저항이 없으므로 기동 때 생긴 DC offset은 사라지지 않는다. 초기 전류를 줄 수 없으면 한 주기 평균을 빼고
비교한다 (교재: 정상 기준해는 zero-DC 조건으로 고른다).

## 4. Probe ↔ 앱 series

| PSIM probe | 측정 | 앱 series |
|---|---|---|
| I(L) | 1차 직렬 전류, A → T1 | `iL` |
| V(A) − V(B) | 1차 bridge 출력 v₁ | `v1` |
| V(T1) − V(B) | 변압기 1차 전압 = v₂′ | `v2` |
| V(A) − V(T1) | L 전압 v_L = v₁ − v₂′ | `vL` |
| I(2차 권선), T2 → S5/S6 방향 | 실제 2차 전류 i_s = n·i₂ | `i2_act` |
| LV bridge → V_L+ 전류 | LV DC 포트 전류 | `idc2` (평균 = `Io_dc`) |
| I(L_m) (mismatch) | 자화 전류 | `im` |
| I(L) − I(L_m) (mismatch) | 변압기 1차 환산 전류 | `i2` |

부호: t = 0에서 nominal `iL` = −2.094 A, `i2_act` = −34.91 A다. 반대로 나오면 변압기 점 방향이나 probe 방향을 확인한다.

## 5. 기대값 (Python 실제 실행)

`python -m convlab run FL08 sps_nominal --preset nominal` (app 4.0.0, request_hash `2efac54d4fabc0e3`, 실행 2026-09-30T19:53:53, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 A (닫힌 식) + C (구간 적분·스위칭 해))

| metric | 뜻 | 값 | 단위 |
|---|---|---|---|
| `n` | 권선비 n = N_p/N_s | 16.66667 |  |
| `V2` | LV 전압의 1차 환산 V₂ = n·V_L | 800 | V |
| `phi` | outer phase φ | 0.3289728 | rad |
| `phi_deg` | φ [deg] | 18.84875 | deg |
| `P_pwl` | 전달전력 (구간 적분, 2차 포트) | 1500 | W |
| `Irms` | 인덕터 RMS (1차) | 2.019882 | A |
| `Ipk` | 인덕터 peak | 2.094306 | A |
| `i0` | i(0): 1차 상승 edge 전류 | -2.094306 | A |
| `iphi` | 2차 상승 edge 전류 (1차 환산) | 2.094306 | A |
| `Is_rms` | 2차 AC RMS (실제 = n·I₂,rms) | 33.66469 | A |
| `Io_dc` | LV DC 출력전류 (모듈당) = P/V_L | 31.25 | A |
| `Isw_rms` | 1차 스위치 1개 RMS | 1.428272 | A |
| `Pmax` | SPS 최대전력 V₁V₂/(8 f_s L) | 4000 | W |

`python -m convlab run FL08 zero_power_mismatch --preset mismatch` (app 4.0.0, request_hash `f6c228f7feabaeea`, 실행 2026-09-30T19:53:53, 결과 상태 PASS_WITHIN_MODEL, 모델 수준 A + C)

| metric | 뜻 | 값 | 단위 |
|---|---|---|---|
| `P` | 전달전력 | 0 | W |
| `Irms_L` | L 순환전류 RMS (L_m 제외) | 2.165064 | A |
| `Ipk_L` | L 순환전류 peak | 3.75 | A |
| `Im_rms` | 여자전류 RMS (L_m) | 0.4330127 | A |
| `Iprim_rms` | 1차 권선 RMS (i_L, L_m 포함 모델) | 2.165064 | A |
| `I2_rms` | 2차 환산 RMS i_L − i_m | 1.732051 | A |

교재 11장 인쇄값: φ = 0.328973 rad = 18.8488°, I_pk 2.09431 A, I_rms 2.01988 A, 2차 AC RMS 33.6647 A, DC 31.25 A;
mismatch I_pk 3.75 A, I_rms 2.16506 A. 두 모듈(interleaving)은 이 시트에 없다.

## 6. 비교

마지막 한 주기에서 I_rms, I_pk, 2차 RMS, LV DC 평균 전류, 전달전력(1/T)∫v₂′i₂dt를 구해 README 허용 범위로 비교한다.
mismatch는 전력이 0 W (±1 W)인지, L 전류가 ±3.75 A 삼각파인지 본다.

## 7. 이 회로가 말하지 않는 것

이상 스위치라 ZVS·dead time·Coss 전하·스위칭 손실은 볼 수 없다 (교재: "ideal switch만 있는 회로는 ZVS를 입증할
수 없다"). 소자 모델을 넣으면 기대값과 직접 비교하지 않고 EX02·EX06 수준의 별도 판단을 한다.
