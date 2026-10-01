# Simscape 빌더 (DAB · CLLC · EX02 commutation · EX07 CPL)

**상태: NOT_RUN_ENVIRONMENT.** 이 환경에는 MATLAB, Simulink, Simscape가 없다. 여기 있는 `.m`·`.ssc` 파일은
모델을 **실행할 때 만들어 내는 스크립트**일 뿐이고, 한 번도 Simulink에서 빌드·시뮬레이션되지 않았다.
교재 18장의 규칙대로 `.m`을 썼다고 `.slx` 실행 완료로 표시하지 않는다. 저장소에는 `.slx`를 두지 않는다.
모델별 상태는 [`status.json`](status.json)에 기계가 읽을 수 있게 적혀 있다 (지금은 전부 NOT_RUN_ENVIRONMENT,
GNU Octave 8.4.0에서 `sc_run_all`의 환경 확인만 실제로 돌려 만든 파일).

## 실행 (Simulink + Simscape가 있는 MATLAB)

```matlab
% 1) 기대값: 저장소 루트에서 verification/run_octave_crosscheck.sh 를 먼저 돌려
%    matlab/results/python_export 에 Python export를 만든다 (또는 python -m convlab run ... --out 그 폴더)
cd matlab/simscape
results = sc_run_all();                 % 9개 모델: 빌드 → 시뮬레이션 → 비교 → status.json 갱신
res = build_dab('nominal');             % 하나만
res = build_cllc(2, 'fha', struct('run', false));   % 빌드와 .slx 저장만
```

- 생성물(`out/`): `ssc_build`로 만든 부품 라이브러리 `xcsc_lib.slx`와 모델 `.slx`. git이 무시한다.
- 결과(`results/<model>.json`): 비교 행(item, expected, simscape_value, 오차, tol, PASS/FAIL), 찾은 블록 경로,
  solver, MATLAB release. `status.json`의 해당 모델 항목이 PASS / FAIL / PENDING_EXPECTED / BUILD_ERROR /
  RUN_ERROR로 바뀐다. 사람이 결과를 확인한 뒤 커밋한다.

## 모델

| 모델 (`status.json`) | 빌더 | 회로 | 비교 대상 (허용오차) |
|---|---|---|---|
| `dab_nominal` | `build_dab('nominal')` | 1차 환산: 이상 bridge B1(±800 V) – ammeter – L 200 µH – ammeter – B2(±800 V, φ = export 값), 100 kHz, 12주기 | FL08 `sps_nominal/nominal`: Irms 2.019882 A, Ipk 2.094306 A, P₂ 1500 W (1e-3) |
| `dab_mismatch` | `build_dab('mismatch')` | ±900 V / ±600 V, φ = 0, L_m 2 mH를 v₂′ 쪽에 | FL08 `zero_power_mismatch/mismatch`: Irms_L 2.165064 A, Ipk_L 3.75 A, I_m rms, i₂ rms (1e-3), P = 0 (±0.5 W) |
| `cllc_branch1_fha`, `cllc_branch2_fha` | `build_cllc(b, 'fha')` | n = 0.93, 920/850 V, L_r1 = L_r2′ = 40 µH, C_r1 = C_r2′ = 28.144773 nF (export 입력), L_m 200 µH; 기본파 사인 전원(4V_link/π)과 R_ac′; FHA 해 주파수는 빌더가 export 입력으로 다시 구함 | FL10 `fix_n093/textbook`: \|H\| = n·V_bat/V_link = 1.006588, FHA 1차 RMS `I1_lo`/`I1_hi` (1e-3) |
| `cllc_branch1_switching`, `cllc_branch2_switching` | `build_cllc(b, 'switching')` | 같은 tank, 이상 bridge ±850 V, 다이오드 4개 정류, 환산 배터리 n·920 V, 10 ms (가장 느린 Floquet 승수 0.989) | FL10 `time_domain/textbook`: 배터리 전력 `P_td_lo`/`P_td_hi` (17.34 / 22.24 kW, 1e-3), 표 `t_td`의 1차 RMS (23.38 / 29.37 A), 같은 실행 안의 전력 균형 (5e-3) |
| `ex02_commutation` | `build_ex02_commutation()` | 고정 rail 800 V, 비선형 C(v) = C₀/√(1+v/V₀) 두 소자, 일정 전류 4 A를 node에 | EX02 `hb_constant_current/textbook`: rail 도달 286.606 ns, t_d 끝 node 전압, turn-on V_ds (1e-3) |
| `ex07_cpl_c100u`, `ex07_cpl_c1m` | `build_ex07_cpl(p)` | V_s 400 V – R 0.2 Ω – L 1 mH – C ∥ 이상 CPL 10 kW, 초기값 (P/V_e, V_e + 1 V) | EX07 `cpl_exact`: 극값에서 읽은 성장률·ω_d vs 극점 220.57 ± j3134.19 / −67.94 ± j991.24 (1e-2), 불안정 여부, 초기 상태 |

모델 수준은 교재 18장의 C(이상 스위칭)이다. bridge는 상보 gating·강한 DC link·dead time 없는 이상 bridge이고
소자 모델·Coss·gate가 없다. 그래서 이 모델로 ZVS나 소자 손실을 말하지 않는다. EX02는 E02의 첫 단계인
전하–전류 원리만 담는다.

## 블록을 추측하지 않는 방법

- **라이브러리 블록**(Inductor, Capacitor, Resistor, Electrical Reference, Solver Configuration)은 경로를 적어 두지
  않는다. `sc_find`가 설치된 라이브러리(`fl_lib`, `nesl_utility`, `ee_lib` 순)를 불러 `find_system`으로 **이름을
  실행 중에 찾는다**. 못 찾거나 한 라이브러리에 같은 이름이 여러 개면 BUILD_ERROR로 멈추고 무엇을 넘기면
  되는지 알려 준다 (`opts.blocks.inductor = '<Library Browser에서 복사한 경로>'`).
- **파라미터**는 dialog prompt를 정규식으로 찾는다(`'^Inductance'` 등). 초기값은 `<이름>_specify` 짝이 있는
  변수 중 prompt가 맞는 것을 찾아 High priority로 둔다. 맞는 것이 하나가 아니면 prompt 목록과 함께 멈춘다.
- **전원·ammeter·비선형 소자**는 `+xcsc/*.ssc`의 작은 Simscape 부품이다 (bridge, ammeter, vdc, sine, idc, coss,
  coss_start0, cpl, diode). 포트(+/p 왼쪽, −/n 오른쪽)를 파일에 직접 선언했으므로 극성을 추측하지 않는다.
  `ssc_build`로 `out/`에 라이브러리를 만들고, 블록은 `ComponentPath`(`xcsc.bridge` 등)로 찾는다.
- 로그는 이 부품들이 선언한 변수(`i`, `v`)만 읽는다. 라이브러리 블록의 내부 변수 이름은 쓰지 않는다.
- 라이브러리 R·L·C는 방향이 바뀌어도 크기 결과가 같다. EX07은 초기 인덕터 전류의 부호까지 비교 행으로
  확인한다.

## 첫 실행에서 사람이 확인할 것

1. `ver`로 Simulink, Simscape, (있으면) Simscape Electrical release를 확인한다. R2026a 이상에서는 SPS/powergui를
   쓰지 않는다 (이 빌더는 Foundation 블록과 자체 `.ssc` 부품만 쓴다).
2. `ssc_build` 출력과 `out/xcsc_lib.slx`의 블록, `results/*.json`의 `blocks`(실제로 찾은 경로)를 본다.
3. 생성된 모델을 열어 연결을 확인한다. 연결 목록은 각 빌더 머리말의 회로와 같아야 한다.
4. FAIL이면 모델을 고치기 전에 solver(daessc/ode23t, MaxStep)와 초기값을 먼저 본다.

빌더의 연결 목록과 후처리 코드는 Octave에서 Simulink API 호출을 기록하는 **흉내 stub**(저장소에 넣지 않음)으로만
점검했다. 이것은 Simscape 실행이 아니며, 결과로 쓰지 않는다.
