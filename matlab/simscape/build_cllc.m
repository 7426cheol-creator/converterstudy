function res = build_cllc(branch, mode, opts)
%BUILD_CLLC Simscape model of the FL10 CLLC, modification A (n = 0.93), at the
%   920 V battery / 850 V link corner, on one of its two FHA solutions.
%
%   res = build_cllc(1, 'fha')         136.099 kHz, first-harmonic circuit
%   res = build_cllc(2, 'fha')         147.061 kHz
%   res = build_cllc(b, 'switching')   ideal bridge and diode rectifier
%   res = build_cllc(b, mode, opts)    options: see sc_opts
%
%   Primary-referred tank (textbook ch.13, referred symmetry):
%     Lr1 = Lr2' = 40 uH, Cr1 = Cr2' = 28.1448 nF, Lm = 200 uH, 11 kW,
%     Rac' = (8/pi^2) n^2 Vbat^2 / P, referred battery n*Vbat.
%     SRC --A1-- Lr1 -- Cr1 --m-- Cr2' -- Lr2' --r-- load
%                             |
%                             Lm -- ground
%   'fha'       SRC = xcsc.sine with the fundamental of the +-Vlink bridge
%               (4 Vlink/pi), load = Rac' through ammeter A2. Checked against
%               the textbook: |H| = n Vbat/Vlink = 1.006588 and the FHA primary
%               rms 14.390 A / 14.765 A.
%   'switching' SRC = xcsc.bridge (+-Vlink, 50 %), load = four xcsc.diode into
%               xcsc.vdc(n Vbat) through ammeter A3. The expected values come
%               from FL10 / EX05, not merged yet: the measured values are kept
%               as PENDING_EXPECTED, and the only pass/fail row is the power
%               balance of the lossless tank (textbook E05 energy identity).
%   Analysis over the last 20 of the simulated periods (0.6 ms FHA, 1 ms
%   switching; slowest FHA mode decays with about 33 us).

if nargin < 3
  opts = struct();
end
opts = sc_opts(opts);
t_start = tic;
f_tb = [136.099e3, 147.061e3];               % textbook ch.13 FHA solutions
irms_tb = [14.390, 14.765];                  % textbook FHA primary series rms
if ~any(branch == [1, 2]) || ~any(strcmp(mode, {'fha', 'switching'}))
  error('build_cllc(branch, mode): branch 1 or 2, mode ''fha'' or ''switching''');
end
name = sprintf('cllc_branch%d_%s', branch, mode);
if strcmp(mode, 'fha')
  source = 'textbook ch.13 FHA values: |H| = 1.006588, primary rms 14.390 / 14.765 A';
else
  source = 'FL10 / EX05 export (pending); power balance inside the same run';
end
res = sc_result(name, sprintf('build_cllc(%d, ''%s'')', branch, mode), source);
res.env = sc_env();
if ~res.env.ok
  res.reason = res.env.reason;
  res = sc_finish(res, opts, t_start);
  return;
end

n = 0.93;
Vbat = 920;
Vlink = 850;
P = 11e3;
Lr = 40e-6;
Cr = 28.1448e-9;
Lm = 200e-6;
Rac = 8 / pi ^ 2 * n ^ 2 * Vbat ^ 2 / P;
f = f_tb(branch);
T = 1 / f;
if strcmp(mode, 'fha')
  t_stop = 0.6e-3;
else
  t_stop = 1.0e-3;
end
src = struct('lab', 'FL10', 'experiment', ['simscape_' mode], 'preset', sprintf('n093_920_850_branch%d', branch), ...
             'file', 'textbook ch.13');
mdl = ['xc_' name];
try
  lib = sc_components(opts);
  blk_L = sc_find('inductor', {'Inductor'}, opts);
  blk_C = sc_find('capacitor', {'Capacitor'}, opts);
  blk_ref = sc_find('electrical_reference', {'Electrical Reference'}, opts);
  blk_cfg = sc_find('solver_configuration', {'Solver Configuration'}, opts);
  res.blocks = struct('inductor', blk_L, 'capacitor', blk_C, 'electrical_reference', blk_ref, ...
                      'solver_configuration', blk_cfg, 'components', lib);

  sc_new(mdl);
  if strcmp(mode, 'fha')
    s1 = sc_place(mdl, sc_comp(lib, 'sine'), 'SRC', 0, 1);
    sc_set(s1, 'Vpk', 4 * Vlink / pi, 'V');
    sc_set(s1, 'f', f, 'Hz');
  else
    s1 = sc_place(mdl, sc_comp(lib, 'bridge'), 'SRC', 0, 1);
    sc_set(s1, 'V', Vlink, 'V');
    sc_set(s1, 'fs', f, 'Hz');
    sc_set(s1, 'w', pi, '1');
    sc_set(s1, 'phi', 0, '1');
  end
  a1 = sc_place(mdl, sc_comp(lib, 'ammeter'), 'A1', 1, 0);
  l1 = sc_place(mdl, blk_L, 'LR1', 2, 0);
  c1 = sc_place(mdl, blk_C, 'CR1', 3, 0);
  lm = sc_place(mdl, blk_L, 'LM', 4, 1);
  c2 = sc_place(mdl, blk_C, 'CR2', 5, 0);
  l2 = sc_place(mdl, blk_L, 'LR2', 6, 0);
  gnd = sc_place(mdl, blk_ref, 'GND', 3, 3);
  cfg = sc_place(mdl, blk_cfg, 'CFG', 0, 3);
  sc_set(l1, '^Inductance', Lr, 'H');
  sc_set(l2, '^Inductance', Lr, 'H');
  sc_set(lm, '^Inductance', Lm, 'H');
  sc_set(c1, '^Capacitance', Cr, 'F');
  sc_set(c2, '^Capacitance', Cr, 'F');
  sc_wire(mdl, sc_port(s1, 'p'), sc_port(a1, 'p'));
  sc_wire(mdl, sc_port(a1, 'n'), sc_port(l1, 'p'));
  sc_wire(mdl, sc_port(l1, 'n'), sc_port(c1, 'p'));
  sc_wire(mdl, sc_port(c1, 'n'), sc_port(lm, 'p'), sc_port(c2, 'p'));   % node m
  sc_wire(mdl, sc_port(c2, 'n'), sc_port(l2, 'p'));
  ground = [sc_port(s1, 'n'), sc_port(lm, 'n'), sc_port(gnd, 'one'), sc_port(cfg, 'one')];
  if strcmp(mode, 'fha')
    blk_R = sc_find('resistor', {'Resistor'}, opts);
    res.blocks.resistor = blk_R;
    a2 = sc_place(mdl, sc_comp(lib, 'ammeter'), 'A2', 7, 0);
    rl = sc_place(mdl, blk_R, 'RAC', 8, 1);
    sc_set(rl, '^Resistance', Rac, 'Ohm');
    sc_wire(mdl, sc_port(l2, 'n'), sc_port(a2, 'p'));                 % node r
    sc_wire(mdl, sc_port(a2, 'n'), sc_port(rl, 'p'));
    ground(end + 1) = sc_port(rl, 'n');
  else
    dio = sc_comp(lib, 'diode');
    d1 = sc_place(mdl, dio, 'D1', 7, 0);     % r -> POS
    d2 = sc_place(mdl, dio, 'D2', 7, 2);     % ground -> POS
    d3 = sc_place(mdl, dio, 'D3', 8, 0);     % NEG -> r
    d4 = sc_place(mdl, dio, 'D4', 8, 2);     % NEG -> ground
    a3 = sc_place(mdl, sc_comp(lib, 'ammeter'), 'A3', 9, 0);
    bt = sc_place(mdl, sc_comp(lib, 'vdc'), 'BAT', 10, 1);
    sc_set(bt, 'V', n * Vbat, 'V');
    sc_wire(mdl, sc_port(l2, 'n'), sc_port(d1, 'p'), sc_port(d3, 'n'));   % node r
    sc_wire(mdl, sc_port(d1, 'n'), sc_port(d2, 'n'), sc_port(a3, 'p'));   % POS
    sc_wire(mdl, sc_port(a3, 'n'), sc_port(bt, 'p'));
    sc_wire(mdl, sc_port(bt, 'n'), sc_port(d3, 'p'), sc_port(d4, 'p'));   % NEG
    ground = [ground, sc_port(d2, 'p'), sc_port(d4, 'n')];
  end
  sc_wire(mdl, ground);
  res.solver = sc_solver(mdl, t_stop, T / 400);
  if opts.save
    if exist(opts.out_dir, 'dir') ~= 7
      mkdir(opts.out_dir);
    end
    save_system(mdl, fullfile(opts.out_dir, [mdl '.slx']));
  end
catch err
  res.status = 'BUILD_ERROR';
  res.reason = err.message;
  res = sc_finish(res, opts, t_start);
  return;
end
if ~opts.run
  res.status = 'BUILT_NOT_RUN';
  res = sc_finish(res, opts, t_start);
  return;
end

try
  out = sim(mdl, 'ReturnWorkspaceOutputs', 'on');
  simlog = out.get('simlog');
  if isempty(simlog)
    error('no Simscape log in the simulation output (SimscapeLogType)');
  end
  Nw = 20;
  tg = linspace(t_stop - Nw * T, t_stop, 40001).';
  [ts, vs] = sc_series(simlog, 'SRC', 'v', 'V');
  [t1, i1] = sc_series(simlog, 'A1', 'i', 'A');
  vsg = sc_grid(ts, vs, tg);
  i1g = sc_grid(t1, i1, tg);
  I1rms = sqrt(trapz(tg, i1g .^ 2) / (Nw * T));
  % source power on the raw log samples (the bridge voltage jumps at its edges)
  [tw, vsw] = sc_window(ts, vs, tg(1), tg(end));
  Pin = trapz(tw, vsw .* sc_grid(t1, i1, tw)) / (Nw * T);
  rows = {};
  m = sprintf('Simscape (%s), last %d of %.0f periods', res.solver, Nw, t_stop / T);
  if strcmp(mode, 'fha')
    [t2, i2] = sc_series(simlog, 'A2', 'i', 'A');
    vrg = Rac * sc_grid(t2, i2, tg);
    ph = exp(-1i * 2 * pi * f * tg);
    Vs1 = 2 / (Nw * T) * trapz(tg, vsg .* ph);      % fundamental phasors
    Vr1 = 2 / (Nw * T) * trapz(tg, vrg .* ph);
    rows{end + 1} = sc_row(sprintf('SC.cllc_b%d_fha.H', branch), src, '|H| = |V_Rac| / |V_src| (fundamental)', '', ...
                           n * Vbat / Vlink, abs(Vr1) / abs(Vs1), 1e-3, 'rel', m, 'textbook ch.13: n Vbat/Vlink at the FHA solution');
    rows{end + 1} = sc_row(sprintf('SC.cllc_b%d_fha.I1rms', branch), src, 'FHA primary series rms', 'A', ...
                           irms_tb(branch), I1rms, 1e-3, 'rel', m, 'textbook ch.13 FHA primary series rms');
  else
    [t3, i3] = sc_series(simlog, 'A3', 'i', 'A');
    Pout = n * Vbat * trapz(tg, sc_grid(t3, i3, tg)) / (Nw * T);
    rows{end + 1} = sc_row(sprintf('SC.cllc_b%d_sw.power_balance', branch), src, 'battery power vs bridge power (lossless tank)', 'W', ...
                           Pin, Pout, 5e-3, 'rel', [m '; diodes Ron 1 mOhm'], 'same simulation: bridge power (1/T) int v1 i1 dt');
    rows{end + 1} = sc_row(sprintf('SC.cllc_b%d_sw.Pout', branch), src, 'battery power', 'W', NaN, Pout, NaN, 'abs', m, 'FL10 / EX05 export (pending)');
    rows{end + 1} = sc_row(sprintf('SC.cllc_b%d_sw.I1rms', branch), src, 'primary series rms', 'A', NaN, I1rms, NaN, 'abs', m, 'FL10 / EX05 export (pending)');
  end
  res.rows = rows;
  res.status = 'RAN';
catch err
  res.status = 'RUN_ERROR';
  res.reason = err.message;
end
res = sc_finish(res, opts, t_start);
end
