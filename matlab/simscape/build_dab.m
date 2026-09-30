function res = build_dab(preset, opts)
%BUILD_DAB Simscape model of the FL08 DAB at one preset, built at run time.
%
%   res = build_dab('nominal')    800 V / 48 V, n = 50/3, 1.5 kW (exported phi)
%   res = build_dab('mismatch')   900 V / 36 V, phi = 0, Lm = 2 mH
%   res = build_dab(preset, opts) options: see sc_opts
%
%   Circuit, primary-referred (textbook ch.11 and E06, model level C):
%     B1 --A1-- L --x--A2-- B2        B1 = xcsc.bridge(V1, w = pi, phi = 0)
%                   |                 B2 = xcsc.bridge(V2' = n VL, w = pi, phi)
%                   A3 -- LM (mismatch preset only)
%     B1-, B2-, LM- to ground
%   The bridges are ideal (complementary gating, stiff DC links, no dead
%   time, no device capacitance): the run checks currents and power, not ZVS.
%   Analysis of the last of 12 periods: the DC offset of the ideal inductor
%   currents is removed (textbook zero-mean steady state), then Irms, Ipk and
%   P2 = (1/T) int v2' i2 dt are compared with the Python export.
%   Library blocks (Inductor, Electrical Reference, Solver Configuration) are
%   looked up by name at run time; the sources and ammeters are the +xcsc
%   components, whose ports are declared in their .ssc files.

if nargin < 2
  opts = struct();
end
opts = sc_opts(opts);
t_start = tic;
switch preset
  case 'nominal'
    key = {'FL08', 'sps_nominal', 'nominal'};
  case 'mismatch'
    key = {'FL08', 'zero_power_mismatch', 'mismatch'};
  otherwise
    error('build_dab: preset must be ''nominal'' or ''mismatch''');
end
res = sc_result(['dab_' preset], sprintf('build_dab(''%s'')', preset), ...
                sprintf('python -m convlab run %s %s --preset %s', key{:}));
res.env = sc_env();
if ~res.env.ok
  res.reason = res.env.reason;
  res = sc_finish(res, opts, t_start);
  return;
end

mdl = ['xc_dab_' preset];
try
  d = xc_load(opts.export_dir, key{:});
  n = xc_get(d, 'input', 'Np') / xc_get(d, 'input', 'Ns');
  V1 = xc_get(d, 'input', 'VH');
  V2 = n * xc_get(d, 'input', 'VL');
  L = xc_get(d, 'input', 'L');
  fs = xc_get(d, 'input', 'fs');
  if strcmp(preset, 'nominal')
    phi = xc_get(d, 'metric', 'phi');
    Lm = 0;
  else
    phi = xc_get(d, 'input', 'phi');
    Lm = xc_get(d, 'input', 'Lm');
  end
  T = 1 / fs;

  lib = sc_components(opts);
  blk_L = sc_find('inductor', {'Inductor'}, opts);
  blk_ref = sc_find('electrical_reference', {'Electrical Reference'}, opts);
  blk_cfg = sc_find('solver_configuration', {'Solver Configuration'}, opts);
  res.blocks = struct('inductor', blk_L, 'electrical_reference', blk_ref, ...
                      'solver_configuration', blk_cfg, 'components', lib);

  sc_new(mdl);
  b1 = sc_place(mdl, sc_comp(lib, 'bridge'), 'B1', 0, 1);
  a1 = sc_place(mdl, sc_comp(lib, 'ammeter'), 'A1', 1, 0);
  l1 = sc_place(mdl, blk_L, 'L1', 2, 0);
  a2 = sc_place(mdl, sc_comp(lib, 'ammeter'), 'A2', 3, 0);
  b2 = sc_place(mdl, sc_comp(lib, 'bridge'), 'B2', 4, 1);
  gnd = sc_place(mdl, blk_ref, 'GND', 2, 3);
  cfg = sc_place(mdl, blk_cfg, 'CFG', 0, 3);
  sc_set(b1, 'V', V1, 'V');
  sc_set(b1, 'fs', fs, 'Hz');
  sc_set(b1, 'w', pi, '1');
  sc_set(b1, 'phi', 0, '1');
  sc_set(b2, 'V', V2, 'V');
  sc_set(b2, 'fs', fs, 'Hz');
  sc_set(b2, 'w', pi, '1');
  sc_set(b2, 'phi', phi, '1');
  sc_set(l1, '^Inductance', L, 'H');

  sc_wire(mdl, sc_port(b1, 'p'), sc_port(a1, 'p'));
  sc_wire(mdl, sc_port(a1, 'n'), sc_port(l1, 'p'));
  sc_wire(mdl, sc_port(l1, 'n'), sc_port(a2, 'p'));
  sc_wire(mdl, sc_port(a2, 'n'), sc_port(b2, 'p'));
  ground = [sc_port(b1, 'n'), sc_port(b2, 'n'), sc_port(gnd, 'one'), sc_port(cfg, 'one')];
  if Lm > 0
    a3 = sc_place(mdl, sc_comp(lib, 'ammeter'), 'A3', 3, 2);
    lm = sc_place(mdl, blk_L, 'LM', 4, 2);
    sc_set(lm, '^Inductance', Lm, 'H');
    sc_wire(mdl, sc_port(l1, 'n'), sc_port(a3, 'p'));
    sc_wire(mdl, sc_port(a3, 'n'), sc_port(lm, 'p'));
    ground(end + 1) = sc_port(lm, 'n');
  end
  sc_wire(mdl, ground);
  res.solver = sc_solver(mdl, 12 * T, T / 400);
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
  [t1, i1] = sc_series(simlog, 'A1', 'i', 'A');
  [t2, i2] = sc_series(simlog, 'A2', 'i', 'A');
  [tv, v2] = sc_series(simlog, 'B2', 'v', 'V');
  tg = linspace(11 * T, 12 * T, 20001).';
  ig = sc_grid(t1, i1, tg);
  ig = ig - trapz(tg, ig) / T;               % zero-mean steady state
  i2g = sc_grid(t2, i2, tg);
  i2g = i2g - trapz(tg, i2g) / T;
  Irms = sqrt(trapz(tg, ig .^ 2) / T);
  Ipk = max(abs(ig));
  % power on the raw log samples: v2' jumps at the edges, i2 is continuous
  [tw, v2w] = sc_window(tv, v2, 11 * T, 12 * T);
  i2w = sc_grid(t2, i2, tw);
  i2w = i2w - trapz(tw, i2w) / T;
  P2 = trapz(tw, v2w .* i2w) / T;
  m = sprintf('Simscape (%s), last of 12 periods, DC offset removed', res.solver);
  rows = {};
  if strcmp(preset, 'nominal')
    rows{end + 1} = sc_row('SC.dab_nominal.Irms', d, 'Irms', 'A', xc_get(d, 'metric', 'Irms'), Irms, 1e-3, 'rel', m);
    rows{end + 1} = sc_row('SC.dab_nominal.Ipk', d, 'Ipk', 'A', xc_get(d, 'metric', 'Ipk'), Ipk, 1e-3, 'rel', m);
    rows{end + 1} = sc_row('SC.dab_nominal.P2', d, 'P2 = (1/T) int v2'' i2 dt', 'W', xc_get(d, 'metric', 'P_pwl'), P2, 1e-3, 'rel', m);
  else
    [t3, i3] = sc_series(simlog, 'A3', 'i', 'A');
    i3g = sc_grid(t3, i3, tg);
    i3g = i3g - trapz(tg, i3g) / T;
    rows{end + 1} = sc_row('SC.dab_mismatch.Irms_L', d, 'Irms_L', 'A', xc_get(d, 'metric', 'Irms_L'), Irms, 1e-3, 'rel', m);
    rows{end + 1} = sc_row('SC.dab_mismatch.Ipk_L', d, 'Ipk_L', 'A', xc_get(d, 'metric', 'Ipk_L'), Ipk, 1e-3, 'rel', m);
    rows{end + 1} = sc_row('SC.dab_mismatch.Im_rms', d, 'magnetizing rms', 'A', xc_get(d, 'metric', 'Im_rms'), sqrt(trapz(tg, i3g .^ 2) / T), 1e-3, 'rel', m);
    rows{end + 1} = sc_row('SC.dab_mismatch.I2_rms', d, 'transformer current rms', 'A', xc_get(d, 'metric', 'I2_rms'), sqrt(trapz(tg, i2g .^ 2) / T), 1e-3, 'rel', m);
    rows{end + 1} = sc_row('SC.dab_mismatch.P', d, 'transferred power', 'W', xc_get(d, 'metric', 'P'), P2, 0.5, 'abs', m);
  end
  res.rows = rows;
  res.status = 'RAN';
catch err
  res.status = 'RUN_ERROR';
  res.reason = err.message;
end
res = sc_finish(res, opts, t_start);
end
