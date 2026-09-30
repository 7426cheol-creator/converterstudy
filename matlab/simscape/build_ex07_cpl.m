function res = build_ex07_cpl(preset, opts)
%BUILD_EX07_CPL Simscape model of the EX07 constant-power load behind an
%   R-L-C input filter (textbook E07), built at run time.
%
%   res = build_ex07_cpl('c100u')   C = 100 uF, unstable (poles 220.57 +- j3134.19)
%   res = build_ex07_cpl('c1m')     C = 1 mF, stable (poles -67.94 +- j991.24)
%   res = build_ex07_cpl(preset, opts)   options: see sc_opts
%
%   Circuit: VS (xcsc.vdc Vs) -- R -- A1 -- L --x-- C || LOAD (xcsc.cpl P)
%   Library Resistor / Inductor / Capacitor; initial targets i_L = Ie and
%   v_C = Ve + dv0 (the export's perturbation), set as high-priority targets
%   found at run time. Ideal CPL i = P/v (infinite bandwidth).
%   The run starts at the perturbed equilibrium and ends at the export's
%   t_exit (|dv| = dv_valid Ve) or t_end. Growth rate and damped frequency
%   come from the extrema of v - Ve with |v - Ve| < 5 % Ve and are compared
%   with the exported linear poles; the initial state is checked as well
%   (it also verifies the sign convention of the library inductor current).

if nargin < 2
  opts = struct();
end
opts = sc_opts(opts);
t_start = tic;
if ~any(strcmp(preset, {'c100u', 'c1m'}))
  error('build_ex07_cpl: preset must be ''c100u'' or ''c1m''');
end
res = sc_result(['ex07_cpl_' preset], sprintf('build_ex07_cpl(''%s'')', preset), ...
                sprintf('python -m convlab run EX07 cpl_exact --preset %s', preset));
res.env = sc_env();
if ~res.env.ok
  res.reason = res.env.reason;
  res = sc_finish(res, opts, t_start);
  return;
end

mdl = ['xc_ex07_cpl_' preset];
try
  d = xc_load(opts.export_dir, 'EX07', 'cpl_exact', preset);
  Vs = xc_get(d, 'input', 'Vs');
  R = xc_get(d, 'input', 'R');
  L = xc_get(d, 'input', 'L');
  C = xc_get(d, 'input', 'C');
  P = xc_get(d, 'input', 'P');
  dv0 = xc_get(d, 'input', 'dv0');
  vmin = xc_get(d, 'input', 'v_min_frac') * Vs;
  Ve = xc_get(d, 'metric', 'Ve');
  Ie = P / Ve;
  wd = xc_get(d, 'metric', 'pole_im');
  if isKey(d.metrics, 't_exit')
    t_stop = xc_get(d, 'metric', 't_exit');
  else
    t_stop = xc_get(d, 'input', 't_end');
  end

  lib = sc_components(opts);
  blk_R = sc_find('resistor', {'Resistor'}, opts);
  blk_L = sc_find('inductor', {'Inductor'}, opts);
  blk_C = sc_find('capacitor', {'Capacitor'}, opts);
  blk_ref = sc_find('electrical_reference', {'Electrical Reference'}, opts);
  blk_cfg = sc_find('solver_configuration', {'Solver Configuration'}, opts);
  res.blocks = struct('resistor', blk_R, 'inductor', blk_L, 'capacitor', blk_C, ...
                      'electrical_reference', blk_ref, 'solver_configuration', blk_cfg, 'components', lib);

  sc_new(mdl);
  vs = sc_place(mdl, sc_comp(lib, 'vdc'), 'VS', 0, 1);
  r1 = sc_place(mdl, blk_R, 'R1', 1, 0);
  a1 = sc_place(mdl, sc_comp(lib, 'ammeter'), 'A1', 2, 0);
  l1 = sc_place(mdl, blk_L, 'L1', 3, 0);
  c1 = sc_place(mdl, blk_C, 'C1', 4, 1);
  ld = sc_place(mdl, sc_comp(lib, 'cpl'), 'LOAD', 5, 1);
  gnd = sc_place(mdl, blk_ref, 'GND', 3, 3);
  cfg = sc_place(mdl, blk_cfg, 'CFG', 0, 3);
  sc_set(vs, 'V', Vs, 'V');
  sc_set(r1, '^Resistance', R, 'Ohm');
  sc_set(l1, '^Inductance', L, 'H');
  sc_set(c1, '^Capacitance', C, 'F');
  sc_set(ld, 'P', P, 'W');
  sc_set(ld, 'v_min', vmin, 'V');
  res.blocks.inductor_initial = sc_init(l1, 'current', Ie, 'A');
  res.blocks.capacitor_initial = sc_init(c1, 'voltage', Ve + dv0, 'V');
  sc_wire(mdl, sc_port(vs, 'p'), sc_port(r1, 'p'));
  sc_wire(mdl, sc_port(r1, 'n'), sc_port(a1, 'p'));
  sc_wire(mdl, sc_port(a1, 'n'), sc_port(l1, 'p'));
  sc_wire(mdl, sc_port(l1, 'n'), sc_port(c1, 'p'), sc_port(ld, 'p'));   % node x
  sc_wire(mdl, sc_port(vs, 'n'), sc_port(c1, 'n'), sc_port(ld, 'n'), sc_port(gnd, 'one'), sc_port(cfg, 'one'));
  res.solver = sc_solver(mdl, t_stop, 2 * pi / wd / 400);
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
  [tv, v] = sc_series(simlog, 'LOAD', 'v', 'V');
  [ti, iL] = sc_series(simlog, 'A1', 'i', 'A');
  [te, ye] = sc_extrema(tv, v - Ve);
  small = abs(ye) < 0.05 * Ve;
  te = te(small);
  ye = ye(small);
  if numel(te) < 4
    error('fewer than 4 small-amplitude extrema in the log');
  end
  cf = polyfit(te, log(abs(ye)), 1);
  m = sprintf('Simscape (%s), %d extrema of v - Ve with |v - Ve| < 5%% Ve', res.solver, numel(te));
  rows = {};
  rows{end + 1} = sc_row(['SC.ex07_' preset '.v0'], d, 'v(0) = Ve + dv0 (initial target)', 'V', Ve + dv0, v(1), 1e-6, 'rel', m, 'export Ve + dv0');
  rows{end + 1} = sc_row(['SC.ex07_' preset '.i0'], d, 'i_L(0) = P/Ve (initial target and sign)', 'A', Ie, iL(1), 1e-6, 'rel', m, 'export P/Ve');
  rows{end + 1} = sc_row(['SC.ex07_' preset '.sigma'], d, 'growth rate vs linear pole', '1/s', xc_get(d, 'metric', 'pole_re'), cf(1), 1e-2, 'rel', m);
  rows{end + 1} = sc_row(['SC.ex07_' preset '.wd'], d, 'damped frequency vs linear pole', 'rad/s', wd, pi / mean(diff(te)), 1e-2, 'rel', m);
  rows{end + 1} = sc_row(['SC.ex07_' preset '.unstable'], d, 'grows (1) / decays (0); expected from app status', '', ...
                         double(strcmp(d.status, 'UNSTABLE')), double(cf(1) > 0), 0, 'bool', m);
  res.rows = rows;
  res.status = 'RAN';
catch err
  res.status = 'RUN_ERROR';
  res.reason = err.message;
end
res = sc_finish(res, opts, t_start);
end
