function res = build_ex02_commutation(opts)
%BUILD_EX02_COMMUTATION Simscape model of the EX02 half-bridge commutation
%   with nonlinear Coss (textbook E02), built at run time.
%
%   res = build_ex02_commutation()       textbook preset, exported values
%   res = build_ex02_commutation(opts)   options: see sc_opts
%
%   Circuit: fixed rails (xcsc.vdc Vb), high device CH = xcsc.coss from the
%   rail to the switch node x, low device CL = xcsc.coss_start0 from x to
%   ground (starts at 0 V), constant current I into x (xcsc.idc),
%   C(v) = C0/sqrt(1 + v/V0) in both devices; optional linear Cpar at x.
%   The node rises with C_node(v) dv/dt = I, C_node(v) = C(v) + C(Vb - v).
%   Checked against the export: time to reach the rail, node voltage at the
%   end of the dead time td, and the incoming device's Vds at turn-on.
%   Only the charge-current principle is modelled (E02: the first stage);
%   no resonant current, no channel current, no gate.

if nargin < 1
  opts = struct();
end
opts = sc_opts(opts);
t_start = tic;
res = sc_result('ex02_commutation', 'build_ex02_commutation()', ...
                'python -m convlab run EX02 hb_constant_current --preset textbook');
res.env = sc_env();
if ~res.env.ok
  res.reason = res.env.reason;
  res = sc_finish(res, opts, t_start);
  return;
end

mdl = 'xc_ex02_commutation';
try
  d = xc_load(opts.export_dir, 'EX02', 'hb_constant_current', 'textbook');
  C0 = xc_get(d, 'input', 'C0');
  V0 = xc_get(d, 'input', 'V0');
  Vb = xc_get(d, 'input', 'Vb');
  I = xc_get(d, 'input', 'I');
  td = xc_get(d, 'input', 'td');
  Cpar = xc_get(d, 'input', 'Cpar');
  t_bound = 2 * (2 * C0 + Cpar) * Vb / I;    % C_node <= 2 C0 + Cpar

  lib = sc_components(opts);
  blk_ref = sc_find('electrical_reference', {'Electrical Reference'}, opts);
  blk_cfg = sc_find('solver_configuration', {'Solver Configuration'}, opts);
  res.blocks = struct('electrical_reference', blk_ref, 'solver_configuration', blk_cfg, 'components', lib);

  sc_new(mdl);
  rail = sc_place(mdl, sc_comp(lib, 'vdc'), 'RAIL', 0, 1);
  ch = sc_place(mdl, sc_comp(lib, 'coss'), 'CH', 1, 0);
  cl = sc_place(mdl, sc_comp(lib, 'coss_start0'), 'CL', 2, 1);
  is = sc_place(mdl, sc_comp(lib, 'idc'), 'ISRC', 3, 1);
  gnd = sc_place(mdl, blk_ref, 'GND', 1, 3);
  cfg = sc_place(mdl, blk_cfg, 'CFG', 0, 3);
  sc_set(rail, 'V', Vb, 'V');
  for b = {ch, cl}
    sc_set(b{1}, 'C0', C0, 'F');
    sc_set(b{1}, 'V0', V0, 'V');
  end
  sc_set(is, 'I', I, 'A');
  sc_wire(mdl, sc_port(rail, 'p'), sc_port(ch, 'p'));                    % rail
  node = [sc_port(ch, 'n'), sc_port(cl, 'p'), sc_port(is, 'p')];         % switch node x
  ground = [sc_port(rail, 'n'), sc_port(cl, 'n'), sc_port(is, 'n'), sc_port(gnd, 'one'), sc_port(cfg, 'one')];
  if Cpar > 0
    blk_C = sc_find('capacitor', {'Capacitor'}, opts);
    res.blocks.capacitor = blk_C;
    cp = sc_place(mdl, blk_C, 'CPAR', 3, 0);
    sc_set(cp, '^Capacitance', Cpar, 'F');
    node(end + 1) = sc_port(cp, 'p');
    ground(end + 1) = sc_port(cp, 'n');
  end
  sc_wire(mdl, node);
  sc_wire(mdl, ground);
  res.solver = sc_solver(mdl, t_bound, t_bound / 4000);
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
  [t, v] = sc_series(simlog, 'CL', 'v', 'V');          % node voltage = low-device Vds
  [t, k] = unique(t, 'last');
  v = v(k);
  if v(end) < Vb
    error('the node did not reach the rail within %.3g s', t(end));
  end
  j = find(v >= Vb, 1, 'first');
  t_rail = interp1(v(j - 1:j), t(j - 1:j), Vb);
  v_td = interp1(t, v, td);
  m = sprintf('Simscape (%s), node voltage of the low device, linear interpolation of the log', res.solver);
  rows = {};
  rows{end + 1} = sc_row('SC.ex02.t_trans', d, 'time to reach the rail at I', 's', xc_get(d, 'metric', 't_trans'), t_rail, 1e-3, 'rel', m);
  rows{end + 1} = sc_row('SC.ex02.v_td', d, 'node voltage at the end of td', 'V', xc_get(d, 'metric', 'v_td'), v_td, 1e-3, 'rel', m);
  rows{end + 1} = sc_row('SC.ex02.vds_on', d, 'Vds of the incoming device at turn-on', 'V', xc_get(d, 'metric', 'vds_on'), Vb - v_td, 1e-3, 'rel', m);
  rows{end + 1} = sc_row('SC.ex02.v0', d, 'node voltage at t = 0 (initial target)', 'V', 0, v(1), 1e-6 * Vb, 'abs', m, 'model definition: the low device starts at 0 V');
  res.rows = rows;
  res.status = 'RAN';
catch err
  res.status = 'RUN_ERROR';
  res.reason = err.message;
end
res = sc_finish(res, opts, t_start);
end
