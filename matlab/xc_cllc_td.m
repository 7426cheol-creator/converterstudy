function rows = xc_cllc_td(export_dir)
%XC_CLLC_TD CLLC switching model (FL10 time_domain, EX05 operating_points)
%   checked with an independent hybrid ode45 model (xc_cllc_period: textbook
%   E05 state equations, ideal full bridge, three-state ideal diode bridge with
%   the conduction logic written here) and Newton shooting (xc_cllc_orbit).
%
%   FL10 time_domain textbook / lossy (stiff 920 V battery, 850 V link):
%     FHA roots with R (own nodal FHA), switching power, I1 rms, rectifier off
%     share and the largest Floquet multiplier at both roots, the same at the
%     two other corners, then continuation along the upper branch to the
%     exported 11 kW frequency (C1 / C2' peaks, I1 rms, power), the 11 kW root
%     of this model and the +-2 Hz slope dP/df.
%   EX05 operating_points textbook (R-C output, Co = 100 uF, R_L = V^2/P):
%     output voltage at both FHA roots, the +-50 Hz slopes and the frequency
%     where the cycle-average output is 920 V.
%
%   Tolerances, fixed before comparing (see matlab/README.md):
%     switching quantities 1e-5 relative: ode45 RelTol 1e-10 agrees with
%       RelTol 1e-11 to < 1e-9 at these points, and the app accepts a shooting
%       residual of 1e-8, which the slowest Floquet multiplier (0.989) can
%       amplify about 100x: 1e-5 is 10x the larger of the two;
%     printed table values: half a unit of the last printed digit (+1e-5 rel);
%     Floquet |lambda|max: 5e-5 (4 printed decimals) + 1e-5 (finite-difference
%       monodromy);
%     the app's 11 kW frequency: 0.5 Hz (it bisects to a 1 Hz interval and
%       reports the midpoint), so the power there may differ from 11 kW by up
%       to 0.5 Hz x |dP/df|;
%     EX05 operating frequency: 1e-3 Hz (brentq xtol of the app) + 1e-4 Hz;
%     slopes: two values each within 1e-5 relative, divided by the difference
%       step.

rows = {};
rows = [rows, local_fl10(export_dir, 'textbook')];
rows = [rows, local_fl10(export_dir, 'lossy')];
rows = [rows, local_ex05(export_dir)];
end

% ======================================================================
function rows = local_fl10(export_dir, preset)
rows = {};
d = xc_load(export_dir, 'FL10', 'time_domain', preset);
p = local_params(d);
p.out = 'stiff';
P = xc_get(d, 'input', 'P');
p.P_guess = P;
fmin = xc_get(d, 'input', 'f_min');
fmax = xc_get(d, 'input', 'f_max');
tag = ['FL10.td_' preset '.'];
rel = 1e-5;
m0 = sprintf('ode45 hybrid model (RelTol 1e-10, diode bridge P/N/off), Newton shooting; R1 = R2'' = %g Ohm', p.R1);

% ---------------------------------------------------------------- high corner, both FHA roots
Vbat = xc_get(d, 'input', 'Vbat_hi');
Vlink = xc_get(d, 'input', 'Vlink_hi');
t0 = tic;
fr = local_fha_roots(p, Vbat, Vlink, P, fmin, fmax);
t_f = toc(t0);
if numel(fr) ~= 2
  error('xc:cllc_td', 'FL10 %s: expected two inductive FHA roots at %g/%g V, found %d', preset, Vbat, Vlink, numel(fr));
end
td = xc_table(d, 't_td');
op = xc_table(d, 't_op');
keys = {'P_td_lo', 'P_td_hi'};
orbs = cell(1, 2);
for j = 1:2
  lab = local_label(d, keys{j});
  f_lab = local_num(lab, '([0-9]+\.[0-9]+) kHz') * 1e3;
  rows{end + 1} = xc_row(sprintf('%sfha_root%d', tag, j), d, sprintf('FHA root %d with R (metric label %s)', j, keys{j}), 'Hz', ...
    f_lab, fr(j), 0.5 + 1e-6, 'abs', 'own nodal FHA with R1, R2'' in series, fzero on |H| = n Vbat/Vlink; label prints kHz with 3 decimals', t_f); %#ok<AGROW>
  q = local_point(p, Vlink, Vbat, fr(j), [], true);
  orbs{j} = q;
  row = local_row(td, '920/850', sprintf('%.3f kHz', fr(j) / 1e3));
  rows{end + 1} = xc_row(sprintf('%sP_root%d', tag, j), d, sprintf('battery power at FHA root %d (%.3f kHz)', j, fr(j) / 1e3), 'W', ...
    xc_get(d, 'metric', keys{j}), q.sum.P_rect, rel, 'rel', m0, q.t); %#ok<AGROW>
  rows = [rows, local_table_rows(d, tag, sprintf('root%d', j), row, q, m0)]; %#ok<AGROW>
end

% ---------------------------------------------------------------- the two other corners (t_td rows)
cn = {'lo', 'mid'};
for c = 1:2
  Vb = xc_get(d, 'input', ['Vbat_' cn{c}]);
  Vl = xc_get(d, 'input', ['Vlink_' cn{c}]);
  fc = local_fha_roots(p, Vb, Vl, P, fmin, fmax);
  for j = 1:numel(fc)
    q = local_point(p, Vl, Vb, fc(j), [], true);
    row = local_row(td, sprintf('%g/%g', Vb, Vl), sprintf('%.3f kHz', fc(j) / 1e3));
    kw = local_num(row{4}, '([0-9]+\.[0-9]+) kW');
    rows{end + 1} = xc_row(sprintf('%sP_%g_%g', tag, Vb, Vl), d, sprintf('battery power at the %g/%g V FHA root (%.3f kHz), table t_td', Vb, Vl, fc(j) / 1e3), 'W', ...
      kw * 1e3, q.sum.P_rect, 5 + rel * kw * 1e3, 'abs', [m0 '; table prints kW with 2 decimals'], q.t); %#ok<AGROW>
    rows = [rows, local_table_rows(d, tag, sprintf('%g_%g', Vb, Vl), row, q, m0)]; %#ok<AGROW>
  end
end

% ---------------------------------------------------------------- 11 kW point on the upper branch
fsw = xc_get(d, 'metric', 'f_sw_hi');
t0 = tic;
qs = local_walk(p, Vlink, Vbat, orbs{2}, fsw);
t_w = toc(t0);
m1 = [m0 sprintf('; continuation from the %.3f kHz root to the exported f_sw_hi', fr(2) / 1e3)];
rows{end + 1} = xc_row([tag 'vC1_pk_fsw'], d, 'C1 voltage peak at the exported 11 kW frequency', 'V', ...
  xc_get(d, 'metric', 'vC1_pk'), qs.sum.vC1_pk, rel, 'rel', m1, t_w); %#ok<AGROW>
rows{end + 1} = xc_row([tag 'vC2_pk_fsw'], d, 'C2'' voltage peak (primary-referred) at the exported 11 kW frequency', 'V', ...
  xc_get(d, 'metric', 'vC2_pk'), qs.sum.vC2_pk, rel, 'rel', m1, t_w); %#ok<AGROW>
orow = local_row(op, '920/850', sprintf('%.3f kHz', fr(2) / 1e3));   % row of the upper-branch root
i1 = local_num(orow{5}, '([0-9]+\.[0-9]+) A');
rows{end + 1} = xc_row([tag 'I1rms_fsw'], d, 'I1 rms at the exported 11 kW frequency, table t_op', 'A', ...
  i1, qs.sum.I1_rms, 0.005 + rel * i1, 'abs', [m1 '; table prints 2 decimals'], t_w); %#ok<AGROW>

% slope by the app's own definition: (P(f + 2 Hz) - P(f - 2 Hz)) / 4 Hz, warm starts
t0 = tic;
qp = local_point(p, Vlink, Vbat, fsw + 2, qs, false);
qm = local_point(p, Vlink, Vbat, fsw - 2, qs, false);
sens = (qp.sum.P_rect - qm.sum.P_rect) / 4;                % W/Hz
t_s = toc(t0);
tol_s = rel * (abs(qp.sum.P_rect) + abs(qm.sum.P_rect)) / 4 * 1e3;
rows{end + 1} = xc_row([tag 'sens_fsw'], d, 'dP/df at the 11 kW point, +-2 Hz central difference', 'W/kHz', ...
  xc_get(d, 'metric', 'sens_hi'), sens * 1e3, tol_s, 'abs', [m1 '; tolerance from the two powers (1e-5 rel each)'], t_s); %#ok<AGROW>
% power at the exported frequency against the target, within the app's bisection half-width
rows{end + 1} = xc_row([tag 'P_at_fsw'], d, sprintf('battery power at the exported f_sw_hi (target %g W)', P), 'W', ...
  P, qs.sum.P_rect, 0.5 * abs(sens), 'abs', [m1 '; tolerance = 0.5 Hz (app bisection half-width) x |dP/df|'], t_w, 'app target power (FL10 input P)'); %#ok<AGROW>

% this model's own 11 kW frequency on the same branch
t0 = tic;
[f_star, q_star] = local_root(p, Vlink, Vbat, qs, fsw, @(q) q.sum.P_rect - P, 0.25);
t_r = toc(t0);
rows{end + 1} = xc_row([tag 'f_11kW'], d, sprintf('frequency of %g W on the upper branch (this model: P = %.3f W there)', P, q_star.sum.P_rect), 'Hz', ...
  fsw, f_star, 0.5 + 1e-4, 'abs', [m0 '; bracketing root search to 1e-4 Hz; app bisects to 1 Hz'], t_r); %#ok<AGROW>
end

% ======================================================================
function rows = local_ex05(export_dir)
rows = {};
d = xc_load(export_dir, 'EX05', 'operating_points', 'textbook');
p = local_params(d);
p.out = 'rc';
P = xc_get(d, 'input', 'P');
Vref = xc_get(d, 'input', 'Vref');
p.Co = xc_get(d, 'input', 'Co');
p.Rout = Vref ^ 2 / P;                     % R_L = V^2 / P (actual)
p.vo_guess = Vref;
Vlink = xc_get(d, 'input', 'Vlink');
rel = 1e-5;
m0 = sprintf('ode45 hybrid model with Co = %g uF and R_L = %.4g Ohm (actual), Newton shooting; cycle-average vo', p.Co * 1e6, p.Rout);
t0 = tic;
fr = local_fha_roots(p, Vref, Vlink, P, xc_get(d, 'input', 'f_min'), xc_get(d, 'input', 'f_max'));
t_f = toc(t0);
if numel(fr) ~= 2
  error('xc:cllc_td', 'EX05: expected two inductive FHA roots, found %d', numel(fr));
end
tags = {'lo', 'hi'};
orbs = cell(1, 2);
for j = 1:2
  q = local_point_rc(p, Vlink, fr(j), []);
  orbs{j} = q;
  rows{end + 1} = xc_row(sprintf('EX05.op.vo_root%d', j), d, sprintf('cycle-average V_o at FHA root %d (%.3f kHz)', j, fr(j) / 1e3), 'V', ...
    xc_get(d, 'metric', ['vo_at_' tags{j}]), q.sum.vo_avg, rel, 'rel', m0, q.t + t_f); %#ok<AGROW>
  t0 = tic;
  qp = local_point_rc(p, Vlink, fr(j) + 50, q);
  qm = local_point_rc(p, Vlink, fr(j) - 50, q);
  t_s = toc(t0);
  slope = (qp.sum.vo_avg - qm.sum.vo_avg) / 100 * 1e3;      % V/kHz
  tol_s = rel * (qp.sum.vo_avg + qm.sum.vo_avg) / 100 * 1e3;
  rows{end + 1} = xc_row(sprintf('EX05.op.slope_root%d', j), d, sprintf('dV_o/df at FHA root %d, +-50 Hz', j), 'V/kHz', ...
    xc_get(d, 'metric', ['slope_td_' tags{j}]), slope, tol_s, 'abs', [m0 '; tolerance from the two V_o values (1e-5 rel each)'], t_s); %#ok<AGROW>
end
f0 = xc_get(d, 'metric', 'f0');
t0 = tic;
q0 = local_walk_rc(p, Vlink, orbs{2}, f0);
[f_star, q_star] = local_root_rc(p, Vlink, q0, f0, @(q) q.sum.vo_avg - Vref, 0.25);
t_r = toc(t0);
rows{end + 1} = xc_row('EX05.op.f_920V', d, sprintf('frequency where the cycle-average V_o = %g V (this model: %.6f V there)', Vref, q_star.sum.vo_avg), 'Hz', ...
  f0, f_star, 1e-3 + 1e-4, 'abs', [m0 '; bracketing root search to 1e-5 Hz; app brentq xtol 1e-3 Hz'], t_r); %#ok<AGROW>
end

% ======================================================================
function p = local_params(d)
% primary-referred symmetric tank of the FL10 modification (L2' = L1, C2' = C1)
p.L1 = xc_get(d, 'input', 'L1');
p.C1 = xc_get(d, 'input', 'C1');
p.Lm = xc_get(d, 'input', 'Lm');
p.L2 = p.L1;
p.C2 = p.C1;
p.R1 = xc_get(d, 'input', 'R');
p.R2 = p.R1;
p.n = xc_get(d, 'input', 'n');
end

function fr = local_fha_roots(p, Vbat, Vlink, P, fmin, fmax)
% inductive roots of |H(f)| = n Vbat / Vlink, H of the T network with Rac'
Rac = 8 / pi ^ 2 * p.n ^ 2 * Vbat ^ 2 / P;
G = p.n * Vbat / Vlink;
fg = linspace(fmin, fmax, 9001);
g = abs(local_H(p, fg, Rac)) - G;
fr = [];
for k = 1:numel(fg) - 1
  if g(k) * g(k + 1) < 0
    f0 = fzero(@(f) abs(local_H(p, f, Rac)) - G, [fg(k), fg(k + 1)], optimset('TolX', 1e-9));
    [~, Zin] = local_H(p, f0, Rac);
    if imag(Zin) > 0
      fr(end + 1) = f0; %#ok<AGROW>
    end
  end
end
end

function [H, Zin] = local_H(p, f, Rac)
w = 2 * pi * f;
Z1 = p.R1 + 1i * w * p.L1 + 1 ./ (1i * w * p.C1);
Zb = p.R2 + 1i * w * p.L2 + 1 ./ (1i * w * p.C2) + Rac;
Zm = 1i * w * p.Lm;
Zp = Zm .* Zb ./ (Zm + Zb);
Zin = Z1 + Zp;
H = Zp ./ Zin .* Rac ./ Zb;
end

function q = local_point(p, Vlink, Vbat, f, start, floquet)
p.Vin = Vlink;
p.Vo = Vbat;
p.f = f;
t0 = tic;
opt = struct('floquet', floquet);
if isempty(start)
  o = xc_cllc_orbit(p, [], opt);
else
  opt.J = start.J;
  o = xc_cllc_orbit(p, start.x0, opt);
end
if ~o.converged
  error('xc:cllc_td', 'no periodic orbit at %.6f Hz (residual %.2e)', f, o.residual);
end
q = o;
q.f = f;
q.t = toc(t0);
end

function q = local_point_rc(p, Vlink, f, start)
p.Vin = Vlink;
q = local_point(p, Vlink, NaN, f, start, false);
end

function q = local_walk(p, Vlink, Vbat, q, f_to)
% continuation in frequency from orbit q to f_to (warm starts, adaptive step)
h = 300 * sign(f_to - q.f);
while q.f ~= f_to
  f_next = q.f + h;
  if (f_next - f_to) * sign(h) >= 0
    f_next = f_to;
  end
  try
    q = local_point(p, Vlink, Vbat, f_next, q, false);
  catch
    h = h / 4;
    if abs(h) < 0.01
      error('xc:cllc_td', 'continuation stalled near %.4f Hz', q.f);
    end
  end
end
end

function q = local_walk_rc(p, Vlink, q, f_to)
q = local_walk(p, Vlink, NaN, q, f_to);
end

function [f, q] = local_root(p, Vlink, Vbat, q0, f0, g, step)
% bracket g(f) = 0 next to f0 by steps of +-step, then Illinois regula falsi
% with warm starts from the nearest bracket end, to 1e-4 Hz (1e-5 Hz for 'rc')
a = q0;
ga = g(a);
b = [];
for k = 1:200
  for s = [1, -1]
    try
      c = local_point(p, Vlink, Vbat, f0 + s * k * step, a, false);
    catch
      continue;                    % no orbit from this warm start: try the other side
    end
    if g(c) * ga <= 0
      b = c;
      break;
    end
  end
  if ~isempty(b)
    break;
  end
end
if isempty(b)
  error('xc:cllc_td', 'no sign change of the target function within %g Hz of %.4f Hz', 200 * step, f0);
end
if strcmp(p.out, 'rc')
  tolf = 1e-5;
else
  tolf = 1e-4;
end
gb = g(b);
side = 0;
q = b;
for it = 1:100
  fc = b.f - gb * (b.f - a.f) / (gb - ga);
  if abs(fc - a.f) < abs(fc - b.f)
    near = a;
  else
    near = b;
  end
  q = local_point(p, Vlink, Vbat, fc, near, false);
  gc = g(q);
  if gc * gb < 0
    a = b;
    ga = gb;
    side = 0;
  else
    if side == 1
      ga = ga / 2;                 % Illinois step
    end
    side = 1;
  end
  b = q;
  gb = gc;
  if abs(b.f - a.f) < tolf || gc == 0
    break;
  end
end
f = q.f;
end

function [f, q] = local_root_rc(p, Vlink, q0, f0, g, step)
[f, q] = local_root(p, Vlink, NaN, q0, f0, g, step);
end

function rows = local_table_rows(d, tag, name, row, q, m0)
% I1 rms, rectifier off share and Floquet |lambda|max printed in table t_td
rows = {};
i1 = local_num(row{5}, '^([0-9]+\.[0-9]+) A');
off = local_num(row{6}, '([0-9]+\.[0-9]+) %') / 100;
rho = str2double(row{7});
rows{end + 1} = xc_row(sprintf('%sI1rms_%s', tag, name), d, sprintf('I1 rms (table t_td, %s)', row{2}), 'A', ...
  i1, q.sum.I1_rms, 0.005 + 1e-5 * i1, 'abs', [m0 '; table prints 2 decimals'], q.t);
rows{end + 1} = xc_row(sprintf('%soff_%s', tag, name), d, sprintf('rectifier off share (table t_td, %s)', row{2}), '', ...
  off, q.sum.off_frac, 5e-4, 'abs', [m0 '; table prints % with 1 decimal'], q.t);
rows{end + 1} = xc_row(sprintf('%srho_%s', tag, name), d, sprintf('largest Floquet |lambda| (table t_td, %s)', row{2}), '', ...
  rho, q.rho, 6e-5, 'abs', [m0 '; monodromy by central differences, step 1e-4 of the state scale; table prints 4 decimals'], q.t);
end

function row = local_row(tab, key1, key2)
for k = 1:numel(tab)
  r = tab{k};
  if ~isempty(strfind(r{1}, key1)) && (~isempty(strfind(r{2}, key2)) || ~isempty(strfind(r{3}, key2)))
    row = r;
    return;
  end
end
error('xc:cllc_td', 'no table row with "%s" and "%s"', key1, key2);
end

function lab = local_label(d, key)
m = xc_metrics(d);
for k = 1:numel(m)
  if strcmp(m{k}.key, key)
    lab = m{k}.label;
    return;
  end
end
error('xc:cllc_td', 'metric %s not found in %s', key, d.file);
end

function v = local_num(txt, pat)
tok = regexp(txt, pat, 'tokens', 'once');
if isempty(tok)
  error('xc:cllc_td', 'no number matching %s in "%s"', pat, txt);
end
v = str2double(tok{1});
end
