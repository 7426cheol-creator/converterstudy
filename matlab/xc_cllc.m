function rows = xc_cllc(export_dir)
%XC_CLLC FL10 CLLC first-harmonic approximation (textbook ch.13).
%   Zp = Zm || (Zr2' + Rac'),  H = Zp/(Zr1 + Zp) * Rac'/(Zr2' + Rac')
%   required gain n*Vbat/Vlink (full-bridge forward FHA), Lr2' = n^2 Lr2,
%   Cr2' = Cr2/n^2, Rac' = (8/pi^2) n^2 Vbat^2/P.
%   Seed: n = 1, Lr1 = Lr2 = 40 uH, Cr1 = Cr2 = 28.1448 nF, Lm = 200 uH,
%   fs = 120 ... 210 kHz, 11 kW. Modification A: n = 0.93, referred symmetry.
%
%   FL10 is not merged yet, so the expected values are the textbook's
%   printed numbers, with half a unit of the last printed digit as tolerance.
%   The time-domain item (FL10 / EX05) is a TODO hook at the end.

rows = {};
src = struct('lab', 'FL10', 'experiment', 'fha', 'preset', 'textbook', 'file', 'textbook ch.13');
tb = 'textbook ch.13 printed value (FL10 export not merged yet)';

Lr1 = 40e-6;
Cr1 = 28.1448e-9;
Lm = 200e-6;
P = 11e3;
f_lo = 120e3;
f_hi = 210e3;
fgrid = linspace(f_lo, f_hi, 9001);

% ---------------------------------------------------------------- seed, n = 1, 920 V battery / 850 V link
t0 = tic;
n = 1;
Vbat = 920;
Vlink = 850;
Greq = n * Vbat / Vlink;
tank = local_tank(Lr1, Cr1, Lm, n, Lr1 / n ^ 2, Cr1 * n ^ 2, Vbat, P);
Gmax = local_max_inductive_gain(tank, fgrid);
t_s = toc(t0);
m = 'grid scan of |H| and Im Zin over 120-210 kHz, fminbnd on interior peaks, fzero on the inductive edges';
rows{end + 1} = xc_row('FL10.seed.Greq', src, 'required gain n Vbat/Vlink at 920/850 V', '', 1.082353, Greq, 5e-7, 'abs', 'n Vo/Vi', t_s, tb);
rows{end + 1} = xc_row('FL10.seed.max_inductive_gain', src, 'max |H| in the inductive region, 120-210 kHz', '', 1.016401, Gmax, 5e-7, 'abs', m, t_s, tb);
rows{end + 1} = xc_row('FL10.seed.no_solution', src, 'no inductive solution (max gain < required)', '', 1, double(Gmax < Greq), 0, 'bool', m, t_s, tb);

% ---------------------------------------------------------------- modification A, n = 0.93
n = 0.93;
Lr2 = Lr1 / n ^ 2;                           % physical secondary values that keep
Cr2 = Cr1 * n ^ 2;                           % the referred tank symmetric
rows{end + 1} = xc_row('FL10.n093.Lr2', src, 'physical Lr2 = Lr1/n^2', 'H', 46.2481e-6, Lr2, 0.5e-10, 'abs', 'referred symmetry: n^2 Lr2 = Lr1', 0, tb);
rows{end + 1} = xc_row('FL10.n093.Cr2', src, 'physical Cr2 = Cr1 n^2', 'F', 24.3424e-9, Cr2, 0.5e-13, 'abs', 'referred symmetry: Cr2/n^2 = Cr1', 0, tb);

corners = [650, 700; 800, 800; 920, 850];    % [battery, link] in V
g_tb = [0.863571, 0.930000, 1.006588];
roots_tb = {164.390e3, 162.811e3, [136.099e3, 147.061e3]};
for c = 1:size(corners, 1)
  t0 = tic;
  Vbat = corners(c, 1);
  Vlink = corners(c, 2);
  tank = local_tank(Lr1, Cr1, Lm, n, Lr2, Cr2, Vbat, P);
  Greq = n * Vbat / Vlink;
  fr = local_roots(tank, fgrid, Greq);
  t_c = toc(t0);
  tag = sprintf('FL10.n093.%d_%d', Vbat, Vlink);
  m = 'sign changes of |H| - Greq on a 9001-point grid, fzero, inductive (Im Zin > 0) roots kept';
  rows{end + 1} = xc_row([tag '.Greq'], src, sprintf('required gain at %d/%d V', Vbat, Vlink), '', g_tb(c), Greq, 5e-7, 'abs', 'n Vo/Vi', 0, tb); %#ok<AGROW>
  rows{end + 1} = xc_row([tag '.n_roots'], src, 'number of inductive FHA solutions in 120-210 kHz', '', numel(roots_tb{c}), numel(fr), 0, 'abs', m, t_c, tb); %#ok<AGROW>
  for j = 1:min(numel(fr), numel(roots_tb{c}))
    rows{end + 1} = xc_row(sprintf('%s.root%d', tag, j), src, sprintf('inductive FHA solution %d', j), 'Hz', roots_tb{c}(j), fr(j), 0.5, 'abs', m, t_c, tb); %#ok<AGROW>
  end
  if c == 3 && numel(fr) == 2
    % local slope by +-10 Hz central difference, and the FHA primary series rms
    slope_tb = [0.001871, -0.001803];        % 1/kHz
    irms_tb = [14.390, 14.765];              % A
    for j = 1:2
      t0 = tic;
      s = (abs(local_H(tank, fr(j) + 10)) - abs(local_H(tank, fr(j) - 10))) / 20 * 1e3;
      V1f = 2 * sqrt(2) / pi * Vlink;        % rms of the full-bridge fundamental
      I1 = V1f / abs(local_Zin(tank, fr(j)));
      t_j = toc(t0);
      rows{end + 1} = xc_row(sprintf('%s.slope%d', tag, j), src, sprintf('d|H|/df at solution %d (+-10 Hz)', j), '1/kHz', slope_tb(j), s, 5e-7, 'abs', 'central difference of |H|', t_j, tb); %#ok<AGROW>
      rows{end + 1} = xc_row(sprintf('%s.I1rms%d', tag, j), src, sprintf('FHA primary series rms at solution %d', j), 'A', irms_tb(j), I1, 5e-4, 'abs', 'I1 = (2 sqrt(2)/pi) Vlink / |Zin|', t_j, tb); %#ok<AGROW>
    end
  end
end

% ---------------------------------------------------------------- 920/850 V roots to 0.01 Hz
% The task lists the FL10 draft roots 136099.47 Hz and 147060.86 Hz, computed
% with Cr1 set for fr = 150 kHz exactly (28.144773 nF; the textbook prints
% 28.1448 nF, which moves the roots by less than 0.1 Hz).
t0 = tic;
Cr1x = 1 / ((2 * pi * 150e3) ^ 2 * Lr1);
tank = local_tank(Lr1, Cr1x, Lm, n, Lr1 / n ^ 2, Cr1x * n ^ 2, 920, P);
fr = local_roots(tank, fgrid, n * 920 / 850);
t_x = toc(t0);
m = 'as above with Cr1 = 1/((2 pi 150 kHz)^2 Lr1)';
draft = [136099.47, 147060.86];
ds = 'FL10 draft value listed in the task (FL10 export pending)';
for j = 1:min(numel(fr), 2)
  rows{end + 1} = xc_row(sprintf('FL10.n093.920_850.root%d.fr150k', j), src, sprintf('inductive FHA solution %d, Cr1 for fr = 150 kHz', j), 'Hz', draft(j), fr(j), 0.005 + 1e-6, 'abs', m, t_x, ds); %#ok<AGROW>
end
if numel(fr) ~= 2
  rows{end + 1} = xc_row('FL10.n093.920_850.n_roots.fr150k', src, 'number of inductive FHA solutions', '', 2, numel(fr), 0, 'abs', m, t_x, ds);
end

% ---------------------------------------------------------------- TODO(FL10/EX05 time domain)
% Hook for the switching / time-domain comparison. When FL10 (CLLC) and EX05
% are merged:
%   1. add their presets to the export list in verification/run_octave_crosscheck.sh;
%   2. integrate the textbook E05 state model x = [i1, i2, vC1, vC2] with ode45
%      (v_m from the node equation, v1 = +-Vlink, v2 from the rectifier mode)
%      at both n = 0.93 branches of the 920/850 V corner, shooting for the
%      periodic state (Phi_T(x0) = x0), and compare the exported switching
%      results (output power / current, rms, capacitor peak voltages);
%   3. replace the TODO row below by those comparison rows.
rows{end + 1} = local_todo(src, 'FL10.time_domain.TODO', ...
  'CLLC switching / time-domain comparison (FL10, EX05)', ...
  'not written yet: FL10/EX05 not merged; FHA numbers above only');
end

% ======================================================================
function tank = local_tank(Lr1, Cr1, Lm, n, Lr2, Cr2, Vbat, P)
tank.Lr1 = Lr1;
tank.Cr1 = Cr1;
tank.Lm = Lm;
tank.Lr2p = n ^ 2 * Lr2;                     % primary-referred secondary branch
tank.Cr2p = Cr2 / n ^ 2;
tank.Racp = 8 / pi ^ 2 * n ^ 2 * Vbat ^ 2 / P;
end

function [H, Zin] = local_H(tank, f)
w = 2 * pi * f;
Zr1 = 1i * w * tank.Lr1 + 1 ./ (1i * w * tank.Cr1);
Zr2 = 1i * w * tank.Lr2p + 1 ./ (1i * w * tank.Cr2p);
Zm = 1i * w * tank.Lm;
Zb = Zr2 + tank.Racp;
Zp = Zm .* Zb ./ (Zm + Zb);
Zin = Zr1 + Zp;
H = Zp ./ Zin .* tank.Racp ./ Zb;
end

function Zin = local_Zin(tank, f)
[~, Zin] = local_H(tank, f);
end

function G = local_max_inductive_gain(tank, fgrid)
[H, Zin] = local_H(tank, fgrid);
g = abs(H);
ind = imag(Zin) > 0;
cand = [];
% interior local maxima of |H| inside the inductive region
for k = 2:numel(fgrid) - 1
  if ind(k) && g(k) >= g(k - 1) && g(k) >= g(k + 1)
    fb = fminbnd(@(f) -abs(local_H(tank, f)), fgrid(k - 1), fgrid(k + 1), optimset('TolX', 1e-6));
    if imag(local_Zin(tank, fb)) > 0
      cand(end + 1) = abs(local_H(tank, fb)); %#ok<AGROW>
    end
  end
end
% edges of the inductive region (Im Zin = 0) and the ends of the range
for k = 1:numel(fgrid) - 1
  if ind(k) ~= ind(k + 1)
    fe = fzero(@(f) imag(local_Zin(tank, f)), [fgrid(k), fgrid(k + 1)], optimset('TolX', 1e-9));
    cand(end + 1) = abs(local_H(tank, fe)); %#ok<AGROW>
  end
end
if ind(1)
  cand(end + 1) = g(1);
end
if ind(end)
  cand(end + 1) = g(end);
end
G = max([cand, -Inf]);
end

function fr = local_roots(tank, fgrid, Greq)
e = abs(local_H(tank, fgrid)) - Greq;
fr = [];
for k = 1:numel(fgrid) - 1
  if e(k) == 0 || e(k) * e(k + 1) < 0
    f0 = fzero(@(f) abs(local_H(tank, f)) - Greq, [fgrid(k), fgrid(k + 1)], optimset('TolX', 1e-9));
    if imag(local_Zin(tank, f0)) > 0
      fr(end + 1) = f0; %#ok<AGROW>
    end
  end
end
fr = sort(fr);
end

function r = local_todo(src, item, quantity, note)
r = struct();
r.item = item;
r.lab = src.lab;
r.experiment = 'time_domain';
r.preset = 'n093_920_850';
r.quantity = quantity;
r.unit = '';
r.expected = NaN;
r.expected_source = 'FL10 / EX05 export (pending)';
r.octave_value = NaN;
r.abs_error = NaN;
r.rel_error = NaN;
r.tol = NaN;
r.tol_kind = 'todo';
r.status = 'TODO';
r.runtime_s = 0;
r.method = note;
end
