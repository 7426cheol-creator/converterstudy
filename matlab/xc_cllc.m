function rows = xc_cllc(export_dir)
%XC_CLLC FL10 CLLC first-harmonic approximation (textbook ch.13).
%   Zp = Zm || (Zr2' + Rac'),  H = Zp/(Zr1 + Zp) * Rac'/(Zr2' + Rac')
%   required gain n*Vbat/Vlink (full-bridge forward FHA), Lr2' = n^2 Lr2,
%   Cr2' = Cr2/n^2, Rac' = (8/pi^2) n^2 Vbat^2/P.
%   Seed: n = 1, Lr1 = Lr2 = 40 uH, Cr1 = Cr2 = 28.1448 nF, Lm = 200 uH,
%   fs = 120 ... 210 kHz, 11 kW. Modification A: n = 0.93, referred symmetry.
%
%   Two sets of expected values:
%   - the textbook's printed numbers (printed Cr1 = 28.1448 nF), tolerance half
%     a unit of the last printed digit;
%   - the FL10 exports (seed_fail seed, fix_n093 textbook; Cr1 = 28.144773 nF,
%     f_r = 150 kHz), recomputed from the exported inputs; the roots and gains
%     are root solves of the same analytic function, so 1e-9 relative covers
%     both solvers.
%   The switching (time-domain) model is checked in xc_cllc_td.

rows = {};
src = struct('lab', 'FL10', 'experiment', 'fha', 'preset', 'textbook', 'file', 'textbook ch.13');
tb = 'textbook ch.13 printed value';

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
rows{end + 1} = xc_row('FL10.tb.seed.Greq', src, 'required gain n Vbat/Vlink at 920/850 V', '', 1.082353, Greq, 5e-7, 'abs', 'n Vo/Vi', t_s, tb);
rows{end + 1} = xc_row('FL10.tb.seed.max_inductive_gain', src, 'max |H| in the inductive region, 120-210 kHz', '', 1.016401, Gmax, 5e-7, 'abs', m, t_s, tb);
rows{end + 1} = xc_row('FL10.tb.seed.no_solution', src, 'no inductive solution (max gain < required)', '', 1, double(Gmax < Greq), 0, 'bool', m, t_s, tb);

% ---------------------------------------------------------------- modification A, n = 0.93
n = 0.93;
Lr2 = Lr1 / n ^ 2;                           % physical secondary values that keep
Cr2 = Cr1 * n ^ 2;                           % the referred tank symmetric
rows{end + 1} = xc_row('FL10.tb.n093.Lr2', src, 'physical Lr2 = Lr1/n^2', 'H', 46.2481e-6, Lr2, 0.5e-10, 'abs', 'referred symmetry: n^2 Lr2 = Lr1', 0, tb);
rows{end + 1} = xc_row('FL10.tb.n093.Cr2', src, 'physical Cr2 = Cr1 n^2', 'F', 24.3424e-9, Cr2, 0.5e-13, 'abs', 'referred symmetry: Cr2/n^2 = Cr1', 0, tb);

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
  tag = sprintf('FL10.tb.n093.%d_%d', Vbat, Vlink);
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

% ---------------------------------------------------------------- FL10 exports
rows = [rows, local_exports(export_dir, fgrid)];
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

function [G, fG] = local_max_inductive_gain(tank, fgrid)
[H, Zin] = local_H(tank, fgrid);
g = abs(H);
ind = imag(Zin) > 0;
cand = [];
fcand = [];
% interior local maxima of |H| inside the inductive region
for k = 2:numel(fgrid) - 1
  if ind(k) && g(k) >= g(k - 1) && g(k) >= g(k + 1)
    fb = fminbnd(@(f) -abs(local_H(tank, f)), fgrid(k - 1), fgrid(k + 1), optimset('TolX', 1e-6));
    if imag(local_Zin(tank, fb)) > 0
      cand(end + 1) = abs(local_H(tank, fb)); %#ok<AGROW>
      fcand(end + 1) = fb; %#ok<AGROW>
    end
  end
end
% edges of the inductive region (Im Zin = 0) and the ends of the range
for k = 1:numel(fgrid) - 1
  if ind(k) ~= ind(k + 1)
    fe = fzero(@(f) imag(local_Zin(tank, f)), [fgrid(k), fgrid(k + 1)], optimset('TolX', 1e-9));
    cand(end + 1) = abs(local_H(tank, fe)); %#ok<AGROW>
    fcand(end + 1) = fe; %#ok<AGROW>
  end
end
if ind(1)
  cand(end + 1) = g(1);
  fcand(end + 1) = fgrid(1);
end
if ind(end)
  cand(end + 1) = g(end);
  fcand(end + 1) = fgrid(end);
end
[G, k] = max([cand, -Inf]);
fG = NaN;
if k <= numel(fcand)
  fG = fcand(k);
end
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

function rows = local_exports(export_dir, fgrid)
% FL10 FHA exports, recomputed from their own inputs
rows = {};
d = xc_load(export_dir, 'FL10', 'seed_fail', 'seed');
L1 = xc_get(d, 'input', 'L1');
C1 = xc_get(d, 'input', 'C1');
Lm = xc_get(d, 'input', 'Lm');
n = xc_get(d, 'input', 'n');
P = xc_get(d, 'input', 'P');
Vb = xc_get(d, 'input', 'Vbat_hi');
Vl = xc_get(d, 'input', 'Vlink_hi');
t0 = tic;
tank = local_tank(L1, C1, Lm, n, L1 / n ^ 2, C1 * n ^ 2, Vb, P);
[Gmax, fG] = local_max_inductive_gain(tank, fgrid);
t_s = toc(t0);
m = 'grid scan of |H| and Im Zin over 120-210 kHz, fminbnd on interior peaks, fzero on the inductive edges';
rows{end + 1} = xc_row('FL10.seed.fr', d, 'f_r = 1/(2 pi sqrt(L1 C1))', 'Hz', xc_get(d, 'metric', 'fr'), 1 / (2 * pi * sqrt(L1 * C1)), 1e-12, 'rel', 'definition', 0);
rows{end + 1} = xc_row('FL10.seed.g_req_hi', d, 'required gain n Vbat/Vlink at 920/850 V', '', xc_get(d, 'metric', 'g_req_hi'), n * Vb / Vl, 1e-12, 'rel', 'n Vo/Vi', 0);
rows{end + 1} = xc_row('FL10.seed.gmax_hi', d, sprintf('max inductive |H| at 920/850 V (at %.2f kHz)', fG / 1e3), '', xc_get(d, 'metric', 'gmax_hi'), Gmax, 1e-9, 'rel', m, t_s);
tab = xc_table(d, 't_corners');
corners = {'650/700', '800/800', '920/850'};
for c = 1:numel(corners)
  row = local_find_row(tab, corners{c});
  v = sscanf(corners{c}, '%f/%f');
  tank = local_tank(L1, C1, Lm, n, L1 / n ^ 2, C1 * n ^ 2, v(1), P);
  [Gc, fc] = local_max_inductive_gain(tank, fgrid);
  tok = regexp(row{3}, '([0-9]+\.[0-9]+) @ ([0-9]+\.[0-9]+) kHz', 'tokens', 'once');
  rows{end + 1} = xc_row(sprintf('FL10.seed.gmax_%s', strrep(corners{c}, '/', '_')), d, sprintf('max inductive |H| at %s V (table t_corners)', corners{c}), '', ...
    str2double(tok{1}), Gc, 5e-7 + 1e-12, 'abs', [m '; table prints 6 decimals'], 0); %#ok<AGROW>
  rows{end + 1} = xc_row(sprintf('FL10.seed.fgmax_%s', strrep(corners{c}, '/', '_')), d, sprintf('frequency of that maximum at %s V (table t_corners)', corners{c}), 'Hz', ...
    str2double(tok{2}) * 1e3, fc, 5 + 1e-6, 'abs', [m '; table prints kHz with 2 decimals'], 0); %#ok<AGROW>
end

d = xc_load(export_dir, 'FL10', 'fix_n093', 'textbook');
L1 = xc_get(d, 'input', 'L1');
C1 = xc_get(d, 'input', 'C1');
Lm = xc_get(d, 'input', 'Lm');
n = xc_get(d, 'input', 'n');
P = xc_get(d, 'input', 'P');
L2 = L1 / n ^ 2;
C2 = C1 * n ^ 2;
rows{end + 1} = xc_row('FL10.n093.L2', d, 'physical L2 = L1/n^2', 'H', xc_get(d, 'metric', 'L2'), L2, 1e-12, 'rel', 'referred symmetry', 0);
rows{end + 1} = xc_row('FL10.n093.C2', d, 'physical C2 = C1 n^2', 'F', xc_get(d, 'metric', 'C2'), C2, 1e-12, 'rel', 'referred symmetry', 0);
ms = xc_metrics(d);
names = {'lo', 'mid', 'hi'};
m = 'sign changes of |H| - Greq on a 9001-point grid, fzero (TolX 1e-9 Hz), inductive roots kept';
for c = 1:3
  Vb = xc_get(d, 'input', ['Vbat_' names{c}]);
  Vl = xc_get(d, 'input', ['Vlink_' names{c}]);
  t0 = tic;
  tank = local_tank(L1, C1, Lm, n, L2, C2, Vb, P);
  fr = local_roots(tank, fgrid, n * Vb / Vl);
  t_c = toc(t0);
  key = sprintf('%g/%g V', Vb, Vl);
  exp_f = [];
  for k = 1:numel(ms)
    if strncmp(ms{k}.key, 'f_', 2) && ~isempty(strfind(ms{k}.label, key))
      exp_f(end + 1) = ms{k}.value; %#ok<AGROW>
    end
  end
  exp_f = sort(exp_f);
  rows{end + 1} = xc_row(sprintf('FL10.n093.%g_%g.n_roots', Vb, Vl), d, sprintf('number of inductive FHA solutions at %s', key), '', numel(exp_f), numel(fr), 0, 'abs', m, t_c); %#ok<AGROW>
  for j = 1:min(numel(fr), numel(exp_f))
    rows{end + 1} = xc_row(sprintf('FL10.n093.%g_%g.root%d', Vb, Vl, j), d, sprintf('inductive FHA solution %d at %s', j, key), 'Hz', exp_f(j), fr(j), 1e-9, 'rel', m, t_c); %#ok<AGROW>
  end
  if c == 3 && numel(fr) == 2
    for j = 1:2
      s = (abs(local_H(tank, fr(j) + 10)) - abs(local_H(tank, fr(j) - 10))) / 20 * 1e3;
      I1 = 2 * sqrt(2) / pi * Vl / abs(local_Zin(tank, fr(j)));
      kk = {'slope_lo', 'slope_hi'};
      ki = {'I1_lo', 'I1_hi'};
      rows{end + 1} = xc_row(sprintf('FL10.n093.920_850.slope%d', j), d, sprintf('d|H|/df at solution %d (+-10 Hz)', j), '1/kHz', xc_get(d, 'metric', kk{j}), s, 1e-6, 'rel', ...
        'central difference of |H| over +-10 Hz (same definition as the app)', 0); %#ok<AGROW>
      rows{end + 1} = xc_row(sprintf('FL10.n093.920_850.I1rms%d', j), d, sprintf('FHA primary series rms at solution %d', j), 'A', xc_get(d, 'metric', ki{j}), I1, 1e-9, 'rel', ...
        'I1 = (2 sqrt(2)/pi) Vlink / |Zin|', 0); %#ok<AGROW>
    end
  end
end
end

function row = local_find_row(tab, key)
for k = 1:numel(tab)
  if ~isempty(strfind(tab{k}{1}, key))
    row = tab{k};
    return;
  end
end
error('xc:cllc', 'no table row with "%s"', key);
end
