function rows = xc_llc(export_dir)
%XC_LLC FL09 LLC first-harmonic approximation (textbook ch.12).
%   Circuit form:    Zr = jwLr + 1/(jwCr), Zm = jwLm, Zp = Zm || Rac, H = Zp/(Zr + Zp)
%   Normalized form: 1/H = 1 + (1/k)(1 - F^-2) + jQ(F - 1/F), F = f/fr, k = Lm/Lr,
%                    Q = sqrt(Lr/Cr)/Rac
%   Checked against the |H| table (Q = 0.2 / 0.8 / 1.5, F = 0.7 ... 1.5) and the
%   inductive/capacitive boundary column of that table (Im Zin = 0 by fzero).

rows = {};
d = xc_load(export_dir, 'FL09', 'fha_gain', 'textbook');
Lr = xc_get(d, 'input', 'Lr');
Cr = xc_get(d, 'input', 'Cr');
Lm = xc_get(d, 'input', 'Lm');
Vin = xc_get(d, 'input', 'Vin');
Vo = xc_get(d, 'input', 'Vo_nom');
Qsel = xc_get(d, 'input', 'Q');
Fmin = xc_get(d, 'input', 'F_min');
Fmax = xc_get(d, 'input', 'F_max');
if strcmp(xc_get(d, 'input', 'bridge'), 'FB')
  kb = 1;                                    % full bridge: Vo/Vi = |H|/n
else
  kb = 2;                                    % half bridge: Vo/Vi = |H|/(2n)
end

% ---------------------------------------------------------------- scalar definitions
t0 = tic;
fr = 1 / (2 * pi * sqrt(Lr * Cr));
Z0 = sqrt(Lr / Cr);
k = Lm / Lr;
n = Vin / (kb * Vo);
Rac = Z0 / Qsel;
RL = pi ^ 2 / 8 * Rac / n ^ 2;               % Rac = 8 n^2 R_dc / pi^2
t_s = toc(t0);
m = 'textbook ch.12 definitions';
rows{end + 1} = xc_row('FL09.fr', d, 'fr = 1/(2 pi sqrt(Lr Cr))', 'Hz', xc_get(d, 'metric', 'fr'), fr, 1e-12, 'rel', m, t_s);
rows{end + 1} = xc_row('FL09.Z0', d, 'Z0 = sqrt(Lr/Cr)', 'Ohm', xc_get(d, 'metric', 'Z0'), Z0, 1e-12, 'rel', m, t_s);
rows{end + 1} = xc_row('FL09.k', d, 'k = Lm/Lr', '', xc_get(d, 'metric', 'k'), k, 1e-12, 'rel', m, t_s);
rows{end + 1} = xc_row('FL09.n', d, 'n at fr (|H| = 1)', '', xc_get(d, 'metric', 'n'), n, 1e-12, 'rel', m, t_s);
rows{end + 1} = xc_row('FL09.Rac', d, 'Rac = Z0/Q', 'Ohm', xc_get(d, 'metric', 'Rac'), Rac, 1e-12, 'rel', m, t_s);
rows{end + 1} = xc_row('FL09.RL', d, 'R_L = (pi^2/8) Rac / n^2', 'Ohm', xc_get(d, 'metric', 'RL'), RL, 1e-12, 'rel', m, t_s);
rows{end + 1} = xc_row('FL09.P_fr', d, 'output at fr = Vo^2/R_L', 'W', xc_get(d, 'metric', 'P_fr'), Vo ^ 2 / RL, 1e-12, 'rel', m, t_s);

% ---------------------------------------------------------------- |H| table
tab = xc_table(d, 't_gain');
Fs = [0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5];
dev_forms = 0;
for r = 1:numel(tab)
  row = tab{r};
  q = str2double(row{1});
  for j = 1:numel(Fs)
    t0 = tic;
    F = Fs(j);
    Hc = local_H(Lr, Cr, Lm, Z0 / q, 2 * pi * F * fr);
    Hn = 1 / (1 + (1 / k) * (1 - F ^ -2) + 1i * q * (F - 1 / F));
    dev_forms = max(dev_forms, abs(abs(Hc) - abs(Hn)));
    t_h = toc(t0);
    expected = str2double(row{j + 1});
    % the table prints 4 decimals: allow half a unit of the last digit
    rows{end + 1} = xc_row(sprintf('FL09.table.H.Q%g.F%g', q, F), d, sprintf('|H| at Q = %g, F = %g (table, 4 decimals)', q, F), '', expected, abs(Hc), 5e-5 + 1e-12, 'abs', 'phasor circuit Zr, Zm || Rac with the exported Lr, Cr, Lm', t_h); %#ok<AGROW>
  end
end
rows{end + 1} = xc_row('FL09.table.H.normalized_vs_circuit', d, 'max | |H| normalized - |H| circuit | over the table', '', 0, dev_forms, 1e-12, 'abs', 'textbook normalized 1/H formula against the phasor circuit (internal consistency)', 0, 'identity (0)');

% ---------------------------------------------------------------- inductive / capacitive boundary column
for r = 1:numel(tab)
  row = tab{r};
  q = str2double(row{1});
  txt = row{end};
  t0 = tic;
  imz = @(F) imag(local_Zin(Lr, Cr, Lm, Z0 / q, 2 * pi * F * fr));
  Fb = fzero(imz, [0.05, 1.0], optimset('TolX', 1e-14));   % capacitive below Fb
  t_b = toc(t0);
  m = 'fzero on Im Zin(F) = 0, Zin = Zr + Zm || Rac (capacitive below the root)';
  tok = regexp(txt, 'F < ([0-9.]+)', 'tokens', 'once');
  if ~isempty(tok)
    % the app interpolates its 241-point phase grid and prints 3 decimals
    rows{end + 1} = xc_row(sprintf('FL09.table.boundary.Q%g', q), d, sprintf('capacitive below F_b, Q = %g (table text "%s")', q, local_ascii(txt)), '', str2double(tok{1}), Fb, 2e-3, 'abs', m, t_b); %#ok<AGROW>
  else
    % no boundary printed: the whole table range [F_min, F_max] must be inductive
    rows{end + 1} = xc_row(sprintf('FL09.table.boundary.Q%g', q), d, sprintf('whole range F >= %g inductive, Q = %g (F_b = %.5f)', Fmin, q, Fb), '', 1, double(Fb < Fmin), 0, 'bool', m, t_b); %#ok<AGROW>
  end
  if Fb > Fmin && Fb < Fmax && isempty(tok)
    rows{end}.method = [m '; table says all inductive but the root is inside the range'];
  end
end
end

function H = local_H(Lr, Cr, Lm, Rac, w)
Zr = 1i * w * Lr + 1 / (1i * w * Cr);
Zm = 1i * w * Lm;
Zp = Zm * Rac / (Zm + Rac);
H = Zp / (Zr + Zp);
end

function Z = local_Zin(Lr, Cr, Lm, Rac, w)
Zm = 1i * w * Lm;
Z = 1i * w * Lr + 1 / (1i * w * Cr) + Zm * Rac / (Zm + Rac);
end

function s = local_ascii(s)
% keep the evidence file ASCII: non-ASCII bytes of the table text become '?'
s(double(s) > 126) = '?';
end
