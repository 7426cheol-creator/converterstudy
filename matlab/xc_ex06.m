function rows = xc_ex06(export_dir)
%XC_EX06 EX06 DAB width modulation, textbook E06 example (900 V / 600 V', 1.5 kW):
%   SPS (w1 = w2 = pi) against the closed form, and the candidate
%   w1 = 0.7*pi, w2 = pi by ode45 (smallest phase root of P2(phi) = P) and by a
%   Fourier series of the two bridge voltages.

rows = {};
fzopt = optimset('TolX', 1e-15);
d = xc_load(export_dir, 'EX06', 'general_modulation', 'textbook');
V1 = xc_get(d, 'input', 'V1');
V2 = xc_get(d, 'input', 'V2');               % primary-referred V2'
L = xc_get(d, 'input', 'L');
fs = xc_get(d, 'input', 'fs');
P = xc_get(d, 'input', 'P');
w1 = xc_get(d, 'input', 'w1') * pi;          % exported as a fraction of pi
w2 = xc_get(d, 'input', 'w2') * pi;
wL = 2 * pi * fs * L;

% ---------------------------------------------------------------- SPS closed form (ch.11)
t0 = tic;
phi_s = pi / 2 * (1 - sqrt(1 - 4 * P * wL / (pi * V1 * V2)));
a = (V1 + V2) / wL;
b = (V1 - V2) / wL;
i0 = -(a * phi_s + b * (pi - phi_s)) / 2;
iphi = i0 + a * phi_s;
ipi = -i0;
irms_s = sqrt((phi_s * (i0 ^ 2 + i0 * iphi + iphi ^ 2) + (pi - phi_s) * (iphi ^ 2 + iphi * ipi + ipi ^ 2)) / (3 * pi));
ipk_s = max(abs([i0, iphi]));
t_cf = toc(t0);
m = 'textbook ch.11 SPS closed form with V1 ~= V2 (piecewise-linear current)';
rows{end + 1} = xc_row('EX06.sps.phi.closed_form', d, 'phi_sps', 'rad', xc_get(d, 'metric', 'phi_sps'), phi_s, 1e-9, 'rel', m, t_cf);
rows{end + 1} = xc_row('EX06.sps.irms.closed_form', d, 'irms_sps', 'A', xc_get(d, 'metric', 'irms_sps'), irms_s, 1e-9, 'rel', m, t_cf);
rows{end + 1} = xc_row('EX06.sps.ipk.closed_form', d, 'ipk_sps', 'A', xc_get(d, 'metric', 'ipk_sps'), ipk_s, 1e-9, 'rel', m, t_cf);

% ---------------------------------------------------------------- SPS by ode45
t0 = tic;
phi_so = fzero(@(ph) local_p2(V1, V2, L, fs, pi, pi, ph) - P, [1e-3, pi / 2], fzopt);
rs = xc_dab_period(V1, V2, L, fs, pi, pi, phi_so, 0);
t_so = toc(t0);
m = 'ode45 edge to edge, v1 = V1 b(theta;pi,0), v2'' = V2 b(theta;pi,phi); fzero on P2(phi) = P';
rows{end + 1} = xc_row('EX06.sps.phi.ode45', d, 'phi_sps', 'rad', xc_get(d, 'metric', 'phi_sps'), phi_so, 1e-8, 'rel', m, t_so);
rows{end + 1} = xc_row('EX06.sps.irms.ode45', d, 'irms_sps', 'A', xc_get(d, 'metric', 'irms_sps'), rs.Irms, 1e-8, 'rel', m, t_so);
rows{end + 1} = xc_row('EX06.sps.ipk.ode45', d, 'ipk_sps', 'A', xc_get(d, 'metric', 'ipk_sps'), rs.Ipk, 1e-8, 'rel', m, t_so);

% ---------------------------------------------------------------- candidate by ode45
t0 = tic;
grid = linspace(0, pi / 2, 33);
pg = zeros(size(grid));
for k = 2:numel(grid)
  pg(k) = local_p2(V1, V2, L, fs, w1, w2, grid(k));
end
k1 = find(pg >= P, 1, 'first');
if isempty(k1)
  error('xc:ex06', 'EX06 candidate: P2(phi) never reaches %g W on (0, pi/2]', P);
end
phi_c = fzero(@(ph) local_p2(V1, V2, L, fs, w1, w2, ph) - P, [grid(k1 - 1), grid(k1)], fzopt);
rc = xc_dab_period(V1, V2, L, fs, w1, w2, phi_c, 0);
t_c = toc(t0);
m = sprintf('ode45 edge to edge, w1 = %.3g pi, w2 = %.3g pi; smallest root of P2(phi) = P (grid scan + fzero)', w1 / pi, w2 / pi);
rows{end + 1} = xc_row('EX06.candidate.phi.ode45', d, 'phi_c', 'rad', xc_get(d, 'metric', 'phi_c'), phi_c, 1e-8, 'rel', m, t_c);
rows{end + 1} = xc_row('EX06.candidate.irms.ode45', d, 'irms_c', 'A', xc_get(d, 'metric', 'irms_c'), rc.Irms, 1e-8, 'rel', m, t_c);
rows{end + 1} = xc_row('EX06.candidate.ipk.ode45', d, 'ipk_c', 'A', xc_get(d, 'metric', 'ipk_c'), rc.Ipk, 1e-8, 'rel', m, t_c);
rows{end + 1} = xc_row('EX06.candidate.drms.ode45', d, 'irms_c/irms_sps - 1', '', xc_get(d, 'metric', 'drms'), rc.Irms / rs.Irms - 1, 1e-7, 'rel', 'ratio of the two ode45 results above', t_c + t_so);

% ---------------------------------------------------------------- candidate by Fourier series
% b(theta; w, phi) = sum_{k odd} (4/(k pi)) sin(k w/2) cos(k (theta - phi))
% P2 = sum_{k odd} 8 V1 V2 sin(k w1/2) sin(k w2/2) sin(k phi) / (pi^2 k^3 w L)
t0 = tic;
phi_app = xc_get(d, 'metric', 'phi_c');
k = 1:2:399999;
A1 = 4 * V1 ./ (k * pi) .* sin(k * w1 / 2);
A2 = 4 * V2 ./ (k * pi) .* sin(k * w2 / 2);
Pf = sum(A1 .* A2 .* sin(k * phi_app) ./ (2 * k * wL));
Ik = abs(A1 - A2 .* exp(-1i * k * phi_app)) ./ (k * wL);
Irms_f = sqrt(sum(Ik .^ 2) / 2);
t_f = toc(t0);
m = 'Fourier series of v1 and v2'' (200000 odd harmonics) at the exported phi_c';
rows{end + 1} = xc_row('EX06.candidate.P.fourier', d, 'P_c', 'W', xc_get(d, 'metric', 'P_c'), Pf, 1e-8, 'rel', m, t_f);
rows{end + 1} = xc_row('EX06.candidate.irms.fourier', d, 'irms_c', 'A', xc_get(d, 'metric', 'irms_c'), Irms_f, 1e-8, 'rel', m, t_f);
end

function p = local_p2(V1, V2, L, fs, w1, w2, phi)
r = xc_dab_period(V1, V2, L, fs, w1, w2, phi, 0);
p = r.P2;
end
