function rows = xc_dab(export_dir)
%XC_DAB FL08 DAB, single-phase shift: closed form (textbook ch.11) and ode45.
%   Presets: nominal (800 V / 48 V, n = 50/3, 1.5 kW per module) and
%   mismatch (900 V / 36 V, phi = 0, Lm = 2 mH).

rows = {};
fzopt = optimset('TolX', 1e-15);

% ---------------------------------------------------------------- nominal
d = xc_load(export_dir, 'FL08', 'sps_nominal', 'nominal');
n = xc_get(d, 'input', 'Np') / xc_get(d, 'input', 'Ns');
V1 = xc_get(d, 'input', 'VH');
V2 = n * xc_get(d, 'input', 'VL');           % primary-referred LV bridge voltage
VL = xc_get(d, 'input', 'VL');
L = xc_get(d, 'input', 'L');
fs = xc_get(d, 'input', 'fs');
P = xc_get(d, 'input', 'P');                 % per module
wL = 2 * pi * fs * L;

% closed form, textbook ch.11: P = V1 V2 phi (1 - phi/pi) / (w L)
t0 = tic;
K = V1 * V2 / wL;
phi_cf = pi / 2 * (1 - sqrt(1 - 4 * P / (pi * K)));
a = (V1 + V2) / wL;
b = (V1 - V2) / wL;
i0 = -(a * phi_cf + b * (pi - phi_cf)) / 2;
iphi = i0 + a * phi_cf;
ipi = -i0;
% piecewise-linear current: int i^2 over a segment = dt (ia^2 + ia ib + ib^2) / 3
Irms_cf = sqrt((phi_cf * (i0 ^ 2 + i0 * iphi + iphi ^ 2) + (pi - phi_cf) * (iphi ^ 2 + iphi * ipi + ipi ^ 2)) / (3 * pi));
Ipk_cf = max(abs([i0, iphi]));
Pmax_cf = V1 * V2 / (8 * fs * L);
t_cf = toc(t0);
m = 'textbook ch.11 closed form: phi from P = V1 V2 phi (1-phi/pi)/(w L); i0 = -(a phi + b (pi-phi))/2';
rows{end + 1} = xc_row('FL08.nominal.phi.closed_form', d, 'phi', 'rad', xc_get(d, 'metric', 'phi'), phi_cf, 1e-9, 'rel', m, t_cf);
rows{end + 1} = xc_row('FL08.nominal.Irms.closed_form', d, 'Irms', 'A', xc_get(d, 'metric', 'Irms'), Irms_cf, 1e-9, 'rel', m, t_cf);
rows{end + 1} = xc_row('FL08.nominal.Ipk.closed_form', d, 'Ipk', 'A', xc_get(d, 'metric', 'Ipk'), Ipk_cf, 1e-9, 'rel', m, t_cf);
rows{end + 1} = xc_row('FL08.nominal.Pmax.closed_form', d, 'Pmax', 'W', xc_get(d, 'metric', 'Pmax'), Pmax_cf, 1e-12, 'rel', 'Pmax = V1 V2 / (8 fs L)', t_cf);

% ode45: phi found by fzero on the time-domain power P2(phi) = P
t0 = tic;
f = @(ph) local_p2(V1, V2, L, fs, ph) - P;
phi_ode = fzero(f, [1e-3, pi / 2], fzopt);
r = xc_dab_period(V1, V2, L, fs, pi, pi, phi_ode, 0);
t_ode = toc(t0);
m = 'ode45 edge to edge, L di/dt = v1 - v2''; zero-mean steady state; fzero on P2(phi) = P';
rows{end + 1} = xc_row('FL08.nominal.phi.ode45', d, 'phi', 'rad', xc_get(d, 'metric', 'phi'), phi_ode, 1e-8, 'rel', m, t_ode);
rows{end + 1} = xc_row('FL08.nominal.Irms.ode45', d, 'Irms', 'A', xc_get(d, 'metric', 'Irms'), r.Irms, 1e-8, 'rel', m, t_ode);
rows{end + 1} = xc_row('FL08.nominal.Ipk.ode45', d, 'Ipk', 'A', xc_get(d, 'metric', 'Ipk'), r.Ipk, 1e-8, 'rel', m, t_ode);
rows{end + 1} = xc_row('FL08.nominal.i0.ode45', d, 'i at v1 rising edge', 'A', xc_get(d, 'metric', 'i0'), r.i_v1_rise, 1e-8, 'rel', m, t_ode);
rows{end + 1} = xc_row('FL08.nominal.iphi.ode45', d, 'i at v2'' rising edge', 'A', xc_get(d, 'metric', 'iphi'), r.i_v2_rise, 1e-8, 'rel', m, t_ode);
rows{end + 1} = xc_row('FL08.nominal.Is_rms.ode45', d, 'secondary AC rms n*Irms', 'A', xc_get(d, 'metric', 'Is_rms'), n * r.Irms, 1e-8, 'rel', [m '; ideal transformer, no Lm'], t_ode);
rows{end + 1} = xc_row('FL08.nominal.Io_dc.ode45', d, 'LV DC current P2/VL', 'A', xc_get(d, 'metric', 'Io_dc'), r.P2 / VL, 1e-8, 'rel', [m '; lossless bridge'], t_ode);

% power at the app's own phi, integrated here
t0 = tic;
r2 = xc_dab_period(V1, V2, L, fs, pi, pi, xc_get(d, 'metric', 'phi'), 0);
t2 = toc(t0);
rows{end + 1} = xc_row('FL08.nominal.P2.ode45_at_app_phi', d, 'P2 at the exported phi', 'W', xc_get(d, 'metric', 'P_pwl'), r2.P2, 1e-8, 'rel', 'ode45 P2 = (1/T) int v2'' i dt at phi from the export', t2);

% ---------------------------------------------------------------- ratio mismatch, phi = 0
d = xc_load(export_dir, 'FL08', 'zero_power_mismatch', 'mismatch');
n = xc_get(d, 'input', 'Np') / xc_get(d, 'input', 'Ns');
V1 = xc_get(d, 'input', 'VH');
V2 = n * xc_get(d, 'input', 'VL');
L = xc_get(d, 'input', 'L');
fs = xc_get(d, 'input', 'fs');
phi = xc_get(d, 'input', 'phi');
Lm = xc_get(d, 'input', 'Lm');

t0 = tic;
Ipk_cf = abs(V1 - V2) / (4 * fs * L);      % triangle, slope (V1-V2)/L for T/2
Irms_cf = Ipk_cf / sqrt(3);
t_cf = toc(t0);
rows{end + 1} = xc_row('FL08.mismatch.Irms_L.closed_form', d, 'Irms_L', 'A', xc_get(d, 'metric', 'Irms_L'), Irms_cf, 1e-9, 'rel', 'textbook ch.11: phi = 0, v_L = +-(V1-V2), Ipk = (V1-V2)/(4 fs L), Irms = Ipk/sqrt(3)', t_cf);

t0 = tic;
r = xc_dab_period(V1, V2, L, fs, pi, pi, phi, Lm);
t_ode = toc(t0);
m = 'ode45 edge to edge with Lm across v2''; zero-mean steady state';
rows{end + 1} = xc_row('FL08.mismatch.Irms_L.ode45', d, 'Irms_L', 'A', xc_get(d, 'metric', 'Irms_L'), r.Irms, 1e-8, 'rel', m, t_ode);
rows{end + 1} = xc_row('FL08.mismatch.Ipk_L.ode45', d, 'Ipk_L', 'A', xc_get(d, 'metric', 'Ipk_L'), r.Ipk, 1e-8, 'rel', m, t_ode);
rows{end + 1} = xc_row('FL08.mismatch.Im_rms.ode45', d, 'magnetizing rms', 'A', xc_get(d, 'metric', 'Im_rms'), r.Im_rms, 1e-8, 'rel', m, t_ode);
rows{end + 1} = xc_row('FL08.mismatch.I2_rms.ode45', d, 'transformer current i - im, rms', 'A', xc_get(d, 'metric', 'I2_rms'), r.I2_rms, 1e-8, 'rel', m, t_ode);
rows{end + 1} = xc_row('FL08.mismatch.P.ode45', d, 'transferred power', 'W', xc_get(d, 'metric', 'P'), r.P2, 1e-6, 'abs', m, t_ode);
end

function p = local_p2(V1, V2, L, fs, phi)
r = xc_dab_period(V1, V2, L, fs, pi, pi, phi, 0);
p = r.P2;
end
