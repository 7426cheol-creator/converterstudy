function rows = xc_fl05(export_dir)
%XC_FL05 FL05 OBC input boundary and DC-link sizing (textbook ch.08).
%   I_line = P_bat / (sqrt(3) V_LL eta PF), SPWM m = 2 sqrt(2) V_phase / V_dc,
%   single-phase ripple dV_pp ~ P/(w C V), hold-up C >= 2 P dt / (V_hi^2 - V_lo^2).
%   Time-domain paths: three-phase instantaneous power averaged by integral()
%   with fzero on the line current; the capacitor energy equation
%   C v dv/dt = p_in(t) - p_out integrated with ode45 (ripple and hold-up).

rows = {};

% ---------------------------------------------------------------- grid boundary
d = xc_load(export_dir, 'FL05', 'grid_boundary', 'nominal');
Pb = xc_get(d, 'input', 'P_bat');
VLL = xc_get(d, 'input', 'V_LL');
eta = xc_get(d, 'input', 'eta');
PF = xc_get(d, 'input', 'PF');
Ilim = xc_get(d, 'input', 'I_lim');
VLL_lo = xc_get(d, 'input', 'V_LL_low');
Vdc = xc_get(d, 'input', 'V_dc');
VLL_hi = xc_get(d, 'input', 'V_LL_hi');
Vdc_lo = xc_get(d, 'input', 'V_dc_lo');
f = xc_get(d, 'input', 'f');

t0 = tic;
I_line = Pb / (sqrt(3) * VLL * eta * PF);
P_lim = sqrt(3) * VLL * Ilim * eta * PF;
I_low = Pb / (sqrt(3) * VLL_lo * eta * PF);
Vph = VLL / sqrt(3);
m_nom = 2 * sqrt(2) * Vph / Vdc;
m_cor = 2 * sqrt(2) * (VLL_hi / sqrt(3)) / Vdc_lo;
t_cf = toc(t0);
m = 'textbook ch.08: I = P_bat/(sqrt(3) V_LL eta PF), m = 2 sqrt(2) V_ph/V_dc';
rows{end + 1} = xc_row('FL05.grid.I_line.closed_form', d, 'line current at 400 V', 'A', xc_get(d, 'metric', 'I_line'), I_line, 1e-12, 'rel', m, t_cf);
rows{end + 1} = xc_row('FL05.grid.P_at_lim.closed_form', d, 'battery power at the current limit', 'W', xc_get(d, 'metric', 'P_at_lim'), P_lim, 1e-12, 'rel', m, t_cf);
rows{end + 1} = xc_row('FL05.grid.I_low.closed_form', d, 'line current at low line', 'A', xc_get(d, 'metric', 'I_low'), I_low, 1e-12, 'rel', m, t_cf);
rows{end + 1} = xc_row('FL05.grid.V_ph.closed_form', d, 'phase voltage', 'V', xc_get(d, 'metric', 'V_ph'), Vph, 1e-12, 'rel', m, t_cf);
rows{end + 1} = xc_row('FL05.grid.m_nom.closed_form', d, 'SPWM modulation, nominal', '', xc_get(d, 'metric', 'm_nom'), m_nom, 1e-12, 'rel', m, t_cf);
rows{end + 1} = xc_row('FL05.grid.m_cor.closed_form', d, 'SPWM modulation, 440 V LL / 700 V DC', '', xc_get(d, 'metric', 'm_cor'), m_cor, 1e-12, 'rel', m, t_cf);

t0 = tic;
w = 2 * pi * f;
ph1 = acos(PF);                              % sinusoidal current: PF = displacement factor
pmean = @(I) f * integral(@(t) local_p3(t, w, Vph, I, ph1), 0, 1 / f, 'AbsTol', 1e-9, 'RelTol', 1e-13);
I_td = fzero(@(I) pmean(I) - Pb / eta, [0.5 * I_line, 2 * I_line], optimset('TolX', 1e-14));
t_td = toc(t0);
rows{end + 1} = xc_row('FL05.grid.I_line.time_domain', d, 'line current at 400 V', 'A', xc_get(d, 'metric', 'I_line'), I_td, 1e-10, 'rel', 'mean of sum_k v_k(t) i_k(t) over one grid period = P_bat/eta, fzero on I', t_td);

% ---------------------------------------------------------------- DC link
d = xc_load(export_dir, 'FL05', 'dclink_power', 'nominal');
f = xc_get(d, 'input', 'f');
w = 2 * pi * f;
P1 = xc_get(d, 'input', 'P1');
P3 = xc_get(d, 'input', 'P3');
Vph = xc_get(d, 'input', 'V_LL') / sqrt(3);
Prip = xc_get(d, 'input', 'P_rip');
Vrip = xc_get(d, 'input', 'V_rip');
dVpp = xc_get(d, 'input', 'dV_pp');
Phold = xc_get(d, 'input', 'P_hold');
dt_hold = xc_get(d, 'input', 'dt_hold');
Vhi = xc_get(d, 'input', 'V_hi');
Vlo = xc_get(d, 'input', 'V_lo');
Cg = xc_get(d, 'input', 'C_given');

% instantaneous power: single phase pulsates at 2w, balanced three phase is constant
t0 = tic;
tt = linspace(0, 1 / f, 20001);
p1 = local_p1(tt, w, Vph, P1 / Vph);
p3 = local_p3(tt, w, Vph, P3 / (3 * Vph), 0);
t_p = toc(t0);
m = 'sampled v(t) i(t) over one grid period, unity PF';
rows{end + 1} = xc_row('FL05.dclink.p1_ripple.samples', d, 'single-phase power ripple amplitude', 'W', xc_get(d, 'metric', 'p1_ripple'), (max(p1) - min(p1)) / 2, 1e-9, 'rel', m, t_p);
rows{end + 1} = xc_row('FL05.dclink.p1_mean.samples', d, 'single-phase mean power', 'W', xc_get(d, 'metric', 'p1_mean'), mean(p1(1:end - 1)), 1e-12, 'rel', m, t_p);
rows{end + 1} = xc_row('FL05.dclink.p3_ripple.samples', d, 'three-phase power ripple amplitude', 'W', xc_get(d, 'metric', 'p3_ripple'), (max(p3) - min(p3)) / 2, 1e-6, 'abs', m, t_p);

t0 = tic;
Crip = Prip / (w * Vrip * dVpp);
Chold = 2 * Phold * dt_hold / (Vhi ^ 2 - Vlo ^ 2);
dE = 0.5 * Cg * (Vhi ^ 2 - Vlo ^ 2);
t_cf = toc(t0);
rows{end + 1} = xc_row('FL05.dclink.C_rip.closed_form', d, 'C for dV_pp (single phase)', 'F', xc_get(d, 'metric', 'C_rip'), Crip, 1e-12, 'rel', 'textbook ch.08: dV_pp ~ P/(w C V_dc)', t_cf);
rows{end + 1} = xc_row('FL05.dclink.C_hold.closed_form', d, 'hold-up C', 'F', xc_get(d, 'metric', 'C_hold'), Chold, 1e-12, 'rel', 'textbook ch.08: C >= 2 P dt/(V_hi^2 - V_lo^2)', t_cf);
rows{end + 1} = xc_row('FL05.dclink.dE.closed_form', d, 'usable energy of C_given', 'J', xc_get(d, 'metric', 'dE'), dE, 1e-12, 'rel', 'C (V_hi^2 - V_lo^2)/2', t_cf);
rows{end + 1} = xc_row('FL05.dclink.dE_frac.closed_form', d, 'usable share of the stored energy', '%', xc_get(d, 'metric', 'dE_frac'), 100 * dE / (0.5 * Cg * Vhi ^ 2), 1e-12, 'rel', '100 dE / (C V_hi^2/2)', t_cf);

% ripple: C v dv/dt = -P cos(2 w t), v(0) = V at the energy centre; extrema where cos(2wt) = 0
t0 = tic;
opts = odeset('RelTol', 1e-12, 'AbsTol', 1e-10, 'Events', @(t, v) local_ext(t, w));
[~, ~, ~, ve] = ode45(@(t, v) -Prip * cos(2 * w * t) / (Crip * v), [0, 1 / f], Vrip, opts);
t_r = toc(t0);
rows{end + 1} = xc_row('FL05.dclink.pp_ode.ode45', d, 'ripple p-p with C_rip, energy equation', 'V', xc_get(d, 'metric', 'pp_ode'), max(ve) - min(ve), 1e-9, 'rel', sprintf('ode45 of C v dv/dt = -P cos(2wt), %d extrema by events', numel(ve)), t_r);

% hold-up: C v dv/dt = -P until v = V_lo
t0 = tic;
opts = odeset('RelTol', 1e-12, 'AbsTol', 1e-10, 'Events', @(t, v) local_floor(t, v, Vlo));
[~, ~, te] = ode45(@(t, v) -Phold / (Cg * v), [0, 10 * dt_hold], Vhi, opts);
t_h = toc(t0);
if isempty(te)
  te = NaN;
end
rows{end + 1} = xc_row('FL05.dclink.t_given.ode45', d, 'hold-up time of C_given at P_hold', 's', xc_get(d, 'metric', 't_given'), te(1), 1e-8, 'rel', 'ode45 of C v dv/dt = -P from V_hi, event v = V_lo', t_h);
end

function p = local_p3(t, w, Vph, I, ph1)
p = zeros(size(t));
for k = 0:2
  a = w * t - 2 * pi * k / 3;
  p = p + sqrt(2) * Vph * cos(a) .* sqrt(2) * I .* cos(a - ph1);
end
end

function p = local_p1(t, w, V, I)
p = (sqrt(2) * V * cos(w * t)) .* (sqrt(2) * I * cos(w * t));
end

function [value, isterminal, direction] = local_ext(t, w)
value = cos(2 * w * t);
isterminal = 0;
direction = 0;
end

function [value, isterminal, direction] = local_floor(~, v, Vlo)
value = v - Vlo;
isterminal = 1;
direction = -1;
end
