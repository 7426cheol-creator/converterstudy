function rows = xc_fl01(export_dir)
%XC_FL01 FL01 synchronous buck in CCM (textbook ch.03, 48 -> 12 V, 5 A):
%   switched R-L-C circuit, periodic steady state by shooting.
%   States x = [iL; vC]; ESR in series with C, load R = D Vin / Io:
%     vo = (vC + esr iL) R / (R + esr)
%     L diL/dt = vsw - dcr iL - vo,   C dvC/dt = iL - vo/R
%     vsw = Vin - rds_hi iL (on, 0 <= t < D T),  -rds_lo iL (off)
%   One period is integrated with ode45 (on and off intervals separately),
%   the periodic state x0 = Phi_T(x0) is solved from the affine period map;
%   integrals of iL, iL^2 and vo ride along as extra states, and the vo
%   extrema are located with ode45 events.

rows = {};
d = xc_load(export_dir, 'FL01', 'buck_ccm', 'nominal');
if ~strcmp(xc_get(d, 'input', 'rect'), 'sync')
  error('xc:fl01', 'FL01 check models synchronous rectification only');
end
p.Vin = xc_get(d, 'input', 'Vin');
p.D = xc_get(d, 'input', 'D');
p.L = xc_get(d, 'input', 'L');
p.fs = xc_get(d, 'input', 'fs');
p.C = xc_get(d, 'input', 'C');
p.esr = xc_get(d, 'input', 'esr');
p.dcr = xc_get(d, 'input', 'dcr');
p.rhi = xc_get(d, 'input', 'rds_hi');
p.rlo = xc_get(d, 'input', 'rds_lo');
Io = xc_get(d, 'input', 'Io');
p.R = p.D * p.Vin / Io;
T = 1 / p.fs;

% Every interval is linear with a constant input, so the one-period map is
% affine, Phi_T(x) = M x + c. M and c come from three ode45 periods and the
% periodic state solves (I - M) x0 = c; one Newton step then removes the
% round-off of the probes (base MATLAB only, no Optimization Toolbox).
t0 = tic;
c = local_map([0; 0], p);
M = [local_map([1; 0], p) - c, local_map([0; 1], p) - c];
xs = (eye(2) - M) \ c;
xs = xs - (M - eye(2)) \ (local_map(xs, p) - xs);
[x1, s] = local_map(xs, p);
t_ps = toc(t0);
dI0 = (p.Vin - p.D * p.Vin) * p.D / (p.L * p.fs);   % textbook hand formula (constant vo)

resid = max(abs(x1 - xs) ./ [1; p.D * p.Vin]);
m = sprintf('ode45 per interval, periodic state from the affine one-period map x0 = Phi_T(x0) (relative residual %.1e)', resid);
tol = 1e-8;
rows{end + 1} = xc_row('FL01.buck.Vo.ode45', d, 'mean output voltage', 'V', xc_get(d, 'metric', 'Vo'), s.Svo / T, tol, 'rel', m, t_ps);
rows{end + 1} = xc_row('FL01.buck.IL_avg.ode45', d, 'mean inductor current', 'A', xc_get(d, 'metric', 'IL_avg'), s.Si / T, tol, 'rel', m, t_ps);
rows{end + 1} = xc_row('FL01.buck.dI_pp.ode45', d, 'inductor ripple p-p', 'A', xc_get(d, 'metric', 'dI_pp'), s.i_on_end - xs(1), tol, 'rel', m, t_ps);
rows{end + 1} = xc_row('FL01.buck.I_peak.ode45', d, 'inductor peak (end of on time)', 'A', xc_get(d, 'metric', 'I_peak'), s.i_on_end, tol, 'rel', m, t_ps);
rows{end + 1} = xc_row('FL01.buck.I_valley.ode45', d, 'inductor valley (start of on time)', 'A', xc_get(d, 'metric', 'I_valley'), xs(1), tol, 'rel', m, t_ps);
rows{end + 1} = xc_row('FL01.buck.IL_rms.ode45', d, 'inductor rms', 'A', xc_get(d, 'metric', 'IL_rms'), sqrt(s.Sii / T), tol, 'rel', m, t_ps);
rows{end + 1} = xc_row('FL01.buck.Q1_avg.ode45', d, 'high-side switch mean current', 'A', xc_get(d, 'metric', 'Q1_avg'), s.Si_on / T, tol, 'rel', [m '; on interval only'], t_ps);
rows{end + 1} = xc_row('FL01.buck.Q1_rms.ode45', d, 'high-side switch rms current', 'A', xc_get(d, 'metric', 'Q1_rms'), sqrt(s.Sii_on / T), tol, 'rel', [m '; on interval only'], t_ps);
rows{end + 1} = xc_row('FL01.buck.Q2_avg.ode45', d, 'low-side switch mean current', 'A', xc_get(d, 'metric', 'Q2_avg'), (s.Si - s.Si_on) / T, tol, 'rel', [m '; off interval only'], t_ps);
rows{end + 1} = xc_row('FL01.buck.Q2_rms.ode45', d, 'low-side switch rms current', 'A', xc_get(d, 'metric', 'Q2_rms'), sqrt((s.Sii - s.Sii_on) / T), tol, 'rel', [m '; off interval only'], t_ps);
rows{end + 1} = xc_row('FL01.buck.dvo_pp.ode45', d, 'output ripple p-p (C and ESR together)', 'V', xc_get(d, 'metric', 'dvo_pp'), s.vo_max - s.vo_min, 1e-7, 'rel', [m '; vo extrema by ode45 events'], t_ps);

% textbook hand formula (constant vo) against the switched-circuit value: an approximation
rows{end + 1} = xc_row('FL01.buck.dI_pp.hand_formula', d, 'inductor ripple p-p, (Vin - Vo) D/(L fs)', 'A', xc_get(d, 'metric', 'dI_pp'), dI0, 1e-3, 'rel', 'textbook ch.03 hand formula; assumes a constant vo, so only an approximation of the switched circuit', 0);
end

function [x1, s] = local_map(x0, p)
T = 1 / p.fs;
Ton = p.D * T;
opts = odeset('RelTol', 1e-12, 'AbsTol', 1e-13);
% extra states: [Si; Sii; Svo]
y0 = [x0(:); 0; 0; 0];
[~, ya] = ode45(@(t, y) local_rhs(y, p, 1), [0, Ton], y0, opts);
ya = ya(end, :).';
s.i_on_end = ya(1);
s.Si_on = ya(3);
s.Sii_on = ya(4);
[~, yb] = ode45(@(t, y) local_rhs(y, p, 0), [Ton, T], ya, opts);
yb = yb(end, :).';
x1 = yb(1:2);
s.Si = yb(3);
s.Sii = yb(4);
s.Svo = yb(5);
if nargout > 1
  % vo extrema: switching instants and interior points where dvo/dt = 0
  vo = @(y) (y(2) + p.esr * y(1)) * p.R / (p.R + p.esr);
  cand = [vo(y0), vo(ya), vo(yb)];
  eopts = odeset(opts, 'Events', @(t, y) local_dvo(y, p, 1));
  [~, ~, ~, ye] = ode45(@(t, y) local_rhs(y, p, 1), [0, Ton], y0, eopts);
  for k = 1:size(ye, 1)
    cand(end + 1) = vo(ye(k, :).'); %#ok<AGROW>
  end
  eopts = odeset(opts, 'Events', @(t, y) local_dvo(y, p, 0));
  [~, ~, ~, ye] = ode45(@(t, y) local_rhs(y, p, 0), [Ton, T], ya, eopts);
  for k = 1:size(ye, 1)
    cand(end + 1) = vo(ye(k, :).'); %#ok<AGROW>
  end
  s.vo_max = max(cand);
  s.vo_min = min(cand);
end
end

function dy = local_rhs(y, p, on)
iL = y(1);
vC = y(2);
vo = (vC + p.esr * iL) * p.R / (p.R + p.esr);
if on
  vsw = p.Vin - p.rhi * iL;
else
  vsw = -p.rlo * iL;
end
diL = (vsw - p.dcr * iL - vo) / p.L;
dvC = (iL - vo / p.R) / p.C;
dy = [diL; dvC; iL; iL ^ 2; vo];
end

function [value, isterminal, direction] = local_dvo(y, p, on)
dy = local_rhs(y, p, on);
value = dy(2) + p.esr * dy(1);               % proportional to dvo/dt
isterminal = 0;
direction = 0;
end
