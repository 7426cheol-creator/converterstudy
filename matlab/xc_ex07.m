function rows = xc_ex07(export_dir)
%XC_EX07 EX07 constant-power load behind an R-L-C input filter (textbook E07):
%   L di/dt = Vs - R i - v,  C dv/dt = i - P/v
%   Ve = (Vs + sqrt(Vs^2 - 4 R P))/2, g = P/Ve^2,
%   A = [-R/L, -1/L; 1/C, g/C], stable only if C > L g / R.
%   Poles by eig(A); the nonlinear model by ode45 from Ve + dv0, with the
%   growth rate and the damped frequency read from the voltage extrema.

rows = {};
presets = {'c100u', 'c1m'};
for p = 1:numel(presets)
  d = xc_load(export_dir, 'EX07', 'cpl_exact', presets{p});
  Vs = xc_get(d, 'input', 'Vs');
  R = xc_get(d, 'input', 'R');
  L = xc_get(d, 'input', 'L');
  C = xc_get(d, 'input', 'C');
  P = xc_get(d, 'input', 'P');
  dv0 = xc_get(d, 'input', 'dv0');
  t_end = xc_get(d, 'input', 't_end');
  dv_valid = xc_get(d, 'input', 'dv_valid');
  tag = ['EX07.' presets{p} '.'];

  % ------------------------------------------------------------ equilibrium and poles
  t0 = tic;
  Ve = (Vs + sqrt(Vs ^ 2 - 4 * R * P)) / 2;
  Ie = P / Ve;
  g = P / Ve ^ 2;
  A = [-R / L, -1 / L; 1 / C, g / C];
  lam = eig(A);
  [~, k] = max(imag(lam));
  lam = lam(k);
  t_eig = toc(t0);
  m = 'textbook E07: Ve = (Vs + sqrt(Vs^2 - 4RP))/2, g = P/Ve^2, C_crit = L g/R, eig(A)';
  rows{end + 1} = xc_row([tag 'Ve'], d, 'Ve', 'V', xc_get(d, 'metric', 'Ve'), Ve, 1e-12, 'rel', m, t_eig); %#ok<AGROW>
  rows{end + 1} = xc_row([tag 'Rinc'], d, 'incremental resistance -Ve^2/P', 'Ohm', xc_get(d, 'metric', 'Rinc'), -1 / g, 1e-12, 'rel', m, t_eig); %#ok<AGROW>
  rows{end + 1} = xc_row([tag 'Ccrit'], d, 'critical C = L g / R', 'F', xc_get(d, 'metric', 'Ccrit'), L * g / R, 1e-12, 'rel', m, t_eig); %#ok<AGROW>
  rows{end + 1} = xc_row([tag 'Pmax'], d, 'Pmax = Vs^2/(4R)', 'W', xc_get(d, 'metric', 'Pmax'), Vs ^ 2 / (4 * R), 1e-12, 'rel', m, t_eig); %#ok<AGROW>
  rows{end + 1} = xc_row([tag 'pole_re.eig'], d, 'pole real part', '1/s', xc_get(d, 'metric', 'pole_re'), real(lam), 1e-10, 'rel', m, t_eig); %#ok<AGROW>
  rows{end + 1} = xc_row([tag 'pole_im.eig'], d, 'pole imaginary part', 'rad/s', xc_get(d, 'metric', 'pole_im'), imag(lam), 1e-10, 'rel', m, t_eig); %#ok<AGROW>

  % ------------------------------------------------------------ nonlinear ode45
  t0 = tic;
  f = @(t, x) [(Vs - R * x(1) - x(2)) / L; (x(1) - P / x(2)) / C];
  ev = @(t, x) local_events(x, P, Ve, dv_valid);
  opts = odeset('RelTol', 1e-11, 'AbsTol', 1e-10, 'Events', ev, 'MaxStep', 2 * pi / abs(imag(lam)) / 50);
  [~, ~, te, xe, ie] = ode45(f, [0, t_end], [Ie; Ve + dv0], opts);
  t_nl = toc(t0);
  ext = (ie == 1);
  te_x = te(ext);
  dv_x = xe(ext, 2) - Ve;
  small = abs(dv_x) < 0.05 * Ve;             % stay in the small-signal range
  te_x = te_x(small);
  dv_x = dv_x(small);
  if numel(te_x) < 4
    error('xc:ex07', 'EX07 %s: fewer than 4 small-amplitude extrema', presets{p});
  end
  cf = polyfit(te_x, log(abs(dv_x)), 1);
  sig = cf(1);
  wd = pi / mean(diff(te_x));
  m = sprintf('ode45 of the nonlinear model from Ve + %g V; slope of log|extrema| (%d extrema with |dv| < 5%% Ve)', dv0, numel(te_x));
  rows{end + 1} = xc_row([tag 'sigma.ode45_nonlinear'], d, 'growth rate vs linear pole', '1/s', xc_get(d, 'metric', 'pole_re'), sig, 1e-3, 'rel', m, t_nl); %#ok<AGROW>
  rows{end + 1} = xc_row([tag 'wd.ode45_nonlinear'], d, 'damped frequency vs linear pole', 'rad/s', xc_get(d, 'metric', 'pole_im'), wd, 1e-3, 'rel', m, t_nl); %#ok<AGROW>
  rows{end + 1} = xc_row([tag 'unstable.ode45_nonlinear'], d, 'grows (1) / decays (0); expected from app status', '', double(strcmp(d.status, 'UNSTABLE')), double(sig > 0), 0, 'bool', m, t_nl); %#ok<AGROW>
  if isKey(d.metrics, 't_exit')
    te_exit = te(ie == 2);
    if isempty(te_exit)
      te_exit = NaN;
    end
    rows{end + 1} = xc_row([tag 't_exit.ode45_nonlinear'], d, sprintf('time |dv| reaches %g Ve', dv_valid), 's', xc_get(d, 'metric', 't_exit'), te_exit(1), 1e-6, 'rel', 'ode45 event |v - Ve| = dv_valid Ve on the nonlinear model', t_nl); %#ok<AGROW>
  end
end
end

function [value, isterminal, direction] = local_events(x, P, Ve, dv_valid)
% 1: extremum of v (C dv/dt = i - P/v = 0); 2: leaving the model's validity band
value = [x(1) - P / x(2); abs(x(2) - Ve) - dv_valid * Ve];
isterminal = [0; 1];
direction = [0; 1];
end
