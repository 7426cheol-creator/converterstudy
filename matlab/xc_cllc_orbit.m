function orb = xc_cllc_orbit(p, x_start, opt)
%XC_CLLC_ORBIT Periodic steady state of the ideal CLLC (xc_cllc_period) by
%   Newton shooting on the period map x0 -> Phi_T(x0), section at the rising
%   edge of v1.
%
%   orb = xc_cllc_orbit(p)              start from the first-harmonic phasor state
%   orb = xc_cllc_orbit(p, x_start)     start from a nearby orbit (continuation)
%   orb = xc_cllc_orbit(p, x_start, opt)
%     opt.ode      odeset for ode45 (default RelTol 1e-10, AbsTol 1e-9,
%                  MaxStep T/100)
%     opt.J        Jacobian of the period map to start with (from a nearby orbit)
%     opt.tol      scaled residual max|Phi_T(x0) - x0| / scale (default 1e-10)
%     opt.floquet  also return the monodromy by central differences and its
%                  largest |eigenvalue| (default false)
%   The Jacobian is taken by forward differences (step 1e-6 of the state
%   scale) and reused while the residual keeps falling by 3x per step (chord
%   Newton); the map is piecewise smooth (event times move with the state).

if nargin < 3
  opt = struct();
end
T = 1 / p.f;
rc = strcmp(p.out, 'rc');
nx = 4 + rc;
if ~isfield(opt, 'ode')
  opt.ode = odeset('RelTol', 1e-10, 'AbsTol', 1e-9, 'MaxStep', T / 100, 'InitialStep', T * 1e-5);
end
if ~isfield(opt, 'tol')
  opt.tol = 1e-10;
end
if ~isfield(opt, 'floquet')
  opt.floquet = false;
end
Z0 = sqrt(p.L1 / p.C1);
sc = [p.Vin / Z0; p.Vin / Z0; p.Vin; p.Vin];
if rc
  sc(5) = p.Vin / p.n;
end
if nargin < 2 || isempty(x_start)
  x = local_phasor_guess(p);
else
  x = x_start(:);
end
J = [];
if isfield(opt, 'J')
  J = opt.J;
end
res = Inf;
res_prev = Inf;
n_eval = 0;
it = 0;
for it = 1:40
  r = xc_cllc_period(p, x, opt.ode);
  n_eval = n_eval + 1;
  F = r.xT - x;
  res = max(abs(F) ./ sc);
  if res < opt.tol
    break;
  end
  if isempty(J) || res > res_prev / 3
    J = zeros(nx);
    for j = 1:nx
      h = 1e-6 * sc(j);
      xj = x;
      xj(j) = xj(j) + h;
      J(:, j) = (xc_cllc_period(p, xj, opt.ode).xT - r.xT) / h;
      n_eval = n_eval + 1;
    end
  end
  res_prev = res;
  x = x - (J - eye(nx)) \ F;
end
orb.x0 = x;
orb.sum = r;
orb.residual = res;
orb.iterations = it;
orb.n_eval = n_eval;
orb.converged = res < opt.tol;
orb.J = J;
orb.scale = sc;
if opt.floquet
  M = zeros(nx);
  for j = 1:nx
    h = 1e-4 * sc(j);
    xp = x;
    xm = x;
    xp(j) = xp(j) + h;
    xm(j) = xm(j) - h;
    M(:, j) = (xc_cllc_period(p, xp, opt.ode).xT - xc_cllc_period(p, xm, opt.ode).xT) / (2 * h);
  end
  orb.monodromy = M;
  orb.rho = max(abs(eig(M)));
end
end

% ======================================================================
function x = local_phasor_guess(p)
% First-harmonic phasors of the same T network with the rectifier replaced by
% Rac' = (8/pi^2) x referred DC resistance, sampled at t = 0 (x(t) = Im(X e^{jwt})).
w = 2 * pi * p.f;
if strcmp(p.out, 'rc')
  Rac = 8 / pi ^ 2 * p.n ^ 2 * p.Rout;
else
  Rac = 8 / pi ^ 2 * p.n ^ 2 * p.Vo ^ 2 / p.P_guess;
end
Z1 = p.R1 + 1i * w * p.L1 + 1 / (1i * w * p.C1);
Z2 = p.R2 + 1i * w * p.L2 + 1 / (1i * w * p.C2) + Rac;
Zm = 1i * w * p.Lm;
Zp = Zm * Z2 / (Zm + Z2);
I1 = (4 * p.Vin / pi) / (Z1 + Zp);   % v1 fundamental = (4 Vin/pi) sin(w t)
I2 = I1 * Zp / Z2;
x = imag([I1; I2; I1 / (1i * w * p.C1); I2 / (1i * w * p.C2)]);
if strcmp(p.out, 'rc')
  x(5) = p.vo_guess;
end
end
