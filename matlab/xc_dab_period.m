function out = xc_dab_period(V1, V2, L, fs, w1, w2, phi, Lm)
%XC_DAB_PERIOD One switching period of the ideal DAB, integrated with ode45.
%
%   Textbook ch.11 and E06 (switching-function form):
%     v1  = V1 * b(theta; w1, 0),   v2' = V2 * b(theta; w2, phi),   theta = 2*pi*fs*t
%     L di/dt = v1 - v2'
%     P2 = (1/T) * int v2' * i2 dt,   Irms = sqrt((1/T) * int i^2 dt)
%   Lm > 0 adds the magnetizing branch across v2' (Lm dim/dt = v2'); the
%   current into bridge 2 is then i2 = i - im. Lm = 0 means no branch.
%
%   An ideal L does not fix the DC offset, so the textbook's zero-mean
%   (half-wave antisymmetric) steady state is selected: the period is
%   integrated from zero current and the mean is removed afterwards, using
%   the running integrals carried as extra ode45 states.
%   The bridge voltages are constant between edges, so ode45 runs edge to
%   edge (no step across a discontinuity).

if nargin < 8
  Lm = 0;
end
T = 1 / fs;
wsw = 2 * pi * fs;

% bridge edges: s(x) toggles where x = 0 or pi (mod 2*pi)
base = [-w1 / 2, w1 / 2, phi - w2 / 2, phi + w2 / 2];
ed = mod([base, base + pi], 2 * pi);
ed(ed > 2 * pi - 1e-12) = 0;
ed = sort([0, ed]);
ed = ed([true, diff(ed) > 1e-12]);
ed = [ed, 2 * pi];

opts = odeset('RelTol', 1e-11, 'AbsTol', 1e-14);
% y = [ih; imh; Si; Sm; Sii; Smm; Sim; Sv1i; Sv2i2; Sv1; Sv2]
y = zeros(11, 1);
tt = [];
ii = [];
mm = [];
ied = zeros(1, numel(ed));   % series current (before the offset) at each edge
for k = 1:numel(ed) - 1
  ta = ed(k) / wsw;
  tb = ed(k + 1) / wsw;
  th = 0.5 * (ed(k) + ed(k + 1));
  v1 = V1 * xc_bridge(th, w1, 0);
  v2 = V2 * xc_bridge(th, w2, phi);
  f = @(t, x) local_rhs(x, v1, v2, L, Lm);
  [t, x] = ode45(f, [ta, tb], y, opts);
  y = x(end, :).';
  ied(k + 1) = y(1);
  tt = [tt; t(:)]; %#ok<AGROW>
  ii = [ii; x(:, 1)]; %#ok<AGROW>
  mm = [mm; x(:, 2)]; %#ok<AGROW>
end

Si = y(3); Sm = y(4); Sii = y(5); Smm = y(6); Sim = y(7);
Sv1i = y(8); Sv2i2 = y(9); Sv1 = y(10); Sv2 = y(11);
i0 = -Si / T;          % zero-mean offset of the series current
m0 = -Sm / T;          % zero-mean offset of the magnetizing current
c2 = i0 - m0;

out.i0 = i0;           % current at theta = 0 of the E06 frame (centre of the v1 pulse)
% currents at the rising edges (textbook ch.11 frame: i0 at the v1 edge, i_phi at the v2' edge)
out.i_v1_rise = local_at(ed, ied + i0, mod(-w1 / 2, 2 * pi));
out.i_v2_rise = local_at(ed, ied + i0, mod(phi - w2 / 2, 2 * pi));
out.Irms = sqrt((Sii + 2 * i0 * Si + i0 ^ 2 * T) / T);
out.Ipk = max(abs(ii + i0));
out.Im_rms = sqrt(max(Smm + 2 * m0 * Sm + m0 ^ 2 * T, 0) / T);
S22 = Sii - 2 * Sim + Smm;
out.I2_rms = sqrt((S22 + 2 * c2 * (Si - Sm) + c2 ^ 2 * T) / T);
out.P1 = (Sv1i + i0 * Sv1) / T;
out.P2 = (Sv2i2 + c2 * Sv2) / T;
out.resid = y(1);      % i(T) - i(0): volt-second balance of the period
out.edges = ed;
out.t = tt;
out.i = ii + i0;
end

function v = local_at(ed, iv, th)
[dist, k] = min(abs(ed - th));
if dist > 1e-9
  error('xc:edge', 'no bridge edge at theta = %g', th);
end
v = iv(k);
end

function dx = local_rhs(x, v1, v2, L, Lm)
ih = x(1);
imh = x(2);
di = (v1 - v2) / L;
if Lm > 0
  dm = v2 / Lm;
else
  dm = 0;
end
dx = [di; dm; ih; imh; ih ^ 2; imh ^ 2; ih * imh; v1 * ih; v2 * (ih - imh); v1; v2];
end
