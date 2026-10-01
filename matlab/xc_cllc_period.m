function out = xc_cllc_period(p, x0, ode_opts)
%XC_CLLC_PERIOD One switching period of the ideal CLLC, integrated with ode45.
%
%   Primary-referred T network of textbook E05:
%     v1 -> R1, L1, C1 -> node m (shunt Lm) -> L2, C2, R2 -> diode bridge -> output
%     L1 di1/dt = v1 - vC1 - R1 i1 - vm,   L2 di2/dt = vm - vC2 - R2 i2 - v2
%     C1 dvC1/dt = i1,  C2 dvC2/dt = i2,   Lm (di1/dt - di2/dt) = vm
%   so in a conduction mode
%     vm = [(v1 - vC1 - R1 i1)/L1 + (vC2 + v2 + R2 i2)/L2] / (1/L1 + 1/Lm + 1/L2).
%   Ideal full bridge: v1 = +Vin on [0, T/2), -Vin on [T/2, T).
%   Ideal diode bridge with clamp Vc = n*Vo (stiff battery) or n*vo (R-C output):
%     mode P: i2 > 0, v2 = +Vc;   mode N: i2 < 0, v2 = -Vc;
%     mode O: all diodes off, i2 = 0, (L1 + Lm) di1/dt = v1 - vC1 - R1 i1 and the
%             rectifier input floats at vr = Lm/(L1 + Lm) (v1 - vC1 - R1 i1) - vC2.
%   Transitions: P/N end when i2 crosses zero; then N (or P) if vr is beyond the
%   opposite clamp, else O. O ends when vr reaches +Vc (-> P) or -Vc (-> N).
%   At a bridge edge vr jumps, so mode O is re-checked there.
%   R-C output (actual values): Co dvo/dt = n |i2| - vo/Rout.
%
%   p: L1 C1 Lm L2 C2 R1 R2 (referred), Vin, f, n, out ('stiff' or 'rc'),
%      Vo (stiff, actual) or Co, Rout (rc, actual)
%   x0: [i1; i2; vC1; vC2] (+ vo for 'rc') at t = 0, the rising edge of v1.
%   out: xT, P_rect (mean power into the clamp, actual W), I1_rms, off_frac,
%        vo_avg (rc), vC1_pk, vC2_pk (max |v| over the period), P_in, n_events.

T = 1 / p.f;
rc = strcmp(p.out, 'rc');
nx = 4 + rc;
y = [x0(:); zeros(5, 1)];          % accumulators: int p_rect, int i1^2, int off, int vo, int p_in
mode = local_start_mode(p, y, 1);
pk1 = abs(y(3));
pk2 = abs(y(4));
nev = 0;
t = 0;
for half = 1:2
  b1 = 3 - 2 * half;               % +1 then -1
  t_end = half * T / 2;
  if mode == 0
    mode = local_start_mode(p, y, b1);   % vr jumps with v1
  end
  guard = 0;
  while t < t_end
    guard = guard + 1;
    if guard > 200
      error('xc:cllc', 'more than 200 mode changes in half a period (f = %g Hz)', p.f);
    end
    term = local_terminal(mode);
    opts = odeset(ode_opts, 'Events', @(tt, yy) local_events(p, yy, b1, mode));
    [tt, yy, te, ye, ie] = ode45(@(tt, yy) local_rhs(p, yy, b1, mode), [t, t_end], y, opts);
    k_term = find(ismember(ie, term), 1);
    if ~isempty(k_term)
      % Octave does not stop on an event found in the first step of a call
      % (MATLAB compatibility), so the first terminal event is taken from the
      % returned list and the samples after it are discarded.
      t_ev = te(k_term);
      y_ev = ye(k_term, :).';
      keep = tt <= t_ev;
      tt = [tt(keep); t_ev];
      yy = [yy(keep, :); y_ev.'];
      before = te < t_ev;
      te = te(before);
      ye = ye(before, :);
    end
    pk1 = max([pk1; abs(yy(:, 3))]);
    pk2 = max([pk2; abs(yy(:, 4))]);
    if ~isempty(ye)
      pk1 = max([pk1; abs(ye(:, 3))]);   % vC1 extrema sit on the i1 = 0 events
      pk2 = max([pk2; abs(ye(:, 4))]);
    end
    if isempty(k_term)
      t = t_end;
      y = yy(end, :).';
    else
      nev = nev + 1;
      t = t_ev;
      y = y_ev;
      y(2) = 0;                    % the event is i2 = 0 (P/N end) or starts from i2 = 0 (O end)
      mode = local_next_mode(p, y, b1, mode, ie(k_term));
    end
  end
end
out.xT = y(1:nx);
out.P_rect = y(nx + 1) / T;
out.I1_rms = sqrt(y(nx + 2) / T);
out.off_frac = y(nx + 3) / T;
out.vo_avg = y(nx + 4) / T;
out.P_in = y(nx + 5) / T;
out.vC1_pk = pk1;
out.vC2_pk = pk2;
out.n_events = nev;
out.mode_end = mode;
end

% ======================================================================
function vc = local_clamp(p, y)
if strcmp(p.out, 'rc')
  vc = p.n * y(5);
else
  vc = p.n * p.Vo;
end
end

function vr = local_vr(p, y, b1)
% rectifier input voltage with all diodes off (i2 = 0)
a = b1 * p.Vin - y(3) - p.R1 * y(1);
vr = p.Lm / (p.L1 + p.Lm) * a - y(4);
end

function m = local_start_mode(p, y, b1)
if y(2) > 0
  m = 1;
elseif y(2) < 0
  m = -1;
else
  vr = local_vr(p, y, b1);
  vc = local_clamp(p, y);
  if vr > vc
    m = 1;
  elseif vr < -vc
    m = -1;
  else
    m = 0;
  end
end
end

function m = local_next_mode(p, y, b1, mode, idx)
vr = local_vr(p, y, b1);
vc = local_clamp(p, y);
switch mode
  case 1                            % i2 fell to zero
    if vr < -vc
      m = -1;
    else
      m = 0;
    end
  case -1                           % i2 rose to zero
    if vr > vc
      m = 1;
    else
      m = 0;
    end
  otherwise                         % off: a clamp was reached
    if idx == 1
      m = 1;
    else
      m = -1;
    end
end
end

function t = local_terminal(mode)
if mode == 0
  t = [1, 2];
else
  t = 1;
end
end

function [value, isterminal, direction] = local_events(p, y, b1, mode)
if mode == 0
  vr = local_vr(p, y, b1);
  vc = local_clamp(p, y);
  value = [vr - vc; vr + vc; y(1)];
  isterminal = [1; 1; 0];
  direction = [1; -1; 0];
else
  value = [y(2); y(1)];
  isterminal = [1; 0];
  direction = [-mode; 0];          % P ends falling, N ends rising
end
end

function dy = local_rhs(p, y, b1, mode)
rc = strcmp(p.out, 'rc');
i1 = y(1);
i2 = y(2);
vc = local_clamp(p, y);
a = b1 * p.Vin - y(3) - p.R1 * i1;
if mode == 0
  di1 = a / (p.L1 + p.Lm);
  di2 = 0;
else
  c = y(4) + mode * vc + p.R2 * i2;
  vm = (a / p.L1 + c / p.L2) / (1 / p.L1 + 1 / p.Lm + 1 / p.L2);
  di1 = (a - vm) / p.L1;
  di2 = (vm - c) / p.L2;
end
d = [di1; di2; i1 / p.C1; i2 / p.C2];
if rc
  vo = y(5);
  d(5) = (mode * p.n * i2 - vo / p.Rout) / p.Co;   % mode*i2 = |i2| in conduction
else
  vo = 0;
end
dy = [d; mode * vc * i2; i1 ^ 2; double(mode == 0); vo; b1 * p.Vin * i1];
end
