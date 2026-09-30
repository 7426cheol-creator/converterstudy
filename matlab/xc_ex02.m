function rows = xc_ex02(export_dir)
%XC_EX02 EX02 nonlinear Coss (textbook E02): C(v) = C0/sqrt(1 + v/V0).
%   Qoss = int_0^V C dv, Eoss = int_0^V v C dv by numerical quadrature and in
%   closed form; half-bridge commutation with fixed rails and a constant
%   current I: C_node(v) = C(v) + C(Vb - v) (+ Cpar), C_node dv/dt = I,
%   integrated with ode45 until the node reaches the rail (event).

rows = {};
qopt = {'AbsTol', 1e-22, 'RelTol', 1e-13};

% ---------------------------------------------------------------- charge and energy
d = xc_load(export_dir, 'EX02', 'qe_integrals', 'textbook');
C0 = xc_get(d, 'input', 'C0');
V0 = xc_get(d, 'input', 'V0');
Vb = xc_get(d, 'input', 'Vb');
C = @(v) C0 ./ sqrt(1 + v ./ V0);

t0 = tic;
Q_int = integral(C, 0, Vb, qopt{:});
E_int = integral(@(v) v .* C(v), 0, Vb, qopt{:});
t_int = toc(t0);
m = 'integral() of C(v) and v*C(v), textbook E02 definitions';
rows{end + 1} = xc_row('EX02.Qoss.integral', d, 'Qoss(Vb)', 'C', xc_get(d, 'metric', 'Qoss'), Q_int, 1e-10, 'rel', m, t_int);
rows{end + 1} = xc_row('EX02.Eoss.integral', d, 'Eoss(Vb)', 'J', xc_get(d, 'metric', 'Eoss'), E_int, 1e-10, 'rel', m, t_int);

t0 = tic;
u = 1 + Vb / V0;
Q_cf = 2 * C0 * V0 * (sqrt(u) - 1);
E_cf = C0 * V0 ^ 2 * ((2 / 3) * u ^ 1.5 - 2 * sqrt(u) + 4 / 3);
Cpt = C(Vb);
t_cf = toc(t0);
m = 'closed form: Q = 2 C0 V0 (sqrt(u)-1), E = C0 V0^2 (2/3 u^1.5 - 2 u^0.5 + 4/3), u = 1 + Vb/V0';
rows{end + 1} = xc_row('EX02.Qoss.closed_form', d, 'Qoss(Vb)', 'C', xc_get(d, 'metric', 'Qoss'), Q_cf, 1e-12, 'rel', m, t_cf);
rows{end + 1} = xc_row('EX02.Eoss.closed_form', d, 'Eoss(Vb)', 'J', xc_get(d, 'metric', 'Eoss'), E_cf, 1e-12, 'rel', m, t_cf);
rows{end + 1} = xc_row('EX02.Cotr.closed_form', d, 'Co(tr) = Q/V', 'F', xc_get(d, 'metric', 'Cotr'), Q_cf / Vb, 1e-12, 'rel', 'Co(tr) = Qoss/V', t_cf);
rows{end + 1} = xc_row('EX02.Coer.closed_form', d, 'Co(er) = 2E/V^2', 'F', xc_get(d, 'metric', 'Coer'), 2 * E_cf / Vb ^ 2, 1e-12, 'rel', 'Co(er) = 2 Eoss/V^2', t_cf);
rows{end + 1} = xc_row('EX02.Q_pt.closed_form', d, 'single-point Q = C(Vb) Vb', 'C', xc_get(d, 'metric', 'Q_pt'), Cpt * Vb, 1e-12, 'rel', 'differential C at Vb used as if constant (the error E02 warns about)', t_cf);
rows{end + 1} = xc_row('EX02.E_pt.closed_form', d, 'single-point E = C(Vb) Vb^2/2', 'J', xc_get(d, 'metric', 'E_pt'), Cpt * Vb ^ 2 / 2, 1e-12, 'rel', 'differential C at Vb used as if constant', t_cf);

% ---------------------------------------------------------------- half-bridge transition
d = xc_load(export_dir, 'EX02', 'hb_constant_current', 'textbook');
C0 = xc_get(d, 'input', 'C0');
V0 = xc_get(d, 'input', 'V0');
Vb = xc_get(d, 'input', 'Vb');
I = xc_get(d, 'input', 'I');
td = xc_get(d, 'input', 'td');
Cpar = xc_get(d, 'input', 'Cpar');
C = @(v) C0 ./ sqrt(1 + v ./ V0);
Cnode = @(v) C(v) + C(Vb - v) + Cpar;
f = @(t, v) I / Cnode(v);

t0 = tic;
t_guess = 2 * Vb * C0 / I;                   % upper bound: C_node <= 2 C0
opts = odeset('RelTol', 1e-12, 'AbsTol', 1e-12, 'Events', @(t, v) local_rail(t, v, Vb));
[~, ~, te] = ode45(f, [0, 2 * t_guess], 0, opts);
t_ode = toc(t0);
if isempty(te)
  error('xc:ex02', 'EX02: the node did not reach the rail in the ode45 window');
end
m = 'ode45 of C_node(v) dv/dt = I from v = 0, event v = Vb';
rows{end + 1} = xc_row('EX02.t_trans.ode45', d, 'transition time at I', 's', xc_get(d, 'metric', 't_trans'), te(1), 1e-8, 'rel', m, t_ode);

t0 = tic;
[~, v] = ode45(f, [0, td], 0, odeset('RelTol', 1e-12, 'AbsTol', 1e-12));
t_td = toc(t0);
m = 'ode45 of C_node(v) dv/dt = I from v = 0 to t = td';
rows{end + 1} = xc_row('EX02.v_td.ode45', d, 'node voltage at the end of dead time', 'V', xc_get(d, 'metric', 'v_td'), v(end), 1e-8, 'rel', m, t_td);
rows{end + 1} = xc_row('EX02.vds_on.ode45', d, 'Vds of the incoming device at turn-on', 'V', xc_get(d, 'metric', 'vds_on'), Vb - v(end), 1e-8, 'rel', m, t_td);

t0 = tic;
Qn = integral(Cnode, 0, Vb, qopt{:});
t_q = toc(t0);
m = 'integral of C_node over 0..Vb (= 2 Qoss when Cpar = 0), t = Q/I';
rows{end + 1} = xc_row('EX02.Qreq.integral', d, 'node charge', 'C', xc_get(d, 'metric', 'Qreq'), Qn, 1e-10, 'rel', m, t_q);
rows{end + 1} = xc_row('EX02.t_trans.integral', d, 'transition time at I', 's', xc_get(d, 'metric', 't_trans'), Qn / I, 1e-10, 'rel', m, t_q);
rows{end + 1} = xc_row('EX02.t_pt.closed_form', d, 'single-point estimate 2 C(Vb) Vb / I', 's', xc_get(d, 'metric', 't_pt'), 2 * C(Vb) * Vb / I, 1e-12, 'rel', 'differential C at Vb used as if constant', 0);
end

function [value, isterminal, direction] = local_rail(~, v, Vb)
value = v - Vb;
isterminal = 1;
direction = 1;
end
