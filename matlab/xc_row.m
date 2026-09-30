function r = xc_row(item, src, quantity, unit, expected, value, tol, tol_kind, method, runtime_s, expected_source)
%XC_ROW One comparison row: the value computed here against an expected value.
%
%   item            short id, e.g. 'FL08.nominal.phi.ode45'
%   src             struct from xc_load (or a struct with lab/experiment/preset/file)
%   quantity, unit  what is compared
%   expected        value from the Python export (or the textbook when stated)
%   value           value computed in this script
%   tol, tol_kind   'rel' : |value-expected|/|expected| <= tol
%                   'abs' : |value-expected| <= tol
%                   'bool': value and expected are both true or both false
%   method          how value was computed (equation / solver)
%   runtime_s       seconds spent on the computation that produced value
%   expected_source optional; defaults to the export file name

if nargin < 11 || isempty(expected_source)
  [~, base, ext] = fileparts(src.file);
  expected_source = ['python export ' base ext];
end
abs_err = abs(value - expected);
if expected ~= 0
  rel_err = abs_err / abs(expected);
else
  rel_err = NaN;
end
switch tol_kind
  case 'rel'
    if expected ~= 0
      ok = rel_err <= tol;
    else
      ok = abs_err <= tol;
    end
  case 'abs'
    ok = abs_err <= tol;
  case 'bool'
    ok = (logical(value) == logical(expected));
    abs_err = double(~ok);
    rel_err = NaN;
  otherwise
    error('xc:tolKind', 'unknown tol_kind %s', tol_kind);
end
if ~isfinite(value)
  ok = false;
end
if ok
  status = 'PASS';
else
  status = 'FAIL';
end
r = struct();
r.item = item;
r.lab = src.lab;
r.experiment = src.experiment;
r.preset = src.preset;
r.quantity = quantity;
r.unit = unit;
r.expected = expected;
r.expected_source = expected_source;
r.octave_value = value;
r.abs_error = abs_err;
r.rel_error = rel_err;
r.tol = tol;
r.tol_kind = tol_kind;
r.status = status;
r.runtime_s = round(runtime_s * 1e6) / 1e6;
r.method = method;
end
