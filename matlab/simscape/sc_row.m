function r = sc_row(item, src, quantity, unit, expected, value, tol, tol_kind, method, expected_source)
%SC_ROW Comparison row of a Simscape result (same fields as ../xc_row, with
%   simscape_value in place of octave_value). expected = NaN marks a value
%   that has no expected number yet (status PENDING_EXPECTED).

if nargin < 10
  expected_source = '';
end
if isnan(expected)
  r = xc_row(item, src, quantity, unit, 0, value, 0, 'abs', method, 0, expected_source);
  r.expected = NaN;
  r.abs_error = NaN;
  r.rel_error = NaN;
  r.status = 'PENDING_EXPECTED';
else
  r = xc_row(item, src, quantity, unit, expected, value, tol, tol_kind, method, 0, expected_source);
end
r.simscape_value = r.octave_value;
r = rmfield(r, {'octave_value', 'runtime_s'});
end
