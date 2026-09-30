function results = sc_run_all(opts)
%SC_RUN_ALL Build, run and compare every Simscape cross-check model and
%   update status.json. In a session without Simulink and Simscape (for
%   example GNU Octave) every model is recorded as NOT_RUN_ENVIRONMENT and
%   nothing is built.
%
%   results = sc_run_all()        defaults of sc_opts
%   results = sc_run_all(opts)    e.g. opts.run = false to build and save only
%
%   The expected values are read from the Python exports in
%   matlab/results/python_export: run verification/run_octave_crosscheck.sh
%   (or python -m convlab run ...) first.

if nargin < 1
  opts = struct();
end
opts = sc_opts(opts);
results = {};
results{end + 1} = build_dab('nominal', opts);
results{end + 1} = build_dab('mismatch', opts);
for b = 1:2
  results{end + 1} = build_cllc(b, 'fha', opts); %#ok<AGROW>
  results{end + 1} = build_cllc(b, 'switching', opts); %#ok<AGROW>
end
results{end + 1} = build_ex02_commutation(opts);
results{end + 1} = build_ex07_cpl('c100u', opts);
results{end + 1} = build_ex07_cpl('c1m', opts);
end
