function ok = run_all(export_dir, out_file)
%RUN_ALL Run every Octave/MATLAB cross-check and write the evidence file.
%
%   ok = run_all()                       uses matlab/results/python_export
%   ok = run_all(export_dir, out_file)
%
%   export_dir  folder with the Python exports
%               (python -m convlab run LAB EXP --preset P --out export_dir)
%   out_file    evidence JSON (default matlab/results/octave_crosscheck.json)
%   ok          true when no item is FAIL; TODO items are listed, not failed.
%
%   Every check recomputes its value from the textbook equations (closed
%   forms, ode45, integral, fzero, eig) and compares it with the number the
%   Python app exported. Nothing is copied from the Python code.
%   verification/run_octave_crosscheck.sh exports the JSON and calls this.

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
if nargin < 1 || isempty(export_dir)
  export_dir = fullfile(here, 'results', 'python_export');
end
if nargin < 2 || isempty(out_file)
  out_file = fullfile(here, 'results', 'octave_crosscheck.json');
end
addpath(here);

checks = {'xc_dab', 'xc_ex06', 'xc_ex02', 'xc_ex07', 'xc_llc', 'xc_cllc', 'xc_fl05', 'xc_fl01'};
rows = {};
groups = {};
t_all = tic;
for k = 1:numel(checks)
  t0 = tic;
  msg = '';
  try
    r = feval(checks{k}, export_dir);
    state = 'ran';
  catch err
    r = {local_error_row(checks{k}, err)};
    state = 'error';
    msg = err.message;
  end
  rows = [rows, r]; %#ok<AGROW>
  g = struct('check', checks{k}, 'items', numel(r), 'runtime_s', round(toc(t0) * 1e3) / 1e3, 'state', state, 'message', msg);
  groups{end + 1} = g; %#ok<AGROW>
  fprintf('%-8s %3d items  %7.3f s  %s %s\n', checks{k}, numel(r), toc(t0), state, msg);
end

st = cellfun(@(x) x.status, rows, 'UniformOutput', false);
n_pass = sum(strcmp(st, 'PASS'));
n_fail = sum(strcmp(st, 'FAIL'));
n_todo = sum(strcmp(st, 'TODO'));

meta = struct();
meta.schema = 'convlab-octave-crosscheck/1';
meta.generated_at_local = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
[meta.engine, meta.engine_version] = local_engine();
meta.platform = computer();
try
  meta.blas = version('-blas');
catch
  meta.blas = '';
end
meta.export_dir = local_rel(export_dir, root);
meta.python_exports = local_exports(export_dir);
meta.expected_values = ['Python exports (python -m convlab run LAB EXP --preset P); ' ...
  'FL10 CLLC FHA rows use the textbook ch.13 printed values because FL10 is not merged yet.'];
meta.tolerance_policy = ['closed form 1e-9..1e-12 rel; ode45 / fzero 1e-8 rel; printed table or textbook ' ...
  'values: half a unit of the last printed digit (abs); nonlinear growth-rate fits 1e-3 rel.'];
meta.checks = groups;
meta.counts = struct('total', numel(rows), 'PASS', n_pass, 'FAIL', n_fail, 'TODO', n_todo);
meta.runtime_s = round(toc(t_all) * 1e3) / 1e3;

local_write(out_file, meta, rows);

fprintf('\n%-5s %-44s %-16s %-16s %-9s\n', 'state', 'item', 'expected', 'octave', 'error');
for k = 1:numel(rows)
  r = rows{k};
  if strcmp(r.tol_kind, 'rel') && isfinite(r.rel_error)
    e = sprintf('%.1e rel', r.rel_error);
  elseif isfinite(r.abs_error)
    e = sprintf('%.1e abs', r.abs_error);
  else
    e = '-';
  end
  fprintf('%-5s %-44s %-16.10g %-16.10g %-9s\n', r.status, r.item, r.expected, r.octave_value, e);
end
fprintf('\n%d items: %d PASS, %d FAIL, %d TODO  (%s %s, %.1f s)\nwritten: %s\n', ...
  numel(rows), n_pass, n_fail, n_todo, meta.engine, meta.engine_version, meta.runtime_s, out_file);
ok = (n_fail == 0);
end

% ======================================================================
function r = local_error_row(check, err)
src = struct('lab', check, 'experiment', '-', 'preset', '-', 'file', '-');
r = xc_row([check '.ERROR'], src, 'check did not run', '', 1, NaN, 0, 'abs', ['ERROR: ' err.message], 0, '-');
end

function [name, ver] = local_engine()
if exist('OCTAVE_VERSION', 'builtin') > 0
  name = 'GNU Octave';
  ver = OCTAVE_VERSION();
else
  name = 'MATLAB';
  ver = version();
end
end

function s = local_rel(p, root)
if strncmp(p, [root filesep], numel(root) + 1)
  s = p(numel(root) + 2:end);
else
  s = p;
end
end

function list = local_exports(export_dir)
list = {};
files = dir(fullfile(export_dir, '*.json'));
names = sort({files.name});
for k = 1:numel(names)
  e = struct('file', names{k});
  try
    j = jsondecode(fileread(fullfile(export_dir, names{k})));
    e.lab = j.lab;
    e.experiment = j.experiment;
    e.preset = j.preset;
    e.cli = sprintf('python -m convlab run %s %s --preset %s', j.lab, j.experiment, j.preset);
    pv = j.provenance;
    keys = {'app_version', 'contract_version', 'request_hash', 'timestamp', 'python', 'numpy', 'scipy'};
    for q = 1:numel(keys)
      if isfield(pv, keys{q})
        e.(keys{q}) = pv.(keys{q});
      end
    end
  catch err
    e.error = err.message;
  end
  list{end + 1} = e; %#ok<AGROW>
end
end

function local_write(fname, meta, rows)
d = fileparts(fname);
if ~isempty(d) && exist(d, 'dir') ~= 7
  mkdir(d);
end
fid = fopen(fname, 'w');
if fid < 0
  error('xc:write', 'cannot write %s', fname);
end
fprintf(fid, '{\n');
fn = fieldnames(meta);
for k = 1:numel(fn)
  v = meta.(fn{k});
  if iscell(v) && ~isempty(v)
    fprintf(fid, '  "%s": [\n', fn{k});
    for q = 1:numel(v)
      if q < numel(v)
        sep = ',';
      else
        sep = '';
      end
      fprintf(fid, '    %s%s\n', xc_json(v{q}), sep);
    end
    fprintf(fid, '  ],\n');
  else
    fprintf(fid, '  "%s": %s,\n', fn{k}, xc_json(v));
  end
end
fprintf(fid, '  "items": [\n');
for k = 1:numel(rows)
  if k < numel(rows)
    sep = ',';
  else
    sep = '';
  end
  fprintf(fid, '    %s%s\n', xc_json(rows{k}), sep);
end
fprintf(fid, '  ]\n}\n');
fclose(fid);
end
