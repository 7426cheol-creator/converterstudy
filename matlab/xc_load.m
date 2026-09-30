function d = xc_load(export_dir, lab, experiment, preset)
%XC_LOAD Read one Python export written by
%   python -m convlab run LAB EXPERIMENT --preset PRESET --out EXPORT_DIR
%
%   d = xc_load(export_dir, lab, experiment, preset) returns a struct with
%   fields file, lab, experiment, preset, status, inputs and metrics
%   (containers.Map key -> value), tables (cell of structs) and provenance.
%   Only the numbers are taken from the export; every check recomputes its
%   own value from the textbook equations.

fname = fullfile(export_dir, sprintf('%s_%s_%s.json', lab, experiment, preset));
if exist(fname, 'file') ~= 2
  error('xc:missingExport', ...
        'Python export not found: %s\n(run verification/run_octave_crosscheck.sh, or python -m convlab run %s %s --preset %s --out %s)', ...
        fname, lab, experiment, preset, export_dir);
end
j = jsondecode(fileread(fname));
d.file = fname;
d.lab = lab;
d.experiment = experiment;
d.preset = preset;
d.status = '';
if isfield(j.result, 'status') && isstruct(j.result.status) && isfield(j.result.status, 'code')
  d.status = j.result.status.code;
end
d.inputs = local_map(j.inputs);
d.metrics = local_map(j.result.metrics);
d.tables = local_list(j.result.tables);
d.provenance = j.provenance;
end

function m = local_map(arr)
m = containers.Map('KeyType', 'char', 'ValueType', 'any');
items = local_list(arr);
for k = 1:numel(items)
  m(items{k}.key) = items{k}.value;
end
end

function c = local_list(arr)
% jsondecode returns a struct array when all objects share their fields and
% a cell array otherwise; accept both.
if iscell(arr)
  c = reshape(arr, 1, []);
elseif isstruct(arr)
  c = cell(1, numel(arr));
  for k = 1:numel(arr)
    c{k} = arr(k);
  end
else
  c = {};
end
end
