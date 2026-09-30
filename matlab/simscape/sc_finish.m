function res = sc_finish(res, opts, t0)
%SC_FINISH Final status from the comparison rows, the result file (when
%   anything was attempted) and the entry in status.json.

if nargin >= 3
  res.runtime_s = round(toc(t0) * 1e3) / 1e3;
end
if strcmp(res.status, 'RAN')
  st = cellfun(@(r) r.status, res.rows, 'UniformOutput', false);
  if any(strcmp(st, 'FAIL'))
    res.status = 'FAIL';
  elseif any(strcmp(st, 'PENDING_EXPECTED'))
    res.status = 'PENDING_EXPECTED';
  else
    res.status = 'PASS';
  end
end
if ~strcmp(res.status, 'NOT_RUN_ENVIRONMENT')
  d = fullfile(opts.here, 'results');
  if exist(d, 'dir') ~= 7
    mkdir(d);
  end
  res.result_file = ['matlab/simscape/results/' res.model '.json'];
  local_write(fullfile(d, [res.model '.json']), res);
end
sc_status(opts.here, res);
fprintf('%-28s %-20s %s\n', res.model, res.status, res.reason);
end

function local_write(fname, res)
fid = fopen(fname, 'w');
if fid < 0
  error('sc:write', 'cannot write %s', fname);
end
fprintf(fid, '{\n');
fn = setdiff(fieldnames(res), {'rows'}, 'stable');
for k = 1:numel(fn)
  fprintf(fid, '  "%s": %s,\n', fn{k}, xc_json(res.(fn{k})));
end
fprintf(fid, '  "rows": [\n');
for k = 1:numel(res.rows)
  sep = ',';
  if k == numel(res.rows)
    sep = '';
  end
  fprintf(fid, '    %s%s\n', xc_json(res.rows{k}), sep);
end
fprintf(fid, '  ]\n}\n');
fclose(fid);
end
