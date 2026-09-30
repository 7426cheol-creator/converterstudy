function sc_status(here, res)
%SC_STATUS Update the entry of RES.model in status.json (machine-readable
%   status of every Simscape model). A model is NOT_RUN_ENVIRONMENT until a
%   MATLAB session with Simulink and Simscape has built and simulated it.

f = fullfile(here, 'status.json');
models = {};
if exist(f, 'file') == 2
  try
    S = jsondecode(fileread(f));
    if isfield(S, 'models')
      m = S.models;
      if isstruct(m)
        models = num2cell(reshape(m, 1, []));
      else
        models = reshape(m, 1, []);
      end
    end
  catch
    models = {};
  end
end
env = res.env;
e = struct();
e.model = res.model;
e.builder = res.builder;
e.status = res.status;
e.reason = res.reason;
e.expected_source = res.expected_source;
e.result_file = res.result_file;
e.engine = strtrim(sprintf('%s %s %s', env.engine, env.version, env.release));
e.updated_at_local = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
k = [];
for q = 1:numel(models)
  if strcmp(models{q}.model, res.model)
    k = q;
  end
end
if isempty(k)
  models{end + 1} = e;
else
  models{k} = e;
end
names = cellfun(@(x) x.model, models, 'UniformOutput', false);
[~, o] = sort(names);
models = models(o);

probe = struct('engine', env.engine, 'version', env.version, 'release', env.release, ...
               'platform', env.platform, 'simulink', env.has_simulink, 'simscape', env.has_simscape, ...
               'simscape_electrical', env.has_simscape_electrical);
fid = fopen(f, 'w');
if fid < 0
  error('sc:write', 'cannot write %s', f);
end
fprintf(fid, '{\n');
fprintf(fid, '  "schema": "convlab-simscape-status/1",\n');
fprintf(fid, '  "rule": %s,\n', xc_json(['NOT_RUN_ENVIRONMENT until a MATLAB session with Simulink and Simscape ' ...
  'has built and simulated the model; writing the .m builders is not a run (textbook ch.18). ' ...
  'No .slx file is kept in the repository.']));
fprintf(fid, '  "last_environment": %s,\n', xc_json(probe));
fprintf(fid, '  "models": [\n');
for q = 1:numel(models)
  sep = ',';
  if q == numel(models)
    sep = '';
  end
  fprintf(fid, '    %s%s\n', xc_json(models{q}), sep);
end
fprintf(fid, '  ]\n}\n');
fclose(fid);
end
