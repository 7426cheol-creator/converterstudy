function m = xc_metrics(d)
%XC_METRICS All metrics of a loaded export with their labels: a cell array of
%   structs (key, label, value, unit). xc_load keeps only key -> value; some
%   checks need the label text (it carries e.g. the corner or the frequency).

j = jsondecode(fileread(d.file));
m = j.result.metrics;
if isstruct(m)
  m = num2cell(reshape(m, 1, []));
else
  m = reshape(m, 1, []);
end
end
