function v = xc_get(d, kind, key)
%XC_GET Value of an exported input ('input') or metric ('metric').
%   Stops with a clear message when the key is absent, so a renamed metric
%   in the Python app cannot pass silently.

switch kind
  case 'input'
    m = d.inputs;
  case 'metric'
    m = d.metrics;
  otherwise
    error('xc:kind', 'kind must be ''input'' or ''metric'', got %s', kind);
end
if ~isKey(m, key)
  error('xc:missingKey', '%s "%s" not found in %s', kind, key, d.file);
end
v = m(key);
end
