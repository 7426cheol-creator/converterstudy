function s = xc_json(v)
%XC_JSON Encode a value as compact JSON (struct, cell, char, logical, numeric).
%   Written by hand because Octave's jsonencode has no pretty printing and
%   the evidence file keeps one item per line. Non-finite numbers -> null.

if ischar(v)
  s = ['"' local_escape(v) '"'];
elseif islogical(v) && isscalar(v)
  if v
    s = 'true';
  else
    s = 'false';
  end
elseif isnumeric(v) || islogical(v)
  v = double(v);
  if isempty(v)
    s = 'null';
  elseif isscalar(v)
    s = local_num(v);
  else
    parts = cell(1, numel(v));
    for k = 1:numel(v)
      parts{k} = local_num(v(k));
    end
    s = ['[' strjoin(parts, ', ') ']'];
  end
elseif iscell(v)
  parts = cell(1, numel(v));
  for k = 1:numel(v)
    parts{k} = xc_json(v{k});
  end
  s = ['[' strjoin(parts, ', ') ']'];
elseif isstruct(v) && isscalar(v)
  fn = fieldnames(v);
  parts = cell(1, numel(fn));
  for k = 1:numel(fn)
    parts{k} = ['"' local_escape(fn{k}) '": ' xc_json(v.(fn{k}))];
  end
  s = ['{' strjoin(parts, ', ') '}'];
elseif isstruct(v)
  parts = cell(1, numel(v));
  for k = 1:numel(v)
    parts{k} = xc_json(v(k));
  end
  s = ['[' strjoin(parts, ', ') ']'];
else
  error('xc:json', 'cannot encode value of class %s', class(v));
end
end

function s = local_num(x)
if ~isfinite(x)
  s = 'null';
elseif x == round(x) && abs(x) < 1e15
  s = sprintf('%d', x);
else
  s = sprintf('%.15g', x);
  if str2double(s) ~= x
    s = sprintf('%.17g', x);   % shortest of the two that round-trips
  end
end
end

function out = local_escape(s)
s = reshape(s, 1, []);
out = '';
for k = 1:numel(s)
  c = s(k);
  switch c
    case '"'
      out = [out '\"']; %#ok<AGROW>
    case '\'
      out = [out '\\']; %#ok<AGROW>
    case char(10)
      out = [out '\n']; %#ok<AGROW>
    case char(13)
      out = [out '\r']; %#ok<AGROW>
    case char(9)
      out = [out '\t']; %#ok<AGROW>
    otherwise
      if double(c) < 32
        out = [out sprintf('\\u%04x', double(c))]; %#ok<AGROW>
      else
        out = [out c]; %#ok<AGROW>
      end
  end
end
end
