function name = sc_set(blk, key, value, unit)
%SC_SET Set one block parameter, found at run time.
%   key is either the parameter name (custom +xcsc components, whose names
%   are declared in the .ssc files) or a regular expression matched against
%   the dialog prompts of the block (library blocks, e.g. '^Inductance').
%   The unit goes to the companion <name>_unit parameter when there is one.
%   Stops with the list of prompts when nothing or more than one matches.

dp = get_param(blk, 'DialogParameters');
fn = fieldnames(dp);
if any(strcmp(fn, key))
  name = key;
else
  hits = {};
  for k = 1:numel(fn)
    if isempty(regexp(fn{k}, '_(unit|specify|priority|conf)$', 'once')) && ...
       ~isempty(regexpi(local_prompt(dp.(fn{k})), key, 'once'))
      hits{end + 1} = fn{k}; %#ok<AGROW>
    end
  end
  if numel(hits) ~= 1
    error('sc:param', 'block %s: %d parameters match "%s". Prompts:\n  %s', ...
          blk, numel(hits), key, local_list(dp, fn));
  end
  name = hits{1};
end
if isnumeric(value)
  value = sprintf('%.17g', value);
end
set_param(blk, name, value);
if nargin >= 4 && ~isempty(unit) && any(strcmp(fn, [name '_unit']))
  set_param(blk, [name '_unit'], unit);
end
end

function p = local_prompt(s)
p = '';
if isstruct(s) && isfield(s, 'Prompt')
  p = s.Prompt;
end
end

function s = local_list(dp, fn)
c = cell(1, numel(fn));
for k = 1:numel(fn)
  c{k} = sprintf('%s: %s', fn{k}, local_prompt(dp.(fn{k})));
end
s = strjoin(c, '\n  ');
end
