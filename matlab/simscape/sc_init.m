function name = sc_init(blk, prompt_re, value, unit)
%SC_INIT Set a high-priority initial target on a library Simscape block.
%   Initial-target variables are recognised at run time by their companion
%   <name>_specify parameter; prompt_re picks one of them by its prompt
%   (e.g. 'current' for an inductor, 'voltage' for a capacitor).

dp = get_param(blk, 'DialogParameters');
fn = fieldnames(dp);
cand = {};
for k = 1:numel(fn)
  if any(strcmp(fn, [fn{k} '_specify']))
    cand{end + 1} = fn{k}; %#ok<AGROW>
  end
end
hits = {};
for k = 1:numel(cand)
  pr = '';
  if isfield(dp.(cand{k}), 'Prompt')
    pr = dp.(cand{k}).Prompt;
  end
  if ~isempty(regexpi(pr, prompt_re, 'once'))
    hits{end + 1} = cand{k}; %#ok<AGROW>
  end
end
if numel(hits) ~= 1
  error('sc:initial', 'block %s: %d initial-target variables match "%s" (candidates: %s)', ...
        blk, numel(hits), prompt_re, strjoin(cand, ', '));
end
name = hits{1};
set_param(blk, [name '_specify'], 'on');
set_param(blk, name, sprintf('%.17g', value));
if any(strcmp(fn, [name '_unit']))
  set_param(blk, [name '_unit'], unit);
end
pn = [name '_priority'];
if any(strcmp(fn, pn))
  opts = {};
  if isfield(dp.(pn), 'Enum')
    opts = dp.(pn).Enum;
  end
  k = find(~cellfun(@isempty, regexpi(opts, '^high', 'once')), 1);
  if isempty(k)
    error('sc:initial', 'block %s: no "High" option for %s (options: %s)', blk, pn, strjoin(opts, ', '));
  end
  set_param(blk, pn, opts{k});
end
end
