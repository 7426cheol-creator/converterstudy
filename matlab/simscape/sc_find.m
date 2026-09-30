function path = sc_find(key, names, opts)
%SC_FIND Library block path, looked up at run time (no hard-coded paths).
%   key    role of the block, also the field of opts.blocks that overrides it
%   names  acceptable block names, in order of preference (whitespace in a
%          name also matches a line break in the library)
%   The libraries in opts.libs are loaded if installed and searched in that
%   order; one exact name match wins. Several matches in one library, or none
%   at all, stop the build with a message that says what to pass instead.

if isfield(opts.blocks, key)
  path = opts.blocks.(key);
  if getSimulinkBlockHandle(path, true) == -1
    error('sc:override', 'opts.blocks.%s = ''%s'' is not a loadable block', key, path);
  end
  return;
end
loaded = {};
for k = 1:numel(opts.libs)
  if exist(opts.libs{k}, 'file') == 4
    load_system(opts.libs{k});
    loaded{end + 1} = opts.libs{k}; %#ok<AGROW>
  end
end
if isempty(loaded)
  error('sc:noLibrary', 'none of the libraries {%s} is installed (block "%s")', strjoin(opts.libs, ', '), key);
end
for q = 1:numel(names)
  pat = ['^' regexprep(regexptranslate('escape', names{q}), '\s+', '\\s+') '$'];
  for k = 1:numel(loaded)
    hits = find_system(loaded{k}, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
                       'RegExp', 'on', 'Type', 'block', 'Name', pat);
    if numel(hits) == 1
      path = hits{1};
      return;
    elseif numel(hits) > 1
      error('sc:ambiguous', 'block "%s": %d blocks named "%s" in %s:\n  %s\nSet opts.blocks.%s to the one to use.', ...
            key, numel(hits), names{q}, loaded{k}, strjoin(reshape(hits, 1, []), '\n  '), key);
    end
  end
end
error('sc:blockNotFound', ['block "%s": no block named {%s} in the libraries {%s}.\n' ...
      'Find it in the Library Browser and set opts.blocks.%s to its path.'], ...
      key, strjoin(names, ' | '), strjoin(loaded, ', '), key);
end
