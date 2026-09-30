function sc_wire(mdl, varargin)
%SC_WIRE Join physical ports (handles) into one electrical node.

h = [varargin{:}];
for k = 2:numel(h)
  add_line(mdl, h(1), h(k), 'autorouting', 'on');
end
end
