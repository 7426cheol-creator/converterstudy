function [t, y] = sc_series(simlog, blk, var, unit)
%SC_SERIES Time series of variable VAR of block BLK from the Simscape log.
%   Only variables declared in the +xcsc components are read (i, v), so no
%   library variable names are assumed.

node = simlog.(blk).(var);
t = node.series.time;
y = node.series.values(unit);
t = t(:);
y = y(:);
end
