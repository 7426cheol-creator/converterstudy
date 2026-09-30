function yg = sc_grid(t, y, tg)
%SC_GRID Logged samples on the time grid tg. Simscape logs an event time
%   twice (before / after a switching edge); the later sample is kept.
%   Grid points a rounding error outside the log are clamped to its ends.

[tu, k] = unique(t, 'last');
tg = min(max(tg, tu(1)), tu(end));
yg = interp1(tu, y(k), tg, 'linear');
end
