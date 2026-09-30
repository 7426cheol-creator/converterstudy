function [tw, yw] = sc_window(t, y, ta, tb)
%SC_WINDOW Raw logged samples inside [ta, tb] with the ends interpolated.
%   Duplicated time points are kept: a switching edge is logged twice at the
%   same time (before and after the jump), so trapz over the result
%   integrates a signal with jumps as the solver resolved them. Resampling a
%   jump onto a uniform grid would smear it over one solver step instead.

t = t(:);
y = y(:);
k = find(t > ta & t < tb);
tw = [ta; t(k); tb];
yw = [sc_grid(t, y, ta); y(k); sc_grid(t, y, tb)];
end
