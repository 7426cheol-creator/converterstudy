function [te, ye] = sc_extrema(t, y)
%SC_EXTREMA Local extrema of a sampled signal, refined by a parabola through
%   the three samples around each one.

[t, k] = unique(t, 'last');
y = y(k);
te = [];
ye = [];
for k = 2:numel(y) - 1
  a = y(k) - y(k - 1);
  b = y(k + 1) - y(k);
  if a * b < 0
    tt = t(k - 1:k + 1);
    c = polyfit(tt - tt(2), y(k - 1:k + 1), 2);
    if c(1) ~= 0
      tv = -c(2) / (2 * c(1));
      te(end + 1) = tt(2) + tv; %#ok<AGROW>
      ye(end + 1) = polyval(c, tv); %#ok<AGROW>
    end
  end
end
end
