function b = xc_bridge(theta, w, phi)
%XC_BRIDGE Normalized bridge output of textbook E06:
%   s(x) = +1 for 0 <= mod(x, 2*pi) < pi, -1 otherwise
%   b(theta; w, phi) = ( s(theta - phi + w/2) - s(theta - phi - w/2) ) / 2
%   w = pi gives the bipolar square wave, w < pi adds a zero interval.

b = 0.5 * (local_s(theta - phi + w / 2) - local_s(theta - phi - w / 2));
end

function y = local_s(x)
y = 2 * (mod(x, 2 * pi) < pi) - 1;
end
