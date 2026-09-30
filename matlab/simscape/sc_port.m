function h = sc_port(blk, which)
%SC_PORT Handle of a physical connection port.
%   'p'   the single left port of a two-terminal block (+ of the .ssc files)
%   'n'   the single right port of a two-terminal block (- of the .ssc files)
%   'one' the only port of a one-port block (Electrical Reference,
%         Solver Configuration)
%   The port counts are checked, so a block with another layout stops here.
%   For library R, L, C the orientation only changes the sign of that
%   element's own variables, which are not read.

ph = get_param(blk, 'PortHandles');
L = ph.LConn;
R = ph.RConn;
switch which
  case 'one'
    all = [L, R];
    if numel(all) ~= 1
      error('sc:ports', '%s: expected one connection port, found %d', blk, numel(all));
    end
    h = all(1);
  case {'p', 'n'}
    if numel(L) ~= 1 || numel(R) ~= 1
      error('sc:ports', '%s: expected one left and one right port, found %d and %d', blk, numel(L), numel(R));
    end
    if strcmp(which, 'p')
      h = L(1);
    else
      h = R(1);
    end
  otherwise
    error('sc:ports', 'unknown port selector %s', which);
end
end
