function name = sc_solver(mdl, t_stop, max_step)
%SC_SOLVER Variable-step stiff solver with a bounded step and Simscape logging.
%   daessc when this release has it, ode23t otherwise.

name = 'daessc';
try
  set_param(mdl, 'Solver', name);
catch
  name = 'ode23t';
  set_param(mdl, 'Solver', name);
end
set_param(mdl, 'StopTime', sprintf('%.17g', t_stop), ...
          'MaxStep', sprintf('%.17g', max_step), ...
          'RelTol', '1e-7', ...
          'SimscapeLogType', 'all', ...
          'SimscapeLogName', 'simlog');
end
