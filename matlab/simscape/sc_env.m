function env = sc_env()
%SC_ENV What this session can run. Textbook ch.18: check the release and the
%   installed products before building; without Simulink the cross-check
%   stays NOT_RUN_ENVIRONMENT, and writing .m files is not a run.

env = struct();
env.is_octave = exist('OCTAVE_VERSION', 'builtin') > 0;
if env.is_octave
  env.engine = 'GNU Octave';
  env.version = OCTAVE_VERSION();
  env.release = '';
else
  env.engine = 'MATLAB';
  env.version = version();
  env.release = version('-release');
end
env.platform = computer();
names = {};
try
  v = ver();
  names = {v.Name};
catch
end
env.products = names;
env.has_simulink = ~env.is_octave && any(strcmp(names, 'Simulink'));
env.has_simscape = ~env.is_octave && any(strcmp(names, 'Simscape'));
env.has_simscape_electrical = ~env.is_octave && any(strcmp(names, 'Simscape Electrical'));
env.ok = env.has_simulink && env.has_simscape;
if env.is_octave
  env.reason = 'GNU Octave session: Simulink and Simscape are MATLAB products and are not available here';
elseif ~env.has_simulink
  env.reason = 'Simulink is not installed in this MATLAB';
elseif ~env.has_simscape
  env.reason = 'Simscape is not installed in this MATLAB';
else
  env.reason = '';
end
end
