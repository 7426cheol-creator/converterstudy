function opts = sc_opts(opts)
%SC_OPTS Default options shared by the builders.
%   export_dir  Python exports read for the expected values
%               (default ../results/python_export, made by
%               verification/run_octave_crosscheck.sh)
%   out_dir     generated files: component library, .slx models (default ./out,
%               ignored by git)
%   run         simulate and compare after building (default true)
%   save        save the model as .slx in out_dir (default true)
%   blocks      struct of library block paths that replace the run-time
%               lookup, e.g. opts.blocks.inductor = '<path from the Library Browser>'
%   libs        library models searched, in order of preference

here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(opts)
  opts = struct();
end
def = struct( ...
  'export_dir', fullfile(fileparts(here), 'results', 'python_export'), ...
  'out_dir', fullfile(here, 'out'), ...
  'run', true, ...
  'save', true, ...
  'blocks', struct(), ...
  'libs', {{'fl_lib', 'nesl_utility', 'ee_lib'}});
fn = fieldnames(def);
for k = 1:numel(fn)
  if ~isfield(opts, fn{k})
    opts.(fn{k}) = def.(fn{k});
  end
end
opts.here = here;
addpath(fileparts(here));          % ../xc_load, ../xc_row, ../xc_json
end
