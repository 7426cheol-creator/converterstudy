function lib = sc_components(opts)
%SC_COMPONENTS Build the +xcsc Simscape components (ports declared in the
%   .ssc files: + / p on the left, - / n on the right) into a block library in
%   opts.out_dir with ssc_build, and load it. Returns the library name.

src = fullfile(opts.here, '+xcsc');
dst = fullfile(opts.out_dir, '+xcsc');
if exist(dst, 'dir') ~= 7
  mkdir(dst);
end
copyfile(fullfile(src, '*.ssc'), dst);
old = cd(opts.out_dir);
back = onCleanup(@() cd(old));
lib = 'xcsc_lib';
if bdIsLoaded(lib)
  close_system(lib, 0);
end
ssc_build('xcsc');
addpath(opts.out_dir);
load_system(lib);
end
