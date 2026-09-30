function path = sc_comp(lib, name)
%SC_COMP Block of the generated library whose ComponentPath is xcsc.NAME.

blks = find_system(lib, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'Type', 'block');
want = ['xcsc.' name];
for k = 1:numel(blks)
  try
    cp = get_param(blks{k}, 'ComponentPath');
  catch
    continue;
  end
  if strcmp(cp, want)
    path = blks{k};
    return;
  end
end
error('sc:componentNotFound', 'component %s not found in %s (check the ssc_build output)', want, lib);
end
