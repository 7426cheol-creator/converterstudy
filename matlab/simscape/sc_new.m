function mdl = sc_new(name)
%SC_NEW Fresh, empty model NAME (an open model of that name is closed unsaved).

mdl = name;
if bdIsLoaded(mdl)
  close_system(mdl, 0);
end
new_system(mdl);
end
