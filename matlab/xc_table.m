function rows = xc_table(d, key)
%XC_TABLE Rows (cell of cells of strings) of the exported table KEY.

for k = 1:numel(d.tables)
  t = d.tables{k};
  if strcmp(t.key, key)
    rows = t.rows;
    if ~iscell(rows)
      error('xc:table', 'table %s in %s has an unexpected row format', key, d.file);
    end
    rows = reshape(rows, 1, []);
    return;
  end
end
error('xc:missingTable', 'table "%s" not found in %s', key, d.file);
end
