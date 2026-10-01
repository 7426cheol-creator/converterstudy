function j = xc_col(d, key, name)
%XC_COL Index of the column titled NAME in the exported table KEY.
%   Columns are matched by their header text, never by position, so adding a column to an app table does
%   not silently shift what a check reads.

for k = 1:numel(d.tables)
  t = d.tables{k};
  if strcmp(t.key, key)
    cols = reshape(t.columns, 1, []);
    for j = 1:numel(cols)
      if strcmp(cols{j}, name)
        return;
      end
    end
    error('xc:missingColumn', 'column "%s" not found in table %s of %s', name, key, d.file);
  end
end
error('xc:missingTable', 'table "%s" not found in %s', key, d.file);
end
