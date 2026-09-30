function blk = sc_place(mdl, src, name, col, row)
%SC_PLACE Add a copy of library block SRC to model MDL as NAME on a grid.

x = 60 + 130 * col;
y = 60 + 110 * row;
blk = [mdl '/' name];
add_block(src, blk, 'Position', [x, y, x + 60, y + 40]);
end
