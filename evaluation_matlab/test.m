fid = fopen(file, 'r');
tline = fgets(fid); 
[outp] = textscan(fid, '%d %s\n');
