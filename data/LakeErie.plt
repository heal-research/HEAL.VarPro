set terminal pngcairo  transparent enhanced font "arial,10" fontscale 1.0 size 600, 400 
set output 'LakeErie.png'
set datafile separator tab
plot 'LakeErie.dat' using 1:2 t "Original data", \
	 'LakeErie.dat' using 1:3 t "Classical VP" smooth csplines dashtype solid linewidth 2, \
	 'LakeErie.dat' using 1:4 t "Regularized VP" smooth csplines dashtype solid linewidth 2