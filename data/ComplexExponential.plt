set terminal pngcairo  transparent enhanced font "arial,10" fontscale 1.0 size 600, 400 
set output 'ComplexExponential.png'
set datafile separator tab
plot 'ComplexExponential.dat' using 1:2 t "Training data", \
     'ComplexExponential.dat' using 1:3 t "Original data" smooth csplines dashtype solid, \
	 'ComplexExponential.dat' using 1:4 t "Classical VP" smooth csplines dashtype 2 linewidth 2, \
	 'ComplexExponential.dat' using 1:5 t "Regularized VP" smooth csplines dashtype 3 linewidth 2