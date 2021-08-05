set terminal png size 500,500
set output outfile
set xlabel "Time taken(in seconds)"
set ylabel "Utility"
plot infile using 2:3:1 w labels notitle point pt 7 lc rgb 'red'
