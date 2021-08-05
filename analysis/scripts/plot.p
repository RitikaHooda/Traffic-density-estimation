set terminal png size 500,500
set output outfile
set xlabel "Time (in seconds)"
set ylabel "Queue Density"
plot infile using 1:2 smooth unique title mname lt 7 lc 6 w lines

