# simple inputs files for the unsplit CTU hydro scheme

[driver]
max_steps = 500
tmax = 0.2

max_dt_change = 1.e33
init_tstep_factor = 1.0


[driver]
cfl = 0.8

[io]
basename = sine_
dt_out = 0.2

[mesh]
nx = 64
ny = 64
xmax = 1.0
ymax = 1.0

xlboundary = periodic
xrboundary = periodic

ylboundary = outflow
yrboundary = outflow


[burgers]
limiter = 2

[sine]
direction = y
