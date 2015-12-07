# simple inputs files for the unsplit CTU hydro scheme

[driver]
max_steps = 100
tmax = 1.0000

[compressible-gr]
limiter = 1

[io]
basename = ../../Documents/Work/pyro/results/sod/sod_128_   ; basename for output files
n_out = 1                   ; number of timesteps between writing output files
do_io = 1                   ; do we output at all?

[vis]

dovis = 0                  ; runtime visualization? (1=yes, 0=no)
store_images = 0           ; store vis images to files (1=yes, 0=no)

[mesh]
nx = 10
ny = 128
xmax = .05
ymax = 1.0
ylboundary = outflow
yrboundary = outflow

[sod]
direction = y

dens_left = 1.0
dens_right = 0.125

u_left = 0.0
u_right = 0.0

p_left = 1.0
p_right = 0.1
