PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 3

COMP	   = gnu

USE_MPI    = FALSE
USE_OMP    = FALSE

EBASE = main

# define the location of the MICROPHYSICS top directory
#MICROPHYSICS_HOME  := ../..

# This sets the EOS directory in Castro/EOS -- note: gamma_law will not work,
# you'll need to use gamma_law_general
EOS_DIR     := gamma_law_general

# This sets the network directory in Castro/Networks
NETWORK_DIR := powerlaw
INTEGRATOR_DIR := VODE

CONDUCTIVITY_DIR := stellar

EXTERN_SEARCH += .

Blocs   := .

F90EXE_sources += $(MICROPHYSICS_HOME)/util/python/AMReX_error_mod.F90
F90EXE_sources += $(MICROPHYSICS_HOME)/util/python/AMReX_ParallelDescriptor_F.F90

include $(MICROPHYSICS_HOME)/util/python/Make.here
