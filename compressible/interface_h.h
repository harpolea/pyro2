#include <stddef.h>

void states_c(int idir, int qx, int qy, int ng, double dx,
           double dt, int irho, int iu, int iv, int ip, int ix, int nvar, int nspec, double gamma,
           double* qv, double* dqv,
           double* q_l, double* q_r);

void riemann_cgf_c(int idir, int qx, int qy, int ng,
               int nvar, int idens, int ixmom, int iymom,
               int iener, int irhoX, int nspec,
               int lower_solid, int upper_solid,
               double gamma, double* U_l, double* U_r, double* F);

void riemann_prim_c(int idir, int qx, int qy, int ng,
                int nvar, int irho, int iu, int iv, int ip,
                int iX, int nspec,
                int lower_solid, int upper_solid,
                double gamma,
                double* q_l,
                double* q_r, double* q_int);

void riemann_hllc_c(int idir, int qx, int qy, int ng,
                int nvar, int idens, int ixmom, int iymom,
                int iener, int irhoX, int nspec,
                int lower_solid, int upper_solid,
                double gamma, double * U_l,
                double * U_r, double * F);

void consFlux(int idir, double gamma, int idens,
              int ixmom, int iymom,
              int iener, int irhoX, int nvar, int nspec,
              double* U_state, double* F);

void artificial_viscosity_c(int qx, int qy, int ng,
                       double dx, double dy,
                       double cvisc,
                       double* u, double* v,
                       double* avisco_x, double* avisco_y);
