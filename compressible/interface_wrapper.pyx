import numpy as np
cimport numpy as np

cdef extern from "interface_h.h":
    void states_c(int idir, int qx, int qy, int ng, double dx,
               double dt, int irho, int iu, int iv, int ip, int ix, int nvar, int nspec, double gamma,
               double* qv, double* dqv,
               double* q_l, double* q_r)

    void riemann_cgf_c(int idir, int qx, int qy, int ng,
                  int nvar, int idens, int ixmom, int iymom,
                  int iener, int irhoX, int nspec,
                  int lower_solid, int upper_solid,
                  double gamma, double* U_l, double* U_r, double* F)

    void riemann_prim_c(int idir, int qx, int qy, int ng,
                  int nvar, int irho, int iu, int iv, int ip,
                  int iX, int nspec,
                  int lower_solid, int upper_solid,
                  double gamma,
                  double* q_l,
                  double* q_r, double* q_int)

    void riemann_hllc_c(int idir, int qx, int qy, int ng,
                    int nvar, int idens, int ixmom, int iymom,
                    int iener, int irhoX, int nspec,
                    int lower_solid, int upper_solid,
                    double gamma, double * U_l,
                    double * U_r, double * F)

    void artificial_viscosity_c(int qx, int qy, int ng,
                           double dx, double dy,
                           double cvisc,
                           double* u, double* v,
                           double* avisco_x, double* avisco_y)

def states(int idir, int qx, int qy, int ng, double dx,
           double dt, int irho, int iu, int iv, int ip, int ix, int nvar, int nspec, double gamma,
           np.ndarray[np.float64_t, ndim=3] qv not None,
           np.ndarray[np.float64_t, ndim=3] dqv not None):

    cdef np.ndarray q_l = np.zeros([qx, qy, nvar], dtype=np.float64)
    cdef np.ndarray q_r = np.zeros([qx, qy, nvar], dtype=np.float64)

    states_c(idir, qx, qy, ng, dx,
               dt, irho, iu, iv, ip, ix, nvar, nspec, gamma,
               <double*> qv.data, <double*> dqv.data,
               <double*> q_l.data, <double*> q_r.data)

    return q_l, q_r

def riemann_cgf(int idir, int qx, int qy, int ng,
               int nvar, int idens, int ixmom, int iymom,
               int iener, int irhoX, int nspec,
               int lower_solid, int upper_solid,
               double gamma,
               np.ndarray[np.float64_t, ndim=3] U_l not None,
               np.ndarray[np.float64_t, ndim=3] U_r not None):

    cdef np.ndarray F = np.zeros([qx, qy, nvar], dtype=np.float64)

    riemann_cgf_c(idir, qx, qy, ng,
                   nvar, idens, ixmom, iymom,
                   iener, irhoX, nspec,
                   lower_solid, upper_solid,
                   gamma, <double*> U_l.data,
                   <double*> U_r.data, <double*> F.data)

    return F

def riemann_prim(int idir, int qx, int qy, int ng,
               int nvar, int irho, int iu, int iv, int ip,
               int iX, int nspec,
               int lower_solid, int upper_solid,
               double gamma,
               np.ndarray[np.float64_t, ndim=3] q_l not None,
               np.ndarray[np.float64_t, ndim=3] q_r not None):

    cdef np.ndarray q_int = np.zeros([qx, qy, nvar], dtype=np.float64)

    riemann_prim_c(idir, qx, qy, ng,
                   nvar, irho, iu, iv, ip,
                   iX, nspec,
                   lower_solid, upper_solid,
                   gamma, <double*> q_l.data,
                   <double*> q_r.data, <double*> q_int.data)

    return q_int

def riemann_hllc(int idir, int qx, int qy, int ng,
               int nvar, int idens, int ixmom, int iymom,
               int iener, int irhoX, int nspec,
               int lower_solid, int upper_solid,
               double gamma,
               np.ndarray[np.float64_t, ndim=3] U_l not None,
               np.ndarray[np.float64_t, ndim=3] U_r not None):

    cdef np.ndarray F = np.zeros([qx, qy, nvar], dtype=np.float64)

    riemann_hllc_c(idir, qx, qy, ng,
                   nvar, idens, ixmom, iymom,
                   iener, irhoX, nspec,
                   lower_solid, upper_solid,
                   gamma, <double*> U_l.data,
                   <double*> U_r.data, <double*> F.data)

    return F

def artificial_viscosity(int qx, int qy, int ng,
                       double dx, double dy,
                       double cvisc,
                       np.ndarray[np.float64_t, ndim=2] u not None,
                       np.ndarray[np.float64_t, ndim=2] v not None):


    cdef np.ndarray avisco_x = np.zeros([qx, qy], dtype=np.float64)
    cdef np.ndarray avisco_y = np.zeros([qx, qy], dtype=np.float64)

    artificial_viscosity_c(qx, qy, ng,
                           dx, dy,
                           cvisc,
                           <double*> u.data, <double*> v.data,
                           <double*> avisco_x.data,
                           <double*> avisco_y.data)

    return avisco_x, avisco_y
