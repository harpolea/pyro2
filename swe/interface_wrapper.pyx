import numpy as np
cimport numpy as np

cdef extern from "interface_h.h":
    void states_c(int idir, int qx, int qy, int ng,
                double dx, double dt,
                int ih, int iu, int iv, int ix, int nvar, int nspec,
                double g,
                double *qv, double *dqv,
                double *q_l, double *q_r)

    void riemann_Roe_c(int idir, int qx, int qy, int ng,
                 int nvar, int ih, int ixmom, int iymom,
                 int ihX, int nspec,
                 int lower_solid, int upper_solid,
                 double g, double *U_l, double *U_r, double *F)

    void riemann_HLLC_c(int idir, int qx, int qy, int ng,
               int nvar, int ih, int ixmom, int iymom,
               int ihX, int nspec,
               int lower_solid, int upper_solid,
               double g, double *U_l, double *U_r,
               double *F)

def states(int idir, int qx, int qy, int ng, double dx,
           double dt, int ih, int iu, int iv, int ix, int nvar, int nspec, double g,
           np.ndarray[np.float64_t, ndim=3] qv not None,
           np.ndarray[np.float64_t, ndim=3] dqv not None):

    cdef np.ndarray q_l = np.zeros([qx, qy, nvar], dtype=np.float64)
    cdef np.ndarray q_r = np.zeros([qx, qy, nvar], dtype=np.float64)

    states_c(idir, qx, qy, ng, dx,
               dt, ih, iu, iv, ix, nvar, nspec, g,
               <double*> qv.data, <double*> dqv.data,
               <double*> q_l.data, <double*> q_r.data)

    return q_l, q_r

def riemann_roe(int idir, int qx, int qy, int ng,
               int nvar, int ih, int ixmom, int iymom,
               int ihX, int nspec,
               int lower_solid, int upper_solid,
               double gamma,
               np.ndarray[np.float64_t, ndim=3] U_l not None,
               np.ndarray[np.float64_t, ndim=3] U_r not None):

    cdef np.ndarray F = np.zeros([qx, qy, nvar], dtype=np.float64)

    riemann_Roe_c(idir, qx, qy, ng,
                nvar, ih, ixmom, iymom,
                ihX, nspec,
                lower_solid, upper_solid,
                gamma, <double*> U_l.data,
                <double*> U_r.data, <double*> F.data)

    return F

def riemann_hllc(int idir, int qx, int qy, int ng,
               int nvar, int ih, int ixmom, int iymom,
               int ihX, int nspec,
               int lower_solid, int upper_solid,
               double gamma,
               np.ndarray[np.float64_t, ndim=3] U_l not None,
               np.ndarray[np.float64_t, ndim=3] U_r not None):

    cdef np.ndarray F = np.zeros([qx, qy, nvar], dtype=np.float64)

    riemann_HLLC_c(idir, qx, qy, ng,
                   nvar, ih, ixmom, iymom,
                   ihX, nspec,
                   lower_solid, upper_solid,
                   gamma, <double*> U_l.data,
                   <double*> U_r.data, <double*> F.data)

    return F
