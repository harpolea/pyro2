import numpy as np
cimport numpy as np

cdef extern from "interface_states_h.h":
    void states_c(double *a, int qx, int qy, int ng, int idir,
                  double *al, double *ar)

    void states_nolimit_c(double *a, int qx, int qy,
                          int ng, int idir,
                          double *al, double *ar)

def states(np.ndarray[np.float64_t, ndim=2] a not None,
           int qx, int qy, int ng, int idir):

    cdef np.ndarray al = np.zeros([qx, qy],
        dtype=np.float64)
    cdef np.ndarray ar = np.zeros([qx, qy],
        dtype=np.float64)

    states_c(<double*> a.data, qx, qy, ng, idir,
             <double*> al.data, <double*> ar.data)

    return al, ar

def states_nolimit(np.ndarray[np.float64_t, ndim=2] a not None,
           int qx, int qy, int ng, int idir):

    cdef np.ndarray al = np.zeros([qx, qy],
        dtype=np.float64)
    cdef np.ndarray ar = np.zeros([qx, qy],
        dtype=np.float64)

    states_nolimit_c(<double*> a.data, qx, qy, ng, idir,
             <double*> al.data, <double*> ar.data)

    return al, ar
