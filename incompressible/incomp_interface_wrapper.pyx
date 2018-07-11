import numpy as np
cimport numpy as np

cdef extern from "incomp_interface_h.h":
    void mac_vels_c(int qx, int qy, int ng, double dx,
                double dy, double dt,
                double *u, double *v,
                double *ldelta_ux, double *ldelta_vx,
                double *ldelta_uy, double *ldelta_vy,
                double *gradp_x, double *gradp_y,
                double *u_MAC, double *v_MAC);

    void states_c(int qx, int qy, int ng, double dx,
                double dy, double dt,
                double *u, double *v,
                double *ldelta_ux, double *ldelta_vx,
                double *ldelta_uy, double *ldelta_vy,
                double *gradp_x, double *gradp_y,
                double *u_MAC, double *v_MAC,
                double *u_xint, double *v_xint,
                double *u_yint, double *v_yint);

def mac_vels(int qx, int qy, int ng, np.float64_t dx,
            np.float64_t dy, np.float64_t dt,
            np.ndarray[np.float64_t, ndim=2] u not None,
            np.ndarray[np.float64_t, ndim=2] v not None,
            np.ndarray[np.float64_t, ndim=2] ldelta_ux not None,
            np.ndarray[np.float64_t, ndim=2] ldelta_vx not None,
            np.ndarray[np.float64_t, ndim=2] ldelta_uy not None,
            np.ndarray[np.float64_t, ndim=2] ldelta_vy not None,
            np.ndarray[np.float64_t, ndim=2] gradp_x not None,
            np.ndarray[np.float64_t, ndim=2] gradp_y not None):

    cdef np.ndarray u_MAC = np.zeros([qx, qy], dtype=np.float64)
    cdef np.ndarray v_MAC = np.zeros([qx, qy], dtype=np.float64)

    mac_vels_c(qx, qy, ng, dx, dy, dt,
            <double*> u.data, <double*> v.data,
            <double*> ldelta_ux.data, <double*> ldelta_vx.data,
            <double*> ldelta_uy.data, <double*> ldelta_vy.data,
            <double*> gradp_x.data, <double*> gradp_y.data,
            <double*> u_MAC.data, <double*> v_MAC.data)

    return u_MAC, v_MAC

def states(int qx, int qy, int ng, np.float64_t dx,
            np.float64_t dy, np.float64_t dt,
            np.ndarray[np.float64_t, ndim=2] u not None,
            np.ndarray[np.float64_t, ndim=2] v not None,
            np.ndarray[np.float64_t, ndim=2] ldelta_ux not None,
            np.ndarray[np.float64_t, ndim=2] ldelta_vx not None,
            np.ndarray[np.float64_t, ndim=2] ldelta_uy not None,
            np.ndarray[np.float64_t, ndim=2] ldelta_vy not None,
            np.ndarray[np.float64_t, ndim=2] gradp_x not None,
            np.ndarray[np.float64_t, ndim=2] gradp_y not None,
            np.ndarray[np.float64_t, ndim=2] u_MAC not None,
            np.ndarray[np.float64_t, ndim=2] v_MAC not None):

    cdef np.ndarray u_xint = np.zeros([qx, qy], dtype=np.float64)
    cdef np.ndarray v_xint = np.zeros([qx, qy], dtype=np.float64)
    cdef np.ndarray u_yint = np.zeros([qx, qy], dtype=np.float64)
    cdef np.ndarray v_yint = np.zeros([qx, qy], dtype=np.float64)

    states_c(qx, qy, ng, dx, dy, dt,
            <double*> u.data, <double*> v.data,
            <double*> ldelta_ux.data, <double*> ldelta_vx.data,
            <double*> ldelta_uy.data, <double*> ldelta_vy.data,
            <double*> gradp_x.data, <double*> gradp_y.data,
            <double*> u_MAC.data, <double*> v_MAC.data,
            <double*> u_xint.data, <double*> v_xint.data,
            <double*> u_yint.data, <double*> v_yint.data)

    return u_xint, v_xint, u_yint, v_yint
