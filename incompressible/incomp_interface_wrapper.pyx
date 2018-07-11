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

ctg = np.ascontiguousarray

# we need extra wrappers here to make sure that the arrays are passed in as C-contiguous
def mac_vels(int qx, int qy, int ng, double dx,
            double dy, double dt,
            u, v, ldelta_ux, ldelta_vx,
            ldelta_uy, ldelta_vy,
            gradp_x, gradp_y):

    return mac_vels_w(qx, qy, ng, dx, dy, dt,
                ctg(u), ctg(v),
                ctg(ldelta_ux), ctg(ldelta_vx),
                ctg(ldelta_uy), ctg(ldelta_vy),
                ctg(gradp_x), ctg(gradp_y))

def mac_vels_w(int qx, int qy, int ng, double dx,
            double dy, double dt,
            np.ndarray[double, ndim=2, mode="c"] u not None,
            np.ndarray[double, ndim=2, mode="c"] v not None,
            np.ndarray[double, ndim=2, mode="c"] ldelta_ux not None,
            np.ndarray[double, ndim=2, mode="c"] ldelta_vx not None,
            np.ndarray[double, ndim=2, mode="c"] ldelta_uy not None,
            np.ndarray[double, ndim=2, mode="c"] ldelta_vy not None,
            np.ndarray[double, ndim=2, mode="c"] gradp_x not None,
            np.ndarray[double, ndim=2, mode="c"] gradp_y not None):

    cdef np.ndarray u_MAC = np.zeros([qx, qy], dtype=np.float64)
    cdef np.ndarray v_MAC = np.zeros([qx, qy], dtype=np.float64)

    mac_vels_c(qx, qy, ng, dx, dy, dt,
            <double*> u.data, <double*> v.data,
            <double*> ldelta_ux.data, <double*> ldelta_vx.data,
            <double*> ldelta_uy.data, <double*> ldelta_vy.data,
            <double*> gradp_x.data, <double*> gradp_y.data,
            <double*> u_MAC.data, <double*> v_MAC.data)

    return u_MAC, v_MAC

def states(int qx, int qy, int ng, double dx,
            double dy, double dt,
            u, v, ldelta_ux, ldelta_vx,
            ldelta_uy, ldelta_vy,
            gradp_x, gradp_y,
            u_MAC, v_MAC):

    return states_w(qx, qy, ng, dx, dy, dt,
                ctg(u), ctg(v),
                ctg(ldelta_ux), ctg(ldelta_vx),
                ctg(ldelta_uy), ctg(ldelta_vy),
                ctg(gradp_x), ctg(gradp_y),
                ctg(u_MAC),
                ctg(v_MAC))

def states_w(int qx, int qy, int ng, double dx,
            double dy, double dt,
            np.ndarray[double, ndim=2, mode="c"] u not None,
            np.ndarray[double, ndim=2, mode="c"] v not None,
            np.ndarray[double, ndim=2, mode="c"] ldelta_ux not None,
            np.ndarray[double, ndim=2, mode="c"] ldelta_vx not None,
            np.ndarray[double, ndim=2, mode="c"] ldelta_uy not None,
            np.ndarray[double, ndim=2, mode="c"] ldelta_vy not None,
            np.ndarray[double, ndim=2, mode="c"] gradp_x not None,
            np.ndarray[double, ndim=2, mode="c"] gradp_y not None,
            np.ndarray[double, ndim=2, mode="c"] u_MAC not None,
            np.ndarray[double, ndim=2, mode="c"] v_MAC not None):

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
