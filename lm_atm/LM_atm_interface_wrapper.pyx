import numpy as np
cimport numpy as np

cdef extern from "LM_atm_interface_h.h":
    void mac_vels_c(int qx, int qy, int ng, double dx,
                  double dy, double dt,
                  double *u, double *v,
                  double *ldelta_ux, double *ldelta_vx,
                  double *ldelta_uy, double *ldelta_vy,
                  double *gradp_x, double *gradp_y,
                  double *source,
                  double *u_MAC, double *v_MAC)

    void states_c(int qx, int qy, int ng, double dx,
              double dy, double dt,
              double *u, double *v,
              double *ldelta_ux, double *ldelta_vx,
              double *ldelta_uy, double *ldelta_vy,
              double *gradp_x, double *gradp_y,
              double *source,
              double *u_MAC, double *v_MAC,
              double *u_xint, double *v_xint,
              double *u_yint, double *v_yint)

    void rho_states_c(int qx, int qy, int ng, double dx,
                  double dy, double dt,
                  double *rho, double *u_MAC, double *v_MAC,
                  double *ldelta_rx, double *ldelta_ry,
                  double *rho_xint, double *rho_yint)

ctg = np.ascontiguousarray

def mac_vels(int qx, int qy, int ng, double dx,
          double dy, double dt,
          u, v, ldelta_ux, ldelta_vx,
          ldelta_uy, ldelta_vy,
          gradp_x, gradp_y, source):

    return mac_vels_w(qx, qy, ng, dx, dy, dt,
              ctg(u), ctg(v),
              ctg(ldelta_ux), ctg(ldelta_vx),
              ctg(ldelta_uy), ctg(ldelta_vy),
              ctg(gradp_x), ctg(gradp_y), ctg(source))

def mac_vels_w(int qx, int qy, int ng, double dx,
              double dy, double dt,
              np.ndarray[np.float64_t, ndim=2] u not None,
              np.ndarray[np.float64_t, ndim=2] v not None,
              np.ndarray[np.float64_t, ndim=2] ldelta_ux not None,
              np.ndarray[np.float64_t, ndim=2] ldelta_vx not None,
              np.ndarray[np.float64_t, ndim=2] ldelta_uy not None,
              np.ndarray[np.float64_t, ndim=2] ldelta_vy not None,
              np.ndarray[np.float64_t, ndim=2] gradp_x not None,
              np.ndarray[np.float64_t, ndim=2] gradp_y not None,
              np.ndarray[np.float64_t, ndim=2] source not None):

    cdef np.ndarray u_MAC = np.zeros([qx, qy], dtype=np.float64)
    cdef np.ndarray v_MAC = np.zeros([qx, qy], dtype=np.float64)

    mac_vels_c(qx, qy, ng, dx, dy, dt,
                  <double*> u.data, <double*> v.data,
                  <double*> ldelta_ux.data, <double*> ldelta_vx.data,
                  <double*> ldelta_uy.data, <double*> ldelta_vy.data,
                  <double*> gradp_x.data, <double*> gradp_y.data,
                  <double*> source.data,
                  <double*> u_MAC.data, <double*> v_MAC.data)

    return u_MAC, v_MAC

def states(int qx, int qy, int ng, double dx,
            double dy, double dt,
            u, v, ldelta_ux, ldelta_vx,
            ldelta_uy, ldelta_vy,
            gradp_x, gradp_y, source,
            u_MAC, v_MAC):

    return states_w(qx, qy, ng, dx, dy, dt,
                ctg(u), ctg(v),
                ctg(ldelta_ux), ctg(ldelta_vx),
                ctg(ldelta_uy), ctg(ldelta_vy),
                ctg(gradp_x), ctg(gradp_y), ctg(source),
                ctg(u_MAC), ctg(v_MAC))

def states_w(int qx, int qy, int ng, double dx,
          double dy, double dt,
          np.ndarray[np.float64_t, ndim=2] u not None,
          np.ndarray[np.float64_t, ndim=2] v not None,
          np.ndarray[np.float64_t, ndim=2] ldelta_ux not None,
          np.ndarray[np.float64_t, ndim=2] ldelta_vx not None,
          np.ndarray[np.float64_t, ndim=2] ldelta_uy not None,
          np.ndarray[np.float64_t, ndim=2] ldelta_vy not None,
          np.ndarray[np.float64_t, ndim=2] gradp_x not None,
          np.ndarray[np.float64_t, ndim=2] gradp_y not None,
          np.ndarray[np.float64_t, ndim=2] source not None,
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
            <double*> source.data,
            <double*> u_MAC.data, <double*> v_MAC.data,
            <double*> u_xint.data, <double*> v_xint.data,
            <double*> u_yint.data, <double*> v_yint.data)

    return u_xint, v_xint, u_yint, v_yint

def rho_states(int qx, int qy, int ng, double dx,
            double dy, double dt,
            rho, u_MAC, v_MAC, ldelta_rx,
            ldelta_ry):

    return rho_states_w(qx, qy, ng, dx, dy, dt,
                ctg(rho),
                ctg(u_MAC), ctg(v_MAC),
                ctg(ldelta_rx), ctg(ldelta_ry))

def rho_states_w(int qx, int qy, int ng, double dx,
                double dy, double dt,
                np.ndarray[np.float64_t, ndim=2] rho not None,
                np.ndarray[np.float64_t, ndim=2] u_MAC not None,
                np.ndarray[np.float64_t, ndim=2] v_MAC not None,
                np.ndarray[np.float64_t, ndim=2] ldelta_rx not None,
                np.ndarray[np.float64_t, ndim=2] ldelta_ry not None):

    cdef np.ndarray rho_xint = np.zeros([qx, qy], dtype=np.float64)
    cdef np.ndarray rho_yint = np.zeros([qx, qy], dtype=np.float64)

    rho_states_c(qx, qy, ng, dx, dy, dt,
         <double*> rho.data, <double*> u_MAC.data, <double*> v_MAC.data,
          <double*> ldelta_rx.data, <double*> ldelta_ry.data,
          <double*> rho_xint.data, <double*> rho_yint.data)

    return rho_xint, rho_yint
