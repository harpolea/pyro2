import lm_gr.LM_gr_interface_f as interface_f
from scipy.optimize import brentq
import numpy as np
cimport numpy as np

def cons_to_prim(np.ndarray[double, ndim=3] Q, float c, double gamma,
                 int qx, int qy, int nvar, int iD,
                 int iUx, int iUy, int iDh, int iDX, np.ndarray[double, ndim=2] alphasq, np.ndarray[double, ndim=4] gamma_mat):
    """
    Cython implementation of code to change the vector of conservative variables
    (D, Sx, Sy, tau, DX) into the vector of primitive variables (rho, u, v, p, X).
    Root finder brentq is applied to the fortran function root_finding from interface_f.

    Main looping done as a list comprehension as this is faster than nested for
    loops in pure python - not so sure this is the case for cython?
    """

    cdef np.ndarray[double, ndim=2] D = Q[:,:,iD]
    cdef np.ndarray[double, ndim=2] Ux = Q[:,:,iUx]
    cdef np.ndarray[double, ndim=2] Uy = Q[:,:,iUy]
    cdef np.ndarray[double, ndim=2] Dh = Q[:,:,iDh]
    cdef np.ndarray[double, ndim=2] DX = Q[:,:,iDX]

    cdef np.ndarray[double, ndim=3] V = np.zeros((qx, qy, nvar))

    cdef np.ndarray[double, ndim=2] pmin = (Dh - D) * (gamma - 1.) / gamma
    cdef np.ndarray[double, ndim=2] pmax = gamma * Dh * c

    pmax[pmax < 0.] = np.fabs(pmax[pmax < 0.])
    pmin[pmin > pmax] = 0.

    pmin[pmin < 0.] = 0.
    pmin[arr_root_find_on_me(pmin, D, Ux, Uy, Dh, alphasq, gamma_mat, gamma, c) < 0.] = 0.
    pmax[pmax == 0.] = c

    # find nans
    pmin[np.isnan(pmin)] = 0.
    pmax[np.isnan(pmax)] = c

    try:
        V[:,:,iDh] = [[brentq(interface_f.root_finding, pmin[i,j], pmax[i,j], args=(D[i,j], Ux[i,j], Uy[i,j], Dh[i,j], alphasq[i,j], gamma_mat[i,j,:,:], gamma, c)) for j in range(qy)] for i in range(qx)]
    except ValueError:
        print('VALUE ERROR')
        print('pmin: {}'.format(pmin))
        print('pmax: {}'.format(pmax))

    cdef np.ndarray[double, ndim=2] u0 = (Dh - D) * (gamma - 1.) / (gamma * V[:,:,iDh])

    V[:,:,iD] = D / u0
    V[:,:,iDX] = DX / D

    #print('u0: {}'.format(u0))
    V[:,:,iUx] = V[:,:,iUx] * u0
    V[:,:,iUy] = V[:,:,iUy] * u0

    return V

def arr_root_find_on_me(np.ndarray[double, ndim=2] pbar,
                        np.ndarray[double, ndim=2] D,
                        np.ndarray[double, ndim=2] Ux,
                        np.ndarray[double, ndim=2] Uy,
                        np.ndarray[double, ndim=2] Dh,
                        np.ndarray[double, ndim=2] alphasq,
                        np.ndarray[double, ndim=4] gamma_mat,
                        double gamma, double c):
    """
    Equation to root find on in order to find the primitive pressure.
    This works on arrays.
    """

    cdef np.ndarray[double, ndim=2] h, rho
    # eps * rho
    rho = D * np.sqrt(alphasq - (gamma_mat[:,:,0,0] * Ux**2 + gamma_mat[:,:,1,1] * Uy**2 + 2. * gamma_mat[:,:,0,1] * Ux * Uy)/c**2) / c
    h = Dh / D

    return (gamma - 1.) * (h - 1.) * rho / gamma  - pbar
