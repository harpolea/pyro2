import compressible_gr.interface_f as interface_f
from scipy.optimize import brentq
import numpy as np
cimport numpy as np

def cons_to_prim(np.ndarray[double, ndim=3] Q, float c, double gamma,
                 int qx, int qy, int nvar, int iD,
                 int iSx, int iSy, int itau, int iDX):
    """
    Cython implementation of code to change the vector of conservative variables (D, Sx, Sy, tau, DX) into the vector of primitive variables (rho, u, v, p, X). Root finder brentq is applied to the fortran function root_finding from interface_f.

    Main looping done as a list comprehension as this is faster than nested for loops in pure python - not so sure this is the case for cython?
    """

    cdef np.ndarray[double, ndim=2] D = Q[:,:,iD]
    cdef np.ndarray[double, ndim=2] Sx = Q[:,:,iSx]
    cdef np.ndarray[double, ndim=2] Sy = Q[:,:,iSy]
    cdef np.ndarray[double, ndim=2] tau = Q[:,:,itau]
    cdef np.ndarray[double, ndim=2] DX = Q[:,:,iDX]

    cdef np.ndarray[double, ndim=3] V = np.zeros((qx, qy, nvar))

    cdef np.ndarray[double, ndim=2] pmin = (Sx**2 + Sy**2)/c**2 - tau - D
    cdef np.ndarray[double, ndim=2] pmax = (gamma - 1.) * tau

    pmax[pmax < 0.] = np.fabs(pmax[pmax < 0.])
    pmin[pmin > pmax] = abs(np.sqrt(Sx[pmin > pmax]**2 + Sy[pmin > pmax]**2)/c - tau[pmin > pmax] - D[pmin > pmax])

    pmin[pmin < 0.] = 0.
    pmin[arr_root_find_on_me(pmin, D, Sx, Sy, tau, c, gamma) < 0.] = 0.
    pmax[pmax == 0.] = c

    # check they have different signs - positive if not.
    #mask = arr_root_find_on_me(pmin, D, Sx, Sy, tau, c, gamma) * arr_root_find_on_me(pmax, D, Sx, Sy, tau, c, gamma) == 0.
    #print(pmin[mask])

    #pmin[mask] = 0.
    #pmax[mask] *= 2.

    # find nans
    pmin[np.isnan(pmin)] = 0.
    pmax[np.isnan(pmax)] = c

    # NOTE: would it be quicker to do this as loops in cython??
    try:
        V[:,:,itau] = [[brentq(interface_f.root_finding, pmin[i,j], pmax[i,j], args=(D[i,j], Sx[i,j], Sy[i,j], tau[i,j], c, gamma)) for j in range(qy)] for i in range(qx)]
    except ValueError:
        print('pmin: {}'.format(pmin))
        print('pmax: {}'.format(pmax))

    V[:,:,iSx] = Sx / (tau + D + V[:,:,itau])
    V[:,:,iSy] = Sy / (tau + D + V[:,:,itau])

    cdef np.ndarray[double, ndim=2] w = 1. / np.sqrt(1. - (V[:,:,iSx]**2 + V[:,:,iSy]**2) / c**2)

    V[:,:,iD] = D / w
    V[:,:,iDX] = DX / D

    return V

def arr_root_find_on_me(np.ndarray[double, ndim=2] pbar,
                        np.ndarray[double, ndim=2] D,
                        np.ndarray[double, ndim=2] Sx,
                        np.ndarray[double, ndim=2] Sy,
                        np.ndarray[double, ndim=2] tau,
                        double c, double gamma):
    """
    Equation to root find on in order to find the primitive pressure.
    This works on arrays.
    """
    cdef np.ndarray[double, ndim=2] v2, w, epsrho
    #if pbar[pbar > 0.]:
    v2 = (Sx**2 + Sy**2) / (c * (tau + D + pbar))**2
    w = 1. / np.sqrt(1. - v2)
    epsrho = (tau + D * (1. - w) + pbar * v2 / (v2 - 1.)) / w**2

    #neg_p = (pbar <= 0.)
    return (gamma - 1.) * epsrho - pbar
    #pbar[neg_p] = 1.e6

    #return pbar
