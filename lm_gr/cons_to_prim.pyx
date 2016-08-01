import lm_gr.LM_gr_interface_f as interface_f
from scipy.optimize import brentq
import numpy as np
cimport numpy as np

def cons_to_prim(np.ndarray[double, ndim=3] Q, float c, double gamma,
                 int qx, int qy, int nvar, int iD,
                 int iUx, int iUy, int iDh, int iDX, np.ndarray[double, ndim=2] alphasq):
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
    pmin[arr_root_find_on_me(pmin, D, Ux, Uy, Dh, alphasq, gamma) < 0.] = 0.
    pmax[pmax == 0.] = c

    # find nans
    pmin[np.isnan(pmin)] = 0.
    pmax[np.isnan(pmax)] = c

    #print('pmin: {}, pmax: {}'.format(pmin, pmax))
    # NOTE: would it be quicker to do this as loops in cython??
    #print('D {}'.format(D))
    #print('Dh {}'.format(Dh))
    try:
        V[:,:,iDh] = [[brentq(interface_f.root_finding, pmin[i,j], pmax[i,j],
        args=(D[i,j], Ux[i,j], Uy[i,j], Dh[i,j], alphasq[i,j], gamma)) for j in range(qy)] for i in range(qx)]
    except ValueError:
        print('VALUE ERROR')
        print('pmin: {}'.format(pmin))
        print('pmax: {}'.format(pmax))

        print('root find on pmin: {}'.format(arr_root_find_on_me(pmin, D, Ux, Uy, Dh, alphasq, gamma)))

        print('root find on pmax: {}'.format(arr_root_find_on_me(pmax, D, Ux, Uy, Dh, alphasq, gamma)))
        V[:,:,iDh] = Dh

    cdef np.ndarray[double, ndim=2] u0 = (Dh - D) * (gamma - 1.) / (gamma * V[:,:,iDh])

    #print('V[:,:,iDh]: {}'.format(V[:,:,iDh]))

    V[:,:,iD] = D / u0
    V[:,:,iDX] = DX / D
    V[:,:,iUx] = V[:,:,iUx] * u0
    V[:,:,iUy] = V[:,:,iUy] * u0

    return V

def arr_root_find_on_me(np.ndarray[double, ndim=2] pbar,
                        np.ndarray[double, ndim=2] D,
                        np.ndarray[double, ndim=2] Ux,
                        np.ndarray[double, ndim=2] Uy,
                        np.ndarray[double, ndim=2] Dh,
                        np.ndarray[double, ndim=2] alphasq,
                        double gamma):
    """
    Equation to root find on in order to find the primitive pressure.
    This works on arrays.
    """
    #cdef np.ndarray[double, ndim=2] eps, rho
    # eps * rho
    #rho = D * gamma * pbar / ((Dh - D) * (gamma - 1.))
    #eps = (Dh - D) / (D * gamma)

    #return (gamma - 1.) * eps * rho - pbar

    cdef np.ndarray[double, ndim=2] h, rho
    # eps * rho
    rho = D * np.sqrt(alphasq - Ux**2 - Uy**2)
    h = Dh / D

    return (gamma - 1.) * (h - 1.) * rho / gamma  - pbar
