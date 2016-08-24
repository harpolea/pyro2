import compressible_gr.interface_f as interface_f
from scipy.optimize import brentq
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

def cons_to_prim(Q, c, gamma, qx, qy, nvar, iD, iSx, iSy, itau, iDX):
    """
    PyCUDA implementation of code to change the vector of conservative variables (D, Sx, Sy, tau, DX) into the vector of primitive variables (rho, u, v, p, X). Root finder brentq is applied to the fortran function root_finding from interface_f.

    Main looping done as a list comprehension as this is faster than nested for loops in pure python - not so sure this is the case for cython?
    """

    mod = SourceModule("""
        #include <cmath.h>
        __device__ float root_finding(float pbar, float D, float Sx, float Sy,
                                    float tau, float c, float gamma)
        {
            double largep = 1.e6;

            if (pbar < 0.0) {
                float v2 = (Sx^2 + Sy^2) / (c * (tau + D + pbar))^2;
                float w = 1.0 / sqrt(1.0 - v2);
                float epsrho = (tau + D * (1.0 - w)) * (1.0 - v2) - pbar * v2;

                pbar = (gamma - 1.0) * epsrho - pbar;
            } else {
                pbar = largep;
            }

            return pbar;
        }

        __global__ void find_p_c(float** pbars, float** pmins, float** pmaxes,
                            float** Ds, float** Sxs, float** Sys,
                            float** taus, float c, float gamma)
        {
            const int ITMAX=100;
            cont float TOL=1.e-8;

            int n_x = blockDim.x*gridDim.x;
            int idx = threadIdx.x + blockDim.x*blockIdx.x;
            int idy = threadIdx.y + blockDim.y*blockIdx.y;
            int threadId = idy*n_x+idx;

            float pmin = pmins[idx][idy];
            float pmax = pmaxes[idx][idy];

            int counter = 0;

            float min_guess = root_finding(pmin, Ds[idx][idy],
                    Sxs[idx][idy], Sys[idx][idy], taus[idx][idy],
                    c, gamma);
            float max_guess = root_finding(pmax, Ds[idx][idy],
                                Sxs[idx][idy], Sys[idx][idy], taus[idx][idy],
                                c, gamma);
            float difference = max_guess - min_guess;

            while (abs(difference > TOL)  && (counter < ITMAX) {
                float pmid = 0.5 * (pmin + pmax);

                float mid_guess = root_finding(pmid, Ds[idx][idy],
                                    Sxs[idx][idy], Sys[idx][idy], taus[idx][idy],
                                    c, gamma);
                if (mid_guess * min_guess < 0) {
                    pmax = pmid;
                    max_guess = root_finding(pmax, Ds[idx][idy],
                                        Sxs[idx][idy], Sys[idx][idy], taus[idx][idy],
                                        c, gamma);
                } else {
                    pmin = pmid;
                    min_guess = root_finding(pmin, Ds[idx][idy],
                            Sxs[idx][idy], Sys[idx][idy], taus[idx][idy],
                            c, gamma);
                }

                difference = max_guess - min_guess;
                counter++;
            }

            pbars[idx][idy] = 0.5 * (pmin + pmax);

        }
        """)

    find_p = mod.get_function("find_p_c")

    D = Q[:,:,iD].astype(np.float32)
    Sx = Q[:,:,iSx].astype(np.float32)
    Sy = Q[:,:,iSy].astype(np.float32)
    tau = Q[:,:,itau].astype(np.float32)
    DX = Q[:,:,iDX].astype(np.float32)

    # allocate memory on device
    D_gpu = cuda.mem_alloc(D.nbytes)
    Sx_gpu = cuda.mem_alloc(Sx.nbytes)
    Sy_gpu = cuda.mem_alloc(Sy.nbytes)
    tau_gpu = cuda.mem_alloc(tau.nbytes)

    c_gpu = cuda.mem_alloc(c.nbytes)
    gamma_gpu = cuda.mem_alloc(gamma.nbytes)

    V = np.zeros((qx, qy, nvar), dtype=np.float32)

    pmin = (Sx**2 + Sy**2)/c**2 - tau - D
    pmax = (gamma - 1.) * tau

    pmin_gpu = cuda.mem_alloc(pmin.nbytes)
    pmax_gpu = cuda.mem_alloc(pmax.nbytes)

    pbar_gpu = cuda.mem_alloc(pmax.nbytes)

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

    # transfer to gpu
    cuda.memcpy_htod(D_gpu, D)
    cuda.memcpy_htod(Sx_gpu, Sx)
    cuda.memcpy_htod(Sy_gpu, Sy)
    cuda.memcpy_htod(tau_gpu, tau)
    cuda.memcpy_htod(c_gpu, c)
    cuda.memcpy_htod(gamma_gpu, gamma)
    cuda.memcpy_htod(pmin_gpu, pmin)
    cuda.memcpy_htod(pmax_gpu, pmax)
    cuda.memcpy_htod(pbar_gpu, pmin)

    find_p(pbar_gpu, pmin_gpu, pmax_gpu, D_gpu, Sx_gpu, Sy_gpu, tau_gpu,
            c_gpu, gamma_gpu, block=(qx, qy, 1))

    cuda.memcpy_dtoh(V[:,:,itau], pbar_gpu)

    # NOTE: would it be quicker to do this as loops in cython??
    #try:
    #    V[:,:,itau] = [[brentq(interface_f.root_finding, pmin[i,j], pmax[i,j], args=(D[i,j], Sx[i,j], Sy[i,j], tau[i,j], c, gamma)) for j in range(qy)] for i in range(qx)]
    #except ValueError:
    #    print('pmin: {}'.format(pmin))
    #    print('pmax: {}'.format(pmax))

    V[:,:,iSx] = Sx / (tau + D + V[:,:,itau])
    V[:,:,iSy] = Sy / (tau + D + V[:,:,itau])

    w = 1. / np.sqrt(1. - (V[:,:,iSx]**2 + V[:,:,iSy]**2) / c**2)

    V[:,:,iD] = D / w
    V[:,:,iDX] = DX / D

    return V

def arr_root_find_on_me(pbar, D, Sx, Sy, tau, c, gamma):
    """
    Equation to root find on in order to find the primitive pressure.
    This works on arrays.
    """
    #if pbar[pbar > 0.]:
    v2 = (Sx**2 + Sy**2) / (c * (tau + D + pbar))**2
    w = 1. / np.sqrt(1. - v2)
    epsrho = (tau + D * (1. - w) + pbar * v2 / (v2 - 1.)) / w**2

    #neg_p = (pbar <= 0.)
    return (gamma - 1.) * epsrho - pbar
    #pbar[neg_p] = 1.e6

    #return pbar
