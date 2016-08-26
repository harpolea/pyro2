import compressible_gr.interface_f as interface_f
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

def initialise_c2p():
    """
    Initialise c2p CUDA function.
    """

    mod = SourceModule("""
        #include <math.h>
        #include <stdio.h>
        __device__ float root_finding(float pbar, float D, float Sx, float Sy,
                                    float tau, float c, float gamma)
        {
            double largep = 1.e6;

            if (pbar > 0.0) {
                float v2 = (Sx*Sx + Sy*Sy) / pow((c * (tau + D + pbar)),2);
                float w = 1.0 / sqrt(1.0 - v2);
                float epsrho = (tau + D * (1.0 - w)) * (1.0 - v2) - pbar * v2;

                pbar = (gamma - 1.0) * epsrho - pbar;
            } else {
                pbar = largep;
            }

            return pbar;
        }

        __global__ void find_p_c(float* pbars, float* pmins, float* pmaxes,
                            float* Ds, float* Sxs, float* Sys,
                            float* taus, float c, float gamma, int nx, int ny)
        {
            const int ITMAX=1000;
            const float TOL=1.e-5;

            int n_x = blockDim.x * gridDim.x;
            int idx = threadIdx.x + blockDim.x*blockIdx.x;
            int idy = threadIdx.y + blockDim.y*blockIdx.y;
            int tid = idy * n_x + idx;

            if ((idx < nx) && (idy < ny)) {

                float pa = pmins[tid];
                float pb = pmaxes[tid];

                int counter = 0;

                float fa = root_finding(pa, Ds[tid],
                        Sxs[tid], Sys[tid], taus[tid],
                        c, gamma);
                float fb = root_finding(pb, Ds[tid],
                                    Sxs[tid], Sys[tid], taus[tid],
                                    c, gamma);

                float pc = pa;
                float fc = fa;

                bool mflag = true;

                // initialise some variables here for later
                float s = 0;
                float d = 0;
                float fs = 0;

                if (abs(fa) < abs(fb)) {
                    // swap
                    s = pa;
                    fs = fa;
                    pa = pb;
                    fa = fb;
                    pb = s;
                    fb = fs;
                }

                while (!(fb == 0) && (abs(float(pb-pa)) > TOL)  && (counter < ITMAX)) {

                    if (!(fa == fc) && !(fb == fc)) {
                        s = pa * fb * fc / ((fa - fb) * (fa - fc)) + \
                        pb * fa * fc / ((fb - fa) * (fb - fc)) + \
                        pc * fa * fb / ((fc - fa) * (fc - fb));
                    } else {
                        s = pb - fb * (pb - pa) / (fb - fa);
                    }

                    if (((0.75 * pa + 0.25 * pb - s) * (pb - s) > 0) || \
                        (mflag  && (abs(s-pb) >= 0.5*abs(pb-pc))) || \
                        (!mflag && (abs(s-pb) >= 0.5*abs(pc-d))) || \
                        (mflag && (abs(pb-pc) < TOL)) || \
                        (!mflag && (abs(pc-d) < TOL))) {
                        s = 0.5 * (pa + pb);
                        mflag = true;
                    } else {
                        mflag = false;
                    }

                    fs = root_finding(s, Ds[tid],
                            Sxs[tid], Sys[tid], taus[tid],
                            c, gamma);
                    d = pc;
                    pc = pb;
                    fc = fb;

                    if (fa * fs < 0.0) {
                        pb = s;
                        fb = fs;
                    } else {
                        pa = s;
                        fa = fs;
                    }

                    if (abs(fa) < abs(fb)) {
                        // swap
                        s = pa;
                        fs = fa;
                        pa = pb;
                        fa = fb;
                        pb = s;
                        fb = fs;
                    }

                    counter++;
                }

                pbars[tid] = pb;
            }
        }
        """)

    return mod.get_function("find_p_c")


def cons_to_prim(find_p, Q, c, gamma, myg, var):#qx, qy, nvar, iD, iSx, iSy, itau, iDX):
    """
    PyCUDA implementation of code to change the vector of conservative variables (D, Sx, Sy, tau, DX) into the vector of primitive variables (rho, u, v, p, X). Root finder brentq is applied to the fortran function root_finding from interface_f.

    Main looping done as a list comprehension as this is faster than nested for loops in pure python - not so sure this is the case for cython?
    """

    nx = myg.qx
    ny = myg.qy

    V = myg.scratch_array(var.nvar)

    D = Q.d[:,:,var.iD].astype(np.float32)
    Sx = Q.d[:,:,var.iSx].astype(np.float32)
    Sy = Q.d[:,:,var.iSy].astype(np.float32)
    tau = Q.d[:,:,var.itau].astype(np.float32)
    DX = Q.d[:,:,var.iDX].astype(np.float32)

    # allocate memory on device
    D_gpu = cuda.mem_alloc(D.nbytes)
    Sx_gpu = cuda.mem_alloc(Sx.nbytes)
    Sy_gpu = cuda.mem_alloc(Sy.nbytes)
    tau_gpu = cuda.mem_alloc(tau.nbytes)

    #c_gpu = cuda.mem_alloc(c.nbytes)
    #gamma_gpu = cuda.mem_alloc((np.float32(gamma)).nbytes)

    #V = np.zeros((qx, qy, var.nvar), dtype=np.float32)

    pmin = (Sx**2 + Sy**2)/c**2 - tau - D
    pmax = (gamma - 1.) * tau

    pmin_gpu = cuda.mem_alloc(pmin.nbytes)
    pmax_gpu = cuda.mem_alloc(pmax.nbytes)

    pbar = np.zeros_like(pmin, dtype=np.float32)
    pbar_gpu = gpuarray.to_gpu(pbar)#cuda.mem_alloc(pbar.nbytes)

    pmax[pmax < 0.] = np.fabs(pmax[pmax < 0.])
    pmin[pmin > pmax] = abs(np.sqrt(Sx[pmin > pmax]**2 + Sy[pmin > pmax]**2)/c - tau[pmin > pmax] - D[pmin > pmax])

    pmin[pmin < 0.] = 0.
    #pmin[arr_root_find_on_me(pmin, D, Sx, Sy, tau, c, gamma) < 0.] = 0.
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
    #cuda.memcpy_htod(c_gpu, np.float32(c))
    #cuda.memcpy_htod(gamma_gpu, np.float32(gamma))
    cuda.memcpy_htod(pmin_gpu, pmin)
    cuda.memcpy_htod(pmax_gpu, pmax)
    #cuda.memcpy_htod(pbar_gpu, pbar)

    # calculate thread, block sizes
    block_dims = (32, 32, 1)
    grid_dims = (int(np.ceil(float(nx)/float(block_dims[0]))), int(np.ceil(float(ny)/float(block_dims[1]))))

    find_p(pbar_gpu, pmin_gpu, pmax_gpu, D_gpu, Sx_gpu, Sy_gpu, tau_gpu,
            np.float32(c), np.float32(gamma), np.int32(nx), np.int32(ny), grid=grid_dims, block=block_dims)

    #pbar = np.empty_like(pmin)
    #cuda.memcpy_dtoh(pbar, pbar_gpu)
    V.d[:,:,var.itau] = pbar_gpu.get()

    V.d[:,:,var.iSx] = Sx / (tau + D + V.d[:,:,var.itau])
    V.d[:,:,var.iSy] = Sy / (tau + D + V.d[:,:,var.itau])

    w = 1. / np.sqrt(1. - (V.d[:,:,var.iSx]**2 + V.d[:,:,var.iSy]**2) / c**2)

    V.d[:,:,var.iD] = D / w
    V.d[:,:,var.iDX] = DX / D

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
