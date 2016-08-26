import lm_gr.LM_gr_interface_f as interface_f
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

def cons_to_prim(Q, c, gamma, alphasq, gamma_mat, myg, var):
    """
    PyCUDA implementation of code to change the vector of conservative variables (D, Ux, Uy, Dh, DX) into the vector of primitive variables (rho, u, v, p, X). Root finder brentq is applied to the fortran function root_finding from interface_f.

    Main looping done as a list comprehension as this is faster than nested for loops in pure python - not so sure this is the case for cython?
    """

    nx = myg.qx
    ny = myg.qy

    V = myg.scratch_array(var.nvar)

    mod = SourceModule("""
        #include <math.h>
        #include <stdio.h>
        __device__ float root_finding(float pbar, float D, float Ux, float Uy,
                                    float Dh, float alphasq, float* gamma_mat,
                                    float c, float gamma)
        {
            float largep = 1.e6;

            if (pbar > 0.0) {
                float rho = D * sqrt(alphasq - (gamma_mat[0] * Ux * Ux +
                    gamma_mat[3] * Uy*Uy +
                    2.0 * gamma_mat[1] * Ux * Uy)/pow(c,2)) / c;

                //printf("rho: %f    ", rho);
                float h = Dh / D;

                pbar = (gamma - 1.0) * (h - 1.0) * rho / gamma - pbar;
            } else {
                pbar = largep;
            }

            return pbar;
        }

        __global__ void find_p_c(float* pbars, float* pmins, float* pmaxes,
                            float* Ds, float* Uxs, float* Uys,
                            float* Dhs, float* alphasqs, float* gamma_mats,
                            float gamma, float c, int nx, int ny)
        {
            const int ITMAX=1000;
            const float TOL=1.e-5;

            int n_x = blockDim.x * gridDim.x;
            int idx = threadIdx.x + blockDim.x*blockIdx.x;
            int idy = threadIdx.y + blockDim.y*blockIdx.y;
            int tid = idy * n_x + idx;

            float gamma_mat[4];

            if (tid < nx*ny) { //((idx < nx) && (idy < ny)) {

                for (int i=0; i < 4; i++) {
                    gamma_mat[i] = gamma_mats[tid * 4 + i];
                }

                //printf("%f, %f, %f", gamma_mat[0], gamma_mat[1], gamma_mat[3]);

                float pa = pmins[tid];
                float pb = pmaxes[tid];

                int counter = 0;

                float fa = root_finding(pa, Ds[tid],
                        Uxs[tid], Uys[tid], Dhs[tid], alphasqs[tid],
                        gamma_mat, c, gamma);
                float fb = root_finding(pb, Ds[tid],
                                    Uxs[tid], Uys[tid], Dhs[tid], alphasqs[tid],
                                    gamma_mat, c, gamma);

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
                            Uxs[tid], Uys[tid], Dhs[tid], alphasqs[tid],
                            gamma_mat, c, gamma);
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

                //if (counter == ITMAX) {
                //    printf("Reached ITMAX  ");
                //}

                //printf("abs difference: %f, min_guess %f, max_guess %f, counter: %d   ", abs(float(difference)), max_guess, min_guess, counter);

                //printf("pmin %f, pmax %f     ", pmin, pmax);

                pbars[tid] = pb;
            }
            //if (tid > 4000) {printf("tid: %d, pbar: %f  ", tid, pbars[tid]);}
        }
        """)

    find_p = mod.get_function("find_p_c")

    D = Q.d[:,:,var.iD].astype(np.float32)
    Ux = Q.d[:,:,var.iUx].astype(np.float32)
    Uy = Q.d[:,:,var.iUy].astype(np.float32)
    Dh = Q.d[:,:,var.iDh].astype(np.float32)
    DX = Q.d[:,:,var.iDX].astype(np.float32)

    # allocate memory on device
    D_gpu = cuda.mem_alloc(D.nbytes)
    Ux_gpu = cuda.mem_alloc(Ux.nbytes)
    Uy_gpu = cuda.mem_alloc(Uy.nbytes)
    Dh_gpu = cuda.mem_alloc(Dh.nbytes)
    alphasq_gpu = cuda.mem_alloc(alphasq.astype(np.float32).nbytes)
    gamma_mat_gpu = cuda.mem_alloc(gamma_mat.astype(np.float32).nbytes)

    pmin = (Dh - D) * (gamma - 1.) / gamma
    pmax = (gamma - 1.) * Dh

    pmin_gpu = cuda.mem_alloc(pmin.astype(np.float32).nbytes)
    pmax_gpu = cuda.mem_alloc(pmax.astype(np.float32).nbytes)

    pbar = np.zeros_like(pmin, dtype=np.float32)
    pbar_gpu = cuda.mem_alloc(pbar.nbytes)

    pmax[pmax < 0.] = np.fabs(pmax[pmax < 0.])
    pmin[pmin > pmax] = 0.

    pmin[pmin < 0.] = 0.
    #pmin[arr_root_find_on_me(pmin, D, Ux, Uy, Dh, alphasq, gamma_mat, gamma, c) < 0.] = 0.
    pmax[pmax == 0.] = c

    # find nans
    pmin[np.isnan(pmin)] = 0.
    pmax[np.isnan(pmax)] = c

    # transfer to gpu
    cuda.memcpy_htod(D_gpu, D)
    cuda.memcpy_htod(Ux_gpu, Ux)
    cuda.memcpy_htod(Uy_gpu, Uy)
    cuda.memcpy_htod(Dh_gpu, Dh)
    cuda.memcpy_htod(alphasq_gpu, alphasq.astype(np.float32))
    cuda.memcpy_htod(gamma_mat_gpu, gamma_mat.astype(np.float32))
    cuda.memcpy_htod(pmin_gpu, pmin.astype(np.float32))
    cuda.memcpy_htod(pmax_gpu, pmax.astype(np.float32))
    cuda.memcpy_htod(pbar_gpu, pbar)

    # calculate thread, block sizes
    block_dims = (32, 32, 1)
    grid_dims = (int(np.ceil(float(nx)/float(block_dims[0]))), int(np.ceil(float(ny)/float(block_dims[1]))))

    find_p(pbar_gpu, pmin_gpu, pmax_gpu, D_gpu, Ux_gpu, Uy_gpu, Dh_gpu,
            alphasq_gpu, gamma_mat_gpu,
            np.float32(gamma), np.float32(c), np.int32(nx), np.int32(ny), grid=grid_dims, block=block_dims)


    cuda.memcpy_dtoh(pbar, pbar_gpu)
    V.d[:,:,var.iDh] = pbar[:,:] #pbar_gpu.get()

    u0 = (Dh - D) * (gamma - 1.) / (gamma * V.d[:,:,var.iDh])

    V.d[:,:,var.iUx] = V.d[:,:,var.iUx] * u0
    V.d[:,:,var.iUy] = V.d[:,:,var.iUy] * u0

    V.d[:,:,var.iD] = D / u0
    V.d[:,:,var.iDX] = DX / D

    return V

def arr_root_find_on_me(pbar, D, Ux, Uy, Dh, alphasq, gamma_mat, gamma, c):
    """
    Equation to root find on in order to find the primitive pressure.
    This works on arrays.
    """
    # eps * rho
    rho = D * np.sqrt(alphasq - (gamma_mat[:,:,0,0] * Ux**2 + gamma_mat[:,:,1,1] * Uy**2 + 2. * gamma_mat[:,:,0,1] * Ux * Uy)/c**2) / c
    h = Dh / D

    return (gamma - 1.) * (h - 1.) * rho / gamma  - pbar
