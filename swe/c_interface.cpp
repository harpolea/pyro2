#include <math.h>
#include "interface_h.h"

void states_c(int idir, int qx, int qy, int ng,
              double dx, double dt,
              int ih, int iu, int iv, int ix, int nvar, int nspec,
              double g,
              double *qv, double *dqv,
              double *q_l, double *q_r)

{
/*
   predict the cell-centered state to the edges in one-dimension
   using the reconstructed, limited slopes.

   We follow the convection here that V_l[i] is the left state at the
   i-1/2 interface and V_l[i+1] is the left state at the i+1/2
   interface.


   We need the left and right eigenvectors and the eigenvalues for
   the system projected along the x-direction

   Taking our state vector as Q = (rho, u, v, p)^T, the eigenvalues
   are u - c, u, u + c.

   We look at the equations of hydrodynamics in a split fashion --
   i.e., we only consider one dimension at a time.

   Considering advection in the x-direction, the Jacobian matrix for
   the primitive variable formulation of the Euler equations
   projected in the x-direction is:

       / u   0   0 \
       | g   u   0 |
   A = \ 0   0   u /

   The right eigenvectors are

        /  h  \       /  0  \      /  h  \
   r1 = | -c  |  r2 = |  0  | r3 = |  c  |
        \  0  /       \  1  /      \  0  /

   The left eigenvectors are

   l1 =     ( 1/(2h),  -h/(2hc),  0 )
   l2 =     ( 0,          0,  1 )
   l3 =     ( -1/(2h), -h/(2hc),  0 )

   The fluxes are going to be defined on the left edge of the
   computational zones

      |             |             |             |
      |             |             |             |
     -+------+------+------+------+------+------+--
      |     i-1     |      i      |     i+1     |
                   ^ ^           ^
               q_l,i q_r,i  q_l,i+1

   q_r,i and q_l,i+1 are computed using the information in zone i,j.
 */

	double dq[nvar];
	double q[nvar];
	double lvec[nvar*nvar];
	double rvec[nvar*nvar];
	double eval[nvar];
	double betal[nvar];
	double betar[nvar];

	double factor;

	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx;
	int jlo = ng;
	int jhi = ng+ny;

	int ns = nvar - nspec;

	double dtdx = dt/dx;
	double dtdx3 = 0.33333*dtdx;

	// this is the loop over zones.  For zone i, we see q_l[i+1] and q_r[i]
	for (int i = ilo-2; i < ihi+2; i++) {
		for (int j = jlo-2; j < jhi+2; j++) {
			int idx = (i*qy + j) * nvar;

			for (int m = 0; m < nvar; m++) {

				dq[m] = dqv[idx+m];
				q[m] = qv[idx+m];
				eval[m] = 0;

				for (int n = 0; n < nvar; n++) {
					lvec[m*nvar+n] = 0;
					rvec[m*nvar+n] = 0;
				}
			}

			double cs = sqrt(g*q[ih]);

			// compute the eigenvalues and eigenvectors
			if (idir == 1) {
				double eval_tmp[3] = {q[iu] - cs, q[iu], q[iu] + cs};

				double l0[3] = {cs, -q[ih], 0.0};
				double l1[3] = {0.0, 0.0, 1.0};
				double l2[3] = {cs, q[ih], 0.0};

				double r0[3] = {q[ih], -cs, 0.0};
				double r1[3] = {0.0, 0.0, 1.0};
				double r2[3] = {q[ih], cs, 0.0};

				for (int n = 0; n < 3; n++) {
					eval[n] = eval_tmp[n];

					lvec[n] = l0[n];
					lvec[nvar+n] = l1[n];
					lvec[2*nvar+n] = l2[n];

					rvec[n] = r0[n];
					rvec[nvar+n] = r1[n];
					rvec[2*nvar+n] = r2[n];

					// multiply by scaling factors
					lvec[n] *= 0.50 / (cs * q[ih]);
					lvec[2*nvar+n] *= -0.50 / (cs * q[ih]);
				}

				for (int n = ns; n < nvar; n++)
					eval[n] = q[iu];

				// now the species -- they only have a 1 in their corresponding slot

				for (int n = ix; n < ix+nspec; n++) {
					lvec[n*nvar+n] = 1.0;
					rvec[n*nvar+n] = 1.0;
				}

			} else {
				double eval_tmp[3] = {q[iv] - cs, q[iv], q[iv] + cs};

				double l0[3] = {cs, 0.0, -q[ih]};
				double l1[3] = {0.0, 1.0, 0.0};
				double l2[3] = {cs, 0.0, q[ih]};

				double r0[3] = {q[ih], 0.0, -cs};
				double r1[3] = {0.0, 1.0, 0.0};
				double r2[3] = {q[ih], 0.0, cs};

				for (int n = 0; n < 3; n++) {
					eval[n] = eval_tmp[n];

					lvec[n] = l0[n];
					lvec[nvar+n] = l1[n];
					lvec[2*nvar+n] = l2[n];

					rvec[n] = r0[n];
					rvec[nvar+n] = r1[n];
					rvec[2*nvar+n] = r2[n];

					// multiply by scaling factors
					lvec[n] *= 0.50 / (cs * q[ih]);
					lvec[2*nvar+n] *= -0.50 / (cs * q[ih]);
				}

				for (int n = ns; n < nvar; n++)
					eval[n] = q[iv];

				// now the species -- they only have a 1 in their corresponding slot

				for (int n = ix; n < ix+nspec; n++) {
					lvec[n*nvar+n] = 1.0;
					rvec[n*nvar+n] = 1.0;
				}
			}

			// define the reference states
			if (idir == 1) {
				// this is one the right face of the current zone,
				// so the fastest moving eigenvalue is eval[2] = u + c
				factor = 0.5*(1.0 - dtdx*fmax(eval[2], 0.0));
				for (int n = 0; n < nvar; n++)
					q_l[((i+1)*qy+j)*nvar+n] = q[n] + factor*dq[n];

				// left face of the current zone, so the fastest moving
				// eigenvalue is eval[3] = u - c
				factor = 0.5*(1.0 + dtdx*fmin(eval[0], 0.0));
				for (int n = 0; n < nvar; n++)
					q_r[idx+n] = q[n] - factor*dq[n];

			} else {

				factor = 0.5*(1.0 - dtdx*fmax(eval[2], 0.0));
				for (int n = 0; n < nvar; n++)
					q_l[(i*qy+j+1)*nvar+n] = q[n] + factor*dq[n];

				factor = 0.5*(1.0 + dtdx*fmin(eval[0], 0.0));
				for (int n = 0; n < nvar; n++)
					q_r[idx+n] = q[n] - factor*dq[n];

			}

			// compute the Vhat functions
			for (int m = 0; m < nvar; m++) {
				double sum = 0;
				for (int n = 0; n < nvar; n++)
					sum += lvec[m*nvar+n]*dq[n];

				betal[m] = dtdx3*(eval[2] - eval[m]) *
				           (copysign(1.0,eval[m]) + 1.0)*sum;
				betar[m] = dtdx3*(eval[0] - eval[m]) *
				           (1.0 - copysign(1.0,eval[m]))*sum;
			}

			// construct the states
			for (int m = 0; m < nvar; m++) {
				double sum_l = 0;
				double sum_r = 0;
				for (int n = 0; n < nvar; n++) {
					sum_l += betal[n] * rvec[n*nvar+m];
					sum_r += betar[n] * rvec[n*nvar+m];
				}

				if (idir == 1) {
					q_l[((i+1)*qy+j)*nvar+m] = q_l[((i+1)*qy+j)*nvar+m] + sum_l;
					q_r[idx+m] = q_r[idx+m] + sum_r;
				} else {
					q_l[(i*qy+j+1)*nvar+m] = q_l[(i*qy+j+1)*nvar+m] + sum_l;
					q_r[idx+  m] = q_r[idx+  m] + sum_r;
				}

			}
		}
	}
}


void riemann_Roe_c(int idir, int qx, int qy, int ng,
                   int nvar, int ih, int ixmom, int iymom,
                   int ihX, int nspec,
                   int lower_solid, int upper_solid,
                   double g, double *U_l, double *U_r,
                   double *F)

{
	// This is the Roe Riemann solver with entropy fix. The implementation
	// follows Toro's SWE book and the clawpack 2d SWE Roe solver.

	double smallc = 1.e-10;
	double tol = 0.1e-1; // entropy fix parameter
	// Note that I've basically assumed that cfl = 0.1 here to get away with
	// not passing dx/dt or cfl to this function. If this isn't the case, will need
	// to pass one of these to the function or } else { things will go wrong.

	double h_l, un_l, ut_l, c_l;
	double h_r, un_r, ut_r, c_r;
	double h_star, u_star, c_star;

	double U_roe[nvar];
	double c_roe, un_roe;
	double lambda_roe[nvar];
	double K_roe[nvar*nvar];
	double alpha_roe[nvar];
	double delta[nvar];
	double F_l[nvar];
	double F_r[nvar];

	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx;
	int jlo = ng;
	int jhi = ng+ny;

	int ns = nvar - nspec;

	for (int i = ilo-1; i < ihi+1; i++) {
		for (int j = jlo-1; j < jhi+1; j++) {
			int idx = (i*qy+j)*nvar;

			// primitive variable states
			h_l  = U_l[idx+ih];

			// un = normal velocity; ut = transverse velocity
			if (idir == 1) {
				un_l    = U_l[idx+ixmom]/h_l;
				ut_l    = U_l[idx+iymom]/h_l;
			} else {
				un_l    = U_l[idx+iymom]/h_l;
				ut_l    = U_l[idx+ixmom]/h_l;
			}

			h_r  = U_r[idx+ih];

			if (idir == 1) {
				un_r    = U_r[idx+ixmom]/h_r;
				ut_r    = U_r[idx+iymom]/h_r;
			} else {
				un_r    = U_r[idx+iymom]/h_r;
				ut_r    = U_r[idx+ixmom]/h_r;
			}

			// compute the sound speeds
			c_l = fmax(smallc, sqrt(g*h_l));
			c_r = fmax(smallc, sqrt(g*h_r));

			for (int n = 0; n < nvar; n++) {
				// Calculate the Roe averages
				U_roe[n] = (U_l[idx+n]/sqrt(h_l) +
				            U_r[idx+n]/sqrt(h_r)) /
				           (sqrt(h_l) + sqrt(h_r));

				delta[n] = U_r[idx+n]/h_r - U_l[idx+n]/h_l;
			}



			U_roe[ih] = sqrt(h_l * h_r);
			c_roe = sqrt(0.5 * (c_l*c_l + c_r*c_r));
			delta[ih] = h_r - h_l;

			// evalues and right evectors
			if (idir == 1) {
				un_roe = U_roe[ixmom];
			} else {
				un_roe = U_roe[iymom];
			}

			for (int n = 0; n < nvar*nvar; n++)
				K_roe[n] = 0;

			double lamda_roe_tmp[3] = {un_roe - c_roe,
				                       un_roe, un_roe + c_roe};

			for (int n = 0; n < 3; n++)
				lambda_roe[n] = lamda_roe_tmp[n];

			if (idir == 1) {
				double alpha_roe_tmp[3] = {0.5*(delta[ih] - U_roe[ih]/c_roe*delta[ixmom]),
					                       U_roe[ih] * delta[iymom],
					                       0.5*(delta[ih] + U_roe[ih]/c_roe*delta[ixmom])};

				double K_roe0[3] = {1.0, un_roe - c_roe,
					                U_roe[iymom]};
				double K_roe1[3] = {0.0, 0.0, 1.0};
				double K_roe2[3] = {1.0, un_roe + c_roe,
					                U_roe[iymom]};

				for (int n = 0; n < 3; n++) {
					alpha_roe[n] = alpha_roe_tmp[n];
					K_roe[n] = K_roe0[n];
					K_roe[nvar+n] = K_roe1[n];
					K_roe[2*nvar+n] = K_roe2[n];
				}
			} else {
				double alpha_roe_tmp[3] = {0.5*(delta[ih] - U_roe[ih]/c_roe*delta[iymom]),
					                       U_roe[ih] * delta[ixmom],
					                       0.5*(delta[ih] + U_roe[ih]/c_roe*delta[iymom])};

				double K_roe0[3] = {1.0, U_roe[ixmom],
					                un_roe - c_roe};
				double K_roe1[3] = {0.0, 1.0, 0.0};
				double K_roe2[3] = {1.0, U_roe[ixmom],
					                un_roe + c_roe};

				for (int n = 0; n < 3; n++) {
					alpha_roe[n] = alpha_roe_tmp[n];
					K_roe[n] = K_roe0[n];
					K_roe[nvar+n] = K_roe1[n];
					K_roe[2*nvar+n] = K_roe2[n];
				}
			}

			for (int n = ns; n < nvar; n++) {
				lambda_roe[n] = un_roe;
				alpha_roe[n] = U_roe[ih] * delta[n];

				for (int m = 0; m < nvar; m++)
					K_roe[n*nvar+m] = 0;

				K_roe[n*nvar+n] = 1.0;
			}

			double U_s[nvar];

			for (int n = 0; n < nvar; n++)
				U_s[n] = U_l[idx+n];

			consFlux(idir, g, ih, ixmom, iymom, ihX,
			         nvar, nspec,
			         U_s, F_l);

			for (int n = 0; n < nvar; n++)
				U_s[n] = U_r[idx+n];

			consFlux(idir, g, ih, ixmom, iymom, ihX,
			         nvar, nspec,
			         U_s, F_r);

			for (int n = 0; n < nvar; n++)
				F[idx+n] = 0.5 * (F_l[n] + F_r[n]);

			h_star = 1.0 / g * pow(0.5 * (c_l + c_r) +
			                       0.25 * (un_l - un_r),2);
			u_star = 0.5 * (un_l + un_r) + c_l - c_r;

			c_star = sqrt(g * h_star);

			// modified evalues for entropy fix
			if (fabs(lambda_roe[0]) < tol) {
				lambda_roe[0] = lambda_roe[0] *
				                (u_star - c_star - lambda_roe[0]) /
				                (u_star - c_star -
				                 (un_l - c_l));
			}
			if (fabs(lambda_roe[2]) < tol) {
				lambda_roe[2] = lambda_roe[2] *
				                (u_star + c_star - lambda_roe[2]) /
				                (u_star + c_star -
				                 (un_r + c_r));
			}

			for (int n =  0; n < nvar; n++) {
				for (int m = 0; m < nvar; m++) {
					F[idx+n] = F[idx+n] -
					           0.5 * alpha_roe[m] *
					           fabs(lambda_roe[m]) * K_roe[m*nvar+n];
				}
			}

		}
	}
}


void riemann_HLLC_c(int idir, int qx, int qy, int ng,
                    int nvar, int ih, int ixmom, int iymom,
                    int ihX, int nspec,
                    int lower_solid, int upper_solid,
                    double g, double *U_l, double *U_r,
                    double *F)

{

	// this is the HLLC Riemann solver.  The implementation follows
	// directly out of Toro's book.  Note: this does not handle the
	// transonic rarefaction.

	double smallc = 1.e-10;

	double h_l, un_l, ut_l;
	double h_r, un_r, ut_r;

	double h_avg;
	double hstar, ustar, u_avg;
	double S_l, S_r, S_c;
	double c_l, c_r, c_avg;

	double U_state[nvar];
	double HLLCfactor;

	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx;
	int jlo = ng;
	int jhi = ng+ny;

	for (int i = ilo-1; i < ihi+1; i++) {
		for (int j = jlo-1; j < jhi+1; j++) {
			int idx = (i*qy + j) * nvar;

			// primitive variable states
			h_l  = U_l[idx+ih];

			// un = normal velocity; ut = transverse velocity
			if (idir == 1) {
				un_l    = U_l[idx+ixmom]/h_l;
				ut_l    = U_l[idx+iymom]/h_l;
			} else {
				un_l    = U_l[idx+iymom]/h_l;
				ut_l    = U_l[idx+ixmom]/h_l;
			}

			h_r  = U_r[idx+ih];

			if (idir == 1) {
				un_r    = U_r[idx+ixmom]/h_r;
				ut_r    = U_r[idx+iymom]/h_r;
			} else {
				un_r    = U_r[idx+iymom]/h_r;
				ut_r    = U_r[idx+ixmom]/h_r;
			}
	// Solve riemann shock tube problem for a general equation of
	// state using the method of Colella, Glaz, and Ferguson.  See
	// Almgren et al. 2010 (the CASTRO paper) for details.
	//
	// The Riemann problem for the Euler's equation produces 4 regions,
	// separated by the three characteristics (u - cs, u, u + cs) {
	//
	//
	//        u - cs    t    u      u + cs
	//                 ^   .       /
	//             *L  |   . *R   /
	//                 |  .     /
	//                 |  .    /
	//         L       | .   /    R
	//                 | .  /
	//                 |. /
	//                 |./
	//        ----------+----------------> x
	//
	// We care about the solution on the axis.  The basic idea is to use
	// estimates of the wave speeds to figure out which region we are in,
	// and: use jump conditions to evaluate the state there.
	//
	// Only density jumps across the u characteristic.  All primitive
	// variables jump across the other two.  Special attention is needed
	// if a rarefaction spans the axis.
			// compute the sound speeds
			c_l = fmax(smallc, sqrt(g*h_l));
			c_r = fmax(smallc, sqrt(g*h_r));

			// Estimate the star quantities -- use one of three methods to
			// do this -- the primitive variable Riemann solver, the two
			// shock approximation, or the two rarefaction approximation.
			// Pick the method based on the pressure states at the
			// interface.

			h_avg = 0.5*(h_l + h_r);
			c_avg = 0.5*(c_l + c_r);
			u_avg = 0.5*(un_l + un_r);

			hstar = h_avg - 0.25 * (un_r - un_l) * h_avg / c_avg;
			ustar = u_avg - (h_r - h_l) * c_avg / h_avg;

			// estimate the nonlinear wave speeds

			if (hstar <= h_l) {
				// rarefaction
				S_l = un_l - c_l;
			} else {
				// shock
				S_l = un_l - c_l *
				      sqrt(0.5 * (hstar+h_l) * hstar) / h_l;
			}

			if (hstar <= h_r) {
				// rarefaction
				S_r = un_r + c_r;
			} else {
				// shock
				S_r = un_r + c_r*sqrt(0.5 * (hstar+h_r) * hstar) / h_r;
			}

			S_c = (S_l*h_r*(un_r-S_r) - S_r*h_l*(un_l-S_l)) /
			      (h_r*(un_r-S_r) - h_l*(un_l-S_l));

			double F_state[nvar];
			double U_s[nvar];

			// figure out which region we are in and compute the state and
			// the interface fluxes using the HLLC Riemann solver
			if (S_r <= 0.0) {
				// R region
				for (int n = 0; n < nvar; n++)
					U_state[n] = U_r[idx+n];

				consFlux(idir, g, ih, ixmom, iymom, ihX, nvar, nspec,
				         U_state, F_state);
				for (int n = 0; n < nvar; n++)
					F[idx+n] = F_state[n];

			} else if (S_r > 0.0 && S_c <= 0) {
				// R* region
				HLLCfactor = h_r*(S_r - un_r)/(S_r - S_c);

				U_state[ih] = HLLCfactor;

				if (idir == 1) {
					U_state[ixmom] = HLLCfactor*S_c;
					U_state[iymom] = HLLCfactor*ut_r;
				} else {
					U_state[ixmom] = HLLCfactor*ut_r;
					U_state[iymom] = HLLCfactor*S_c;
				}

				// species
				if (nspec > 0) {
					for (int n = 0; n < nspec; n++)
						U_state[ihX+n] = HLLCfactor*U_r[idx+ihX+n]/h_r;
				}

				for (int n = 0; n < nvar; n++)
					U_s[n] = U_r[idx+n];

				// find the flux on the right interface
				consFlux(idir, g, ih, ixmom, iymom, ihX, nvar, nspec,
				         U_s, F_state);

				// correct the flux
				for (int n = 0; n < nvar; n++)
					F[idx+n] = F_state[n] +
					           S_r*(U_state[n] - U_r[idx+n]);

			} else if (S_c > 0.0 && S_l < 0.0) {
				// L* region
				HLLCfactor = h_l*(S_l - un_l)/(S_l - S_c);

				U_state[ih] = HLLCfactor;

				if (idir == 1) {
					U_state[ixmom] = HLLCfactor*S_c;
					U_state[iymom] = HLLCfactor*ut_l;
				} else {
					U_state[ixmom] = HLLCfactor*ut_l;
					U_state[iymom] = HLLCfactor*S_c;
				}

				// species
				if (nspec > 0) {
					for (int n = 0; n < nspec; n++)
						U_state[ihX+n] = HLLCfactor*U_l[idx+ihX+n]/h_l;
				}

				for (int n = 0; n < nvar; n++)
					U_s[n] = U_l[idx+n];

				// find the flux on the left interface
				consFlux(idir, g, ih, ixmom, iymom, ihX, nvar, nspec,
				         U_s, F_state);

				// correct the flux
				for (int n = 0; n < nvar; n++)
					F[idx+n] = F_state[n] + S_l*(U_state[n] - U_l[idx+n]);

			} else {
				// L region
				for (int n = 0; n < nvar; n++)
					U_state[n] = U_l[idx+n];

				consFlux(idir, g, ih, ixmom, iymom, ihX, nvar, nspec,
				         U_state, F_state);

				for (int n = 0; n < nvar; n++)
					F[idx+n] = F_state[n];

			}

			// we should deal with solid boundaries somehow here

		}
	}
}


void consFlux(int idir, double g, int ih, int ixmom,
              int iymom, int ihX, int nvar, int nspec,
              double *U_state, double *F)  {

    /*

   Calculate the conserved flux for the shallow water equations. In the
   x-direction, this is given by

       /      hu       \
   F = | hu^2 + gh^2/2 |
       \      huv      /
    */

	double u = U_state[ixmom]/U_state[ih];
	double v = U_state[iymom]/U_state[ih];

	if (idir == 1) {
		F[ih] = U_state[ih]*u;
		F[ixmom] = U_state[ixmom]*u + 0.5 * g * U_state[ih]*U_state[ih];
		F[iymom] = U_state[iymom]*u;
		if (nspec > 0) {
			for (int n = 0; n < nspec; n++)
				F[ihX+n] = U_state[ihX+n]*u;
		}
	} else {
		F[ih] = U_state[ih]*v;
		F[ixmom] = U_state[ixmom]*v;
		F[iymom] = U_state[iymom]*v + 0.5 * g * U_state[ih]*U_state[ih];
		if (nspec > 0) {
			for (int n = 0; n < nspec; n++)
				F[ihX+n] = U_state[ihX+n]*v;
		}
	}

}
