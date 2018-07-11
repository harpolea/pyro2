#include <math.h>
#include "interface_h.h"

void states_c(int idir, int qx, int qy, int ng,
              double dx,
              double dt, int irho, int iu, int iv, int ip, int ix, int nvar, int nspec, double gamma,
              double *qv, double *dqv, double *q_l, double *q_r) {

	double dq[nvar];
	double q[nvar];
	double lvec[nvar*nvar];
	double rvec[nvar*nvar];
	double eval[nvar];
	double betal[nvar];
	double betar[nvar];

	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx;
	int jlo = ng;
	int jhi = ng+ny;

	int ns = nvar - nspec;

	double dtdx = dt/dx;
	double dtdx4 = 0.25*dtdx;

	// this is the loop over zones.  For zone i, we see q_l[i+1] and q_r[i]
	for (int i = ilo-2; i < ihi+2; i++) {
		for (int j = jlo-2; j < jhi+2; j++) {
			int idx = (i*qy + j)*nvar;

			for (int n = 0; n < nvar; n++) {
				dq[n] = dqv[idx+n];
				q[n] = qv[idx+n];

				eval[n] = 0.0;

				for (int m = 0; m < nvar; m++) {
					lvec[n*nvar+m] = 0.0;
					rvec[n*nvar+m] = 0.0;
				}
			}

			double cs = sqrt(gamma*q[ip]/q[irho]);

			// compute the eigenvalues and eigenvectors
			if (idir == 1) {
				double eval_temp[4] = {q[iu] - cs, q[iu], q[iu], q[iu] + cs};

				double l0[4] = { 0.0, -0.5*q[irho]/cs, 0.0, 0.5/(cs*cs)  };
				double l1[4] = {1.0, 0.0,             0.0, -1.0/(cs*cs) };
				double l2[4] = {0.0, 0.0,             1.0, 0.0          };
				double l3[4] = {0.0, 0.5*q[irho]/cs,  0.0, 0.5/(cs*cs)  };

				double r0[4] = {1.0, -cs/q[irho], 0.0, cs*cs };
				double r1[4] = {1.0, 0.0,       0.0, 0.0 };
				double r2[4] = {0.0, 0.0,       1.0, 0.0 };
				double r3[4] = {1.0, cs/q[irho],  0.0, cs*cs };

				for (int n = 0; n < 4; n++) {
					eval[n] = eval_temp[n];

					lvec[n] = l0[n];
					lvec[nvar+n] = l1[n];
					lvec[2*nvar+n] = l2[n];
					lvec[3*nvar+n] = l3[n];

					rvec[n] = r0[n];
					rvec[nvar+n] = r1[n];
					rvec[2*nvar+n] = r2[n];
					rvec[3*nvar+n] = r3[n];

				}

				// now the species -- they only have a 1 in their corresponding slot
				for (int n = ns; n < nvar; n++)
					eval[n] = q[iu];
				for (int n = ix; n < ix+nspec; n++) {
					lvec[n*nvar+n] = 1.0;
					rvec[n*nvar+n] = 1.0;
				}

			} else {
				double eval_temp[4] = {q[iv] - cs, q[iv], q[iv], q[iv] + cs};

				double l0[4] = { 0.0, 0.0, -0.5*q[irho]/cs, 0.5/(cs*cs)  };
				double l1[4] = { 1.0, 0.0, 0.0,             -1.0/(cs*cs) };
				double l2[4] = { 0.0, 1.0, 0.0,             0.0          };
				double l3[4] = { 0.0, 0.0, 0.5*q[irho]/cs,  0.5/(cs*cs)  };

				double r0[4] = {1.0, 0.0, -cs/q[irho], cs*cs };
				double r1[4] = {1.0, 0.0, 0.0,       0.0 };
				double r2[4] = {0.0, 1.0, 0.0,       0.0 };
				double r3[4] = {1.0, 0.0, cs/q[irho],  cs*cs };

				for (int n = 0; n < 4; n++) {
					eval[n] = eval_temp[n];

					lvec[n] = l0[n];
					lvec[nvar+n] = l1[n];
					lvec[2*nvar+n] = l2[n];
					lvec[3*nvar+n] = l3[n];

					rvec[n] = r0[n];
					rvec[nvar+n] = r1[n];
					rvec[2*nvar+n] = r2[n];
					rvec[3*nvar+n] = r3[n];

				}

				// now the species -- they only have a 1 in their corresponding slot
				for (int n = ns; n < nvar; n++)
					eval[n] = q[iv];
				for (int n = ix; n < ix+nspec; n++) {
					lvec[n*nvar+n] = 1.0;
					rvec[n*nvar+n] = 1.0;
				}
			}

			// define the reference states
			if (idir == 1) {
				// this is one the right face of the current zone,
				// so the fastest moving eigenvalue is eval[3] = u + c
				double factor = 0.5*(1.0 - dtdx*fmax(eval[3], 0.0));
				for (int n = 0; n < nvar; n++)
					q_l[((i+1)*qy+j)*nvar+n] = q[n] + factor*dq[n];

				// left face of the current zone, so the fastest moving
				// eigenvalue is eval[3] = u - c
				factor = 0.5*(1.0 + dtdx*fmin(eval[0], 0.0));
				for (int n = 0; n < nvar; n++)
					q_r[idx+n] = q[n] - factor*dq[n];

			} else {

				double factor = 0.5*(1.0 - dtdx*fmax(eval[3], 0.0));
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

				betal[m] = dtdx4*(eval[3] - eval[m])*(copysign(1.0, eval[m]) + 1.0)*sum;
				betar[m] = dtdx4*(eval[0] - eval[m])*(1.0 - copysign(1.0, eval[m]))*sum;
			}

			// construct the states
			for (int m = 0; m < nvar; m++) {
				double sum_l = 0;
				double sum_r = 0;
				for (int n = 0; n < nvar; n++) {
					sum_l += betal[n]*rvec[n*nvar+m];
					sum_r += betar[n]*rvec[n*nvar+m];
				}

				if (idir == 1) {
					q_l[((i+1)*qy+j)*nvar+m] = q_l[((i+1)*qy+j)*nvar+m] + sum_l;
					q_r[idx+m] = q_r[idx+m] + sum_r;
				} else {
					q_l[(i*qy+j+1)*nvar+m] = q_l[(i*qy+j+1)*nvar+m] + sum_l;
					q_r[idx+ m] = q_r[idx+m] + sum_r;
				}
			}
		}
	}
}


void riemann_cgf_c(int idir, int qx, int qy, int ng,
                   int nvar, int idens, int ixmom, int iymom,
                   int iener, int irhoX, int nspec,
                   int lower_solid, int upper_solid,
                   double gamma, double *U_l, double *U_r,
                   double *F) {

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

	double smallc = 1.e-10;
	double smallrho = 1.e-10;
	double smallp = 1.e-10;

	double rho_l, un_l, ut_l, p_l, rhoe_l;
	double rho_r, un_r, ut_r, p_r, rhoe_r;
	double xn[nspec];
	double rhostar_l, rhostar_r, rhoestar_l, rhoestar_r;
	double ustar, pstar, cstar_l, cstar_r;
	double lambda_l, lambdastar_l, lambda_r, lambdastar_r;
	double W_l, W_r, c_l, c_r, sigma, alpha;

	double rho_state, un_state, ut_state, p_state, rhoe_state;

	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx;
	int jlo = ng;
	int jhi = ng+ny;

	for (int i = ilo-1; i < ihi+1; i++) {
		for (int j = jlo-1; j < jhi+1; j++) {
			int idx = (i*qy + j)*nvar;

			// primitive variable states
			rho_l  = U_l[idx+idens];

			// un = normal velocity; ut = transverse velocity
			if (idir == 1) {
				un_l    = U_l[idx+ixmom]/rho_l;
				ut_l    = U_l[idx+iymom]/rho_l;
			} else {
				un_l    = U_l[idx+iymom]/rho_l;
				ut_l    = U_l[idx+ixmom]/rho_l;
			}

			rhoe_l = U_l[idx+iener] - 0.5*rho_l*(un_l*un_l + ut_l*ut_l);

			p_l   = rhoe_l*(gamma - 1.0);
			p_l = fmax(p_l, smallp);

			rho_r  = U_r[idx+idens];

			if (idir == 1) {
				un_r    = U_r[idx+ixmom]/rho_r;
				ut_r    = U_r[idx+iymom]/rho_r;
			} else {
				un_r    = U_r[idx+iymom]/rho_r;
				ut_r    = U_r[idx+ixmom]/rho_r;
			}

			rhoe_r = U_r[idx+iener] - 0.5*rho_r*(un_r*un_r + ut_r*ut_r);

			p_r   = rhoe_r*(gamma - 1.0);
			p_r = fmax(p_r, smallp);

			// define the Lagrangian sound speed
			W_l = fmax(smallrho*smallc, sqrt(gamma*p_l*rho_l));
			W_r = fmax(smallrho*smallc, sqrt(gamma*p_r*rho_r));

			// and the regular sound speeds
			c_l = fmax(smallc, sqrt(gamma*p_l/rho_l));
			c_r = fmax(smallc, sqrt(gamma*p_r/rho_r));

			// define the star states
			pstar = (W_l*p_r + W_r*p_l + W_l*W_r*(un_l - un_r))/(W_l + W_r);
			pstar = fmax(pstar, smallp);
			ustar = (W_l*un_l + W_r*un_r + (p_l - p_r))/(W_l + W_r);

			// now compute the remaining state to the left and right
			// of the contact (in the star region)
			rhostar_l = rho_l + (pstar - p_l)/(c_l*c_l);
			rhostar_r = rho_r + (pstar - p_r)/(c_r*c_r);

			rhoestar_l = rhoe_l +
			             (pstar - p_l)*(rhoe_l/rho_l + p_l/rho_l)/(c_l*c_l);
			rhoestar_r = rhoe_r +
			             (pstar - p_r)*(rhoe_r/rho_r + p_r/rho_r)/(c_r*c_r);

			cstar_l = fmax(smallc, sqrt(gamma*pstar/rhostar_l));
			cstar_r = fmax(smallc, sqrt(gamma*pstar/rhostar_r));

			// figure out which state we are in, based on the location of
			// the waves
			if (ustar > 0.0) {

				// contact is moving to the right, we need to understand
				// the L and *L states

				// Note: transverse velocity only jumps across contact
				ut_state = ut_l;

				// define eigenvalues
				lambda_l = un_l - c_l;
				lambdastar_l = ustar - cstar_l;

				if (pstar > p_l) {
					// the wave is a shock -- find the shock speed
					sigma = (lambda_l + lambdastar_l)/2.0;

					if (sigma > 0.0) {
						// shock is moving to the right -- solution is L state
						rho_state = rho_l;
						un_state = un_l;
						p_state = p_l;
						rhoe_state = rhoe_l;

					} else {
						// solution is *L state
						rho_state = rhostar_l;
						un_state = ustar;
						p_state = pstar;
						rhoe_state = rhoestar_l;
					}

				} else {
					// the wave is a rarefaction
					if ((lambda_l < 0.0) && (lambdastar_l < 0.0)) {
						// rarefaction fan is moving to the left -- solution is
						// *L state
						rho_state = rhostar_l;
						un_state = ustar;
						p_state = pstar;
						rhoe_state = rhoestar_l;

					} else if ((lambda_l > 0.0) && (lambdastar_l > 0.0)) {
						// rarefaction fan is moving to the right -- solution is
						// L state
						rho_state = rho_l;
						un_state = un_l;
						p_state = p_l;
						rhoe_state = rhoe_l;

					} else {
						// rarefaction spans x/t = 0 -- interpolate
						alpha = lambda_l/(lambda_l - lambdastar_l);

						rho_state  = alpha*rhostar_l  + (1.0 - alpha)*rho_l;
						un_state   = alpha*ustar      + (1.0 - alpha)*un_l;
						p_state    = alpha*pstar      + (1.0 - alpha)*p_l;
						rhoe_state = alpha*rhoestar_l + (1.0 - alpha)*rhoe_l;
					}
				}

			} else if (ustar < 0) {
				// contact moving left, we need to understand the R and *R
				// states

				// Note: transverse velocity only jumps across contact
				ut_state = ut_r;

				// define eigenvalues
				lambda_r = un_r + c_r;
				lambdastar_r = ustar + cstar_r;

				if (pstar > p_r) {
					// the wave if a shock -- find the shock speed
					sigma = (lambda_r + lambdastar_r)/2.0;

					if (sigma > 0.0) {
						// shock is moving to the right -- solution is *R state
						rho_state = rhostar_r;
						un_state = ustar;
						p_state = pstar;
						rhoe_state = rhoestar_r;

					} else {
						// solution is R state
						rho_state = rho_r;
						un_state = un_r;
						p_state = p_r;
						rhoe_state = rhoe_r;
					}

				} else {
					// the wave is a rarefaction
					if ((lambda_r < 0.0) && (lambdastar_r < 0.0)) {
						// rarefaction fan is moving to the left -- solution is
						// R state
						rho_state = rho_r;
						un_state = un_r;
						p_state = p_r;
						rhoe_state = rhoe_r;

					} else if ((lambda_r > 0.0) && (lambdastar_r > 0.0)) {
						// rarefaction fan is moving to the right -- solution is
						// *R state
						rho_state = rhostar_r;
						un_state = ustar;
						p_state = pstar;
						rhoe_state = rhoestar_r;

					} else {
						// rarefaction spans x/t = 0 -- interpolate
						alpha = lambda_r/(lambda_r - lambdastar_r);

						rho_state  = alpha*rhostar_r  + (1.0 - alpha)*rho_r;
						un_state   = alpha*ustar      + (1.0 - alpha)*un_r;
						p_state    = alpha*pstar      + (1.0 - alpha)*p_r;
						rhoe_state = alpha*rhoestar_r + (1.0 - alpha)*rhoe_r;
					}
				}

			} else {  // ustar == 0

				rho_state = 0.5*(rhostar_l + rhostar_r);
				un_state = ustar;
				ut_state = 0.5*(ut_l + ut_r);
				p_state = pstar;
				rhoe_state = 0.5*(rhoestar_l + rhoestar_r);
			}

			// species now
			if (nspec > 0) {
				if (ustar > 0.0) {
					for (int n = 0; n < nspec; n++)
						xn[n] = U_l[idx+irhoX+n]/U_l[idx+idens];

				} else if (ustar < 0.0) {
					for (int n = 0; n < nspec; n++)
						xn[n] = U_r[idx+irhoX+n]/U_r[idx+idens];
				} else {
					for (int n = 0; n < nspec; n++)
						xn[n] = 0.5*(U_l[idx+irhoX+n]/U_l[idx+idens] +
						             U_r[idx+irhoX+n]/U_r[idx+idens]);
				}
			}

			// are we on a solid boundary?
			if (idir == 1) {
				if ((i == ilo) && (lower_solid == 1))
					un_state = 0.0;

				if ((i == ihi+1) && (upper_solid == 1) )
					un_state = 0.0;

			} else {
				if ((j == jlo) && (lower_solid == 1))
					un_state = 0.0;

				if ((j == jhi+1) && (upper_solid == 1) )
					un_state = 0.0;
			}

			// compute the fluxes
			F[idx+idens] = rho_state*un_state;

			if (idir == 1) {
				F[idx+ixmom] = rho_state*un_state*un_state + p_state;
				F[idx+iymom] = rho_state*ut_state*un_state;
			} else {
				F[idx+ixmom] = rho_state*ut_state*un_state;
				F[idx+iymom] = rho_state*un_state*un_state + p_state;
			}


			F[idx+iener] = rhoe_state*un_state +
			               0.5*rho_state*(un_state*un_state + ut_state*ut_state)*un_state +
			               p_state*un_state;

			if (nspec > 0) {
				for (int n = 0; n < nspec; n++)
					F[idx+irhoX+n] = xn[n]*rho_state*un_state;
			}
		}

	}
}

void riemann_prim_c(int idir, int qx, int qy, int ng,
                    int nvar, int irho, int iu,
                    int iv, int ip,
                    int iX, int nspec,
                    int lower_solid, int upper_solid,
                    double gamma,
                    double *q_l,
                    double *q_r,
                    double *q_int) {

	// this is like riemann_cgf, except that it works on a primitive
	// variable input state and returns the primitive variable interface
	// state

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

	double smallc = 1.e-10;
	double smallrho = 1.e-10;
	double smallp = 1.e-10;

	double rho_l, un_l, ut_l, p_l;
	double rho_r, un_r, ut_r, p_r;
	double xn[nspec];
	double rhostar_l, rhostar_r;
	double ustar, pstar, cstar_l, cstar_r;
	double lambda_l, lambdastar_l, lambda_r, lambdastar_r;
	double W_l, W_r, c_l, c_r, sigma;
	double alpha;

	double rho_state, un_state, ut_state, p_state;

	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx;
	int jlo = ng;
	int jhi = ng+ny;

	for (int i = ilo-1; i < ihi+1; i++) {
		for (int j = jlo-1; j < jhi+1; j++) {
			int idx = (i*qy + j)*nvar;

			// primitive variable states
			rho_l  = q_l[idx+irho];

			// un = normal velocity; ut = transverse velocity
			if (idir == 1) {
				un_l    = q_l[idx+iu];
				ut_l    = q_l[idx+iv];
			} else {
				un_l    = q_l[idx+iv];
				ut_l    = q_l[idx+iu];
			}


			p_l   = q_l[idx+ip];
			p_l = fmax(p_l, smallp);

			rho_r  = q_r[idx+irho];

			if (idir == 1) {
				un_r    = q_r[idx+iu];
				ut_r    = q_r[idx+iv];
			} else {
				un_r    = q_r[idx+iv];
				ut_r    = q_r[idx+iu];
			}

			p_r   = q_r[idx+ip];
			p_r = fmax(p_r, smallp);

			// define the Lagrangian sound speed
			W_l = fmax(smallrho*smallc, sqrt(gamma*p_l*rho_l));
			W_r = fmax(smallrho*smallc, sqrt(gamma*p_r*rho_r));

			// and the regular sound speeds
			c_l = fmax(smallc, sqrt(gamma*p_l/rho_l));
			c_r = fmax(smallc, sqrt(gamma*p_r/rho_r));

			// define the star states
			pstar = (W_l*p_r + W_r*p_l + W_l*W_r*(un_l - un_r))/(W_l + W_r);
			pstar = fmax(pstar, smallp);
			ustar = (W_l*un_l + W_r*un_r + (p_l - p_r))/(W_l + W_r);

			// now compute the remaining state to the left and right
			// of the contact (in the star region)
			rhostar_l = rho_l + (pstar - p_l)/(c_l*c_l);
			rhostar_r = rho_r + (pstar - p_r)/(c_r*c_r);

			cstar_l = fmax(smallc, sqrt(gamma*pstar/rhostar_l));
			cstar_r = fmax(smallc, sqrt(gamma*pstar/rhostar_r));

			// figure out which state we are in, based on the location of
			// the waves
			if (ustar > 0.0) {

				// contact is moving to the right, we need to understand
				// the L and *L states

				// Note: transverse velocity only jumps across contact
				ut_state = ut_l;

				// define eigenvalues
				lambda_l = un_l - c_l;
				lambdastar_l = ustar - cstar_l;

				if (pstar > p_l) {
					// the wave is a shock -- find the shock speed
					sigma = (lambda_l + lambdastar_l)/2.0;

					if (sigma > 0.0) {
						// shock is moving to the right -- solution is L state
						rho_state = rho_l;
						un_state = un_l;
						p_state = p_l;

					} else {
						// solution is *L state
						rho_state = rhostar_l;
						un_state = ustar;
						p_state = pstar;
					}

				} else {
					// the wave is a rarefaction
					if ((lambda_l < 0.0) && (lambdastar_l < 0.0)) {
						// rarefaction fan is moving to the left -- solution is
						// *L state
						rho_state = rhostar_l;
						un_state = ustar;
						p_state = pstar;

					} else if ((lambda_l > 0.0) && (lambdastar_l > 0.0)) {
						// rarefaction fan is moving to the right -- solution is
						// L state
						rho_state = rho_l;
						un_state = un_l;
						p_state = p_l;

					} else {
						// rarefaction spans x/t = 0 -- interpolate
						alpha = lambda_l/(lambda_l - lambdastar_l);

						rho_state  = alpha*rhostar_l  + (1.0 - alpha)*rho_l;
						un_state   = alpha*ustar      + (1.0 - alpha)*un_l;
						p_state    = alpha*pstar      + (1.0 - alpha)*p_l;
					}
				}

			} else if (ustar < 0) {

				// contact moving left, we need to understand the R and *R
				// states

				// Note: transverse velocity only jumps across contact
				ut_state = ut_r;

				// define eigenvalues
				lambda_r = un_r + c_r;
				lambdastar_r = ustar + cstar_r;

				if (pstar > p_r) {
					// the wave if a shock -- find the shock speed
					sigma = (lambda_r + lambdastar_r)/2.0;

					if (sigma > 0.0) {
						// shock is moving to the right -- solution is *R state
						rho_state = rhostar_r;
						un_state = ustar;
						p_state = pstar;

					} else {
						// solution is R state
						rho_state = rho_r;
						un_state = un_r;
						p_state = p_r;
					}

				} else {
					// the wave is a rarefaction
					if ((lambda_r < 0.0) && (lambdastar_r < 0.0)) {
						// rarefaction fan is moving to the left -- solution is
						// R state
						rho_state = rho_r;
						un_state = un_r;
						p_state = p_r;

					} else if ((lambda_r > 0.0) && (lambdastar_r > 0.0)) {
						// rarefaction fan is moving to the right -- solution is
						// *R state
						rho_state = rhostar_r;
						un_state = ustar;
						p_state = pstar;

					} else {
						// rarefaction spans x/t = 0 -- interpolate
						alpha = lambda_r/(lambda_r - lambdastar_r);

						rho_state  = alpha*rhostar_r  + (1.0 - alpha)*rho_r;
						un_state   = alpha*ustar      + (1.0 - alpha)*un_r;
						p_state    = alpha*pstar      + (1.0 - alpha)*p_r;
					}
				}

			} else {  // ustar == 0

				rho_state = 0.5*(rhostar_l + rhostar_r);
				un_state = ustar;
				ut_state = 0.5*(ut_l + ut_r);
				p_state = pstar;
			}

			// species now
			if (nspec > 0) {
				if (ustar > 0.0) {
					for (int n = 0; n < nspec; n++)
						xn[n] = q_l[idx+iX+n];

				} else if (ustar < 0.0) {
					for (int n = 0; n < nspec; n++)
						xn[n] = q_r[idx+iX+n];
				} else {
					for (int n = 0; n < nspec; n++)
						xn[n] = 0.5*(q_l[idx+iX+n] + q_r[idx+iX+n]);
				}
			}

			// are we on a solid boundary?
			if (idir == 1) {
				if ((i == ilo) && (lower_solid == 1))
					un_state = 0.0;

				if ((i == ihi+1) && (upper_solid == 1))
					un_state = 0.0;

			} else if (idir == 2) {
				if ((j == jlo) && (lower_solid == 1) )
					un_state = 0.0;

				if ((j == jhi+1) && (upper_solid == 1) )
					un_state = 0.0;
			}

			q_int[idx+irho] = rho_state;

			if (idir == 1) {
				q_int[idx+iu] = un_state;
				q_int[idx+iv] = ut_state;
			} else {
				q_int[idx+iu] = ut_state;
				q_int[idx+iv] = un_state;
			}

			q_int[idx+ip] = p_state;

			if (nspec > 0) {
				for (int n = 0; n < nspec; n++)
					q_int[idx+iX+n] = xn[n];
			}

		}
	}
}



void riemann_hllc_c(int idir, int qx, int qy,int ng,
                    int nvar, int idens, int ixmom, int iymom,
                    int iener, int irhoX, int nspec,
                    int lower_solid, int upper_solid,
                    double gamma,
                    double *U_l,
                    double *U_r, double *F) {

	// this is the HLLC Riemann solver.  The implementation follows
	// directly out of Toro's book.  Note: this fores not handle the
	// transonic rarefaction.

	double smallc = 1.e-10;
	double smallrho = 1.e-10;
	double smallp = 1.e-10;

	double rho_l, un_l, ut_l, rhoe_l, p_l;
	double rho_r, un_r, ut_r, rhoe_r, p_r;
	double xn[nspec];

	double rhostar_l, rhostar_r, rho_avg;
	double ustar, pstar;
	double Q, p_fmin, p_fmax, p_lr, p_guess;
	double factor, factor2;
	double g_l, g_r, A_l, B_l, A_r, B_r, z;
	double S_l, S_r, S_c;
	double c_l, c_r, c_avg;

	double U_state[nvar];
	double HLLCfactor;

	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx-1;
	int jlo = ng;
	int jhi = ng+ny-1;

	for (int i = ilo-1; i < ihi+1; i++) {
		for (int j = jlo-1; j < jhi+1; j++) {
			int idx = (i*qy + j)*nvar;

			// primitive variable states
			rho_l  = U_l[idx+idens];

			// un = normal velocity; ut = transverse velocity
			if (idir == 1) {
				un_l    = U_l[idx+ixmom]/rho_l;
				ut_l    = U_l[idx+iymom]/rho_l;
			} else {
				un_l    = U_l[idx+iymom]/rho_l;
				ut_l    = U_l[idx+ixmom]/rho_l;
			}

			rhoe_l = U_l[idx+iener] -
			         0.5*rho_l*(un_l*un_l + ut_l*ut_l);

			p_l   = rhoe_l*(gamma - 1.0);
			p_l = fmax(p_l, smallp);

			rho_r  = U_r[idx+idens];

			if (idir == 1) {
				un_r    = U_r[idx+ixmom]/rho_r;
				ut_r    = U_r[idx+iymom]/rho_r;
			} else {
				un_r    = U_r[idx+iymom]/rho_r;
				ut_r    = U_r[idx+ixmom]/rho_r;
			}

			rhoe_r = U_r[idx+iener] -
			         0.5*rho_r*(un_r*un_r + ut_r*ut_r);

			p_r   = rhoe_r*(gamma - 1.0);
			p_r = fmax(p_r, smallp);

			// compute the sound speeds
			c_l = fmax(smallc, sqrt(gamma*p_l/rho_l));
			c_r = fmax(smallc, sqrt(gamma*p_r/rho_r));

			// Estimate the star quantities -- use one of three methods to
			// for this -- the primitive variable Riemann solver, the two
			// shock approximation, or the two rarefaction approximation.
			// Pick the method based on the pressure states at the
			// interface.

			p_fmax = fmax(p_l, p_r);
			p_fmin = fmin(p_l, p_r);

			Q = p_fmax/p_fmin;

			rho_avg = 0.5*(rho_l + rho_r);
			c_avg = 0.5*(c_l + c_r);

			// primitive variable Riemann solver (Toro, 9.3)
			factor = rho_avg*c_avg;
			factor2 = rho_avg/c_avg;

			pstar = 0.5*(p_l + p_r) +
			        0.5*(un_l - un_r)*factor;
			ustar = 0.5*(un_l + un_r) +
			        0.5*(p_l - p_r)/factor;

			rhostar_l = rho_l + (un_l - ustar)*factor2;
			rhostar_r = rho_r + (ustar - un_r)*factor2;

			if ((Q > 2) && ((pstar < p_fmin) || (pstar > p_fmax))) {

				// use a more accurate Riemann solver for the estimate here

				if (pstar < p_fmin) {

					// 2-rarefaction Riemann solver
					z = (gamma - 1.0)/(2.0*gamma);
					p_lr = pow(p_l/p_r,z);

					ustar = (p_lr*un_l/c_l + un_r/c_r +
					         2.0*(p_lr - 1.0)/(gamma - 1.0)) /
					        (p_lr/c_l + 1.0/c_r);

					pstar = 0.5*(p_l*pow(1.0 + (gamma - 1.0)*(un_l - ustar)/
					                     (2.0*c_l), 1.0/z) +
					             p_r*pow(1.0 + (gamma - 1.0)*(ustar - un_r)/
					                     (2.0*c_r), 1.0/z) );

					rhostar_l = rho_l*pow(pstar/p_l, 1.0/gamma);
					rhostar_r = rho_r*pow(pstar/p_r, 1.0/gamma);

				} else {

					// 2-shock Riemann solver
					A_r = 2.0/((gamma + 1.0)*rho_r);
					B_r = p_r*(gamma - 1.0)/(gamma + 1.0);

					A_l = 2.0/((gamma + 1.0)*rho_l);
					B_l = p_l*(gamma - 1.0)/(gamma + 1.0);

					// guess of the pressure
					p_guess = fmax(0.0, pstar);

					g_l = sqrt(A_l / (p_guess + B_l));
					g_r = sqrt(A_r / (p_guess + B_r));

					pstar = (g_l*p_l + g_r*p_r - (un_r - un_l))/(g_l + g_r);

					ustar = 0.5*(un_l + un_r) +
					        0.5*( (pstar - p_r)*g_r - (pstar - p_l)*g_l);

					rhostar_l = rho_l*(pstar/p_l + (gamma-1.0)/(gamma+1.0))/
					            ( (gamma-1.0) / (gamma+1.0)*(pstar/p_l) + 1.0);

					rhostar_r = rho_r*(pstar/p_r +
					                   (gamma-1.0)/(gamma+1.0))/
					            ( (gamma-1.0) / (gamma+1.0)*(pstar/p_r) + 1.0);
				}
			}

			// estimate the nonlinear wave speeds

			if (pstar <= p_l) {
				// rarefaction
				S_l = un_l - c_l;
			} else {
				// shock
				S_l = un_l - c_l *
				      sqrt(1.0 + ((gamma+1.0)/(2.0*gamma))* (pstar/p_l - 1.0));
			}

			if (pstar <= p_r) {
				// rarefaction
				S_r = un_r + c_r;
			} else {
				// shock
				S_r = un_r + c_r*sqrt(1.0 +
				                      ((gamma+1.0)/(2.0/gamma))* (pstar/p_r - 1.0));
			}

			//  We could just take S_c = u_star as the estimate for the
			//  contact speed, but we can actually for this more accurately
			//  by using the Rankine-Hugonoit jump conditions across each
			//  of the waves (see Toro 10.58, Batten et al. SIAM
			//  J. Sci. and Stat. Comp., 18:1553 (1997)
			S_c = (p_r - p_l + rho_l*un_l*(S_l - un_l) -
			       rho_r*un_r*(S_r - un_r))/
			      (rho_l*(S_l - un_l) - rho_r*(S_r - un_r));


			// figure out which region we are in and compute the state and
			// the interface fluxes using the HLLC Riemann solver

			double F_state[nvar];
			double U_s[nvar];
			if (S_r <= 0.0) {
				// R region
				for (int n = 0; n < nvar; n++)
					U_state[n] = U_r[idx+n];

				consFlux(idir, gamma, idens, ixmom, iymom,
				         iener, irhoX, nvar, nspec, U_state, F_state);

				for (int n = 0; n < nvar; n++)
					F[idx+n] = F_state[n];


			} else if ((S_r > 0.0) && (S_c <= 0)) {
				// R* region
				HLLCfactor = rho_r*(S_r - un_r)/(S_r - S_c);

				U_state[idens] = HLLCfactor;

				if (idir == 1) {
					U_state[ixmom] = HLLCfactor*S_c;
					U_state[iymom] = HLLCfactor*ut_r;
				} else {
					U_state[ixmom] = HLLCfactor*ut_r;
					U_state[iymom] = HLLCfactor*S_c;
				}


				U_state[iener] = HLLCfactor *
				                 (U_r[idx+iener]/rho_r +
				                  (S_c - un_r)*(S_c + p_r/(rho_r*(S_r - un_r))));

				// species
				if (nspec > 0) {
					for (int n = 0; n < nspec; n++)
						U_state[irhoX+n] = HLLCfactor *
						                   U_r[idx+irhoX+n]/rho_r;
				}

				for (int n = 0; n < nvar; n++)
					U_s[n] = U_r[idx+n];


				// find the flux on the right interface
				consFlux(idir, gamma, idens, ixmom, iymom,
				         iener, irhoX, nvar, nspec,
				         U_s, F_state);

				// correct the flux
				for (int n = 0; n < nvar; n++)
					F[idx+n] = F_state[n] + S_r*(U_state[n] - U_r[idx+n]);

			} else if (S_c > 0.0 && S_l < 0.0) {
				// L* region
				HLLCfactor = rho_l*(S_l - un_l)/(S_l - S_c);

				U_state[idens] = HLLCfactor;

				if (idir == 1) {
					U_state[ixmom] = HLLCfactor*S_c;
					U_state[iymom] = HLLCfactor*ut_l;
				} else {
					U_state[ixmom] = HLLCfactor*ut_l;
					U_state[iymom] = HLLCfactor*S_c;
				}


				U_state[iener] = HLLCfactor *
				                 (U_l[idx+iener]/rho_l +
				                  (S_c - un_l)*(S_c + p_l/(rho_l*(S_l - un_l))));

				// species
				if (nspec > 0) {
					for (int n = 0; n < nspec; n++)
						U_state[irhoX+n] = HLLCfactor *
						                   U_l[idx+irhoX+n]/rho_l;
				}

				for (int n = 0; n < nvar; n++)
					U_s[n] = U_l[idx+n];


				// find the flux on the left interface
				consFlux(idir, gamma, idens, ixmom, iymom,
				         iener, irhoX, nvar, nspec,
				         U_s, F_state);

				// correct the flux
				for (int n = 0; n < nvar; n++)
					F[idx+n] = F_state[n] +
					           S_l*(U_state[n] - U_l[idx+n]);

			} else {
				// L region
				for (int n = 0; n < nvar; n++)
					U_state[n] = U_l[idx+n];

				consFlux(idir, gamma, idens, ixmom, iymom,
				         iener, irhoX, nvar, nspec,
				         U_state, F_state);
				for (int n = 0; n < nvar; n++)
					F[idx+n] = F_state[n];
			}
		}
	}
}

// we should deal with solid boundaries somehow here

void consFlux(int idir,
              double gamma, int idens, int ixmom, int iymom,
              int iener, int irhoX, int nvar, int nspec,
              double *U_state, double *F) {

	double u = U_state[ixmom]/U_state[idens];
	double v = U_state[iymom]/U_state[idens];

	double p = (U_state[iener] - 0.5*U_state[idens]*(u*u + v*v))*(gamma - 1.0);

	if (idir == 1) {
		F[idens] = U_state[idens]*u;
		F[ixmom] = U_state[ixmom]*u + p;
		F[iymom] = U_state[iymom]*u;
		F[iener] = (U_state[iener] + p)*u;
		if (nspec > 0) {
			for (int n = 0; n < nspec; n++)
				F[irhoX+n] = U_state[irhoX+n]*u;
		}

	} else {
		F[idens] = U_state[idens]*v;
		F[ixmom] = U_state[ixmom]*v;
		F[iymom] = U_state[iymom]*v + p;
		F[iener] = (U_state[iener] + p)*v;
		if (nspec > 0) {
			for (int n = 0; n < nspec; n++)
				F[irhoX+n] = U_state[irhoX+n]*v;
		}
	}
}


void artificial_viscosity_c(int qx, int qy, int ng,
                            double dx, double dy,
                            double cvisc,
                            double *u,
                            double *v,
                            double *avisco_x, double *avisco_y) {

	// compute the artifical viscosity.  Here, we compute edge-centered
	// approximations to the divergence of the velocity.  This follows
	// directly Colella  Woodward (1984) Eq. 4.5
	//
	// data locations:
	//
	//   j+3/2--+---------+---------+---------+
	//          |         |         |         |
	//     j+1  +         |         |         |
	//          |         |         |         |
	//   j+1/2--+---------+---------+---------+
	//          |         |         |         |
	//        j +         X         |         |
	//          |         |         |         |
	//   j-1/2--+---------+----Y----+---------+
	//          |         |         |         |
	//      j-1 +         |         |         |
	//          |         |         |         |
	//   j-3/2--+---------+---------+---------+
	//          |    |    |    |    |    |    |
	//              i-1        i        i+1
	//        i-3/2     i-1/2     i+1/2     i+3/2
	//
	// X is the location of avisco_x[i, j]
	// Y is the location of avisco_y[i, j]

	double divU_x, divU_y;

	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx;
	int jlo = ng;
	int jhi = ng+ny;

	for (int i = ilo-1; i < ihi+1; i++) {
		for (int j = jlo-1; j < jhi+1; j++) {
			int idx = i*qy + j;

			// start by computing the divergence on the x-interface.  The
			// x-difference is simply the difference of the cell-centered
			// x-velocities on either side of the x-interface.  For the
			// y-difference, first average the four cells to the node on
			// each end of the edge, and: difference these to find the
			// edge centered y difference.
			divU_x = (u[idx] - u[(i-1)*qy+j])/dx +
			         0.25*(v[idx+1] + v[(i-1)*qy+j+1] -
			               v[idx-1] - v[(i-1)*qy-1])/dy;

			avisco_x[idx] = cvisc*fmax(-divU_x*dx, 0.0);

			// now the y-interface value
			divU_y = 0.25*(u[(i+1)*qy+j] +
			               u[(i+1)*qy+j-1] - u[(i-1)*qy+j] -
			               u[(i-1)*qy+j-1])/dx +
			         (v[idx] - v[idx-1])/dy;

			avisco_y[idx] = cvisc*fmax(-divU_y*dy, 0.0);

		}
	}
}
