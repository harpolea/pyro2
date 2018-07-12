#include <math.h>
#include "interface_states_h.h"

void states_c(double *a, int qx, int qy, int ng, int idir,
              double *al, double *ar) {

	double a_int[qx*qy];
	double dafm[qx*qy];
	double dafp[qx*qy];
	double d2af[qx*qy];
	double d2ac[qx*qy];
	double d3a[qx*qy];

	double C2 = 1.25;
	double C3 = 0.1;

	double rho, s;

	double d2a_lim, d3a_fmin, d3a_fmax;

	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx;
	int jlo = ng;
	int jhi = ng+ny;

	// our convention here is that:
	//     al[idx]   will be al_{i-1/2,j),
	//     al[(i+1)*qy+j] will be al_{i+1/2,j)

	// we need interface values on all faces of the domain
	if (idir == 1)  {

		for (int i = ilo-2; i < ihi+3; i++) {
			for (int j = jlo-1; j < jhi+1; j++) {
				int idx = i*qy+j;

				// interpolate to the edges
				a_int[idx] = (7.0/12.0)*(a[(i-1)*qy+j] + a[idx]) -
				             (1.0/12.0)*(a[(i-2)*qy+j] + a[(i+1)*qy+j]);

				al[idx] = a_int[idx];
				ar[idx] = a_int[idx];

			}
		}

		for (int i = ilo-2; i < ihi+3; i++) {
			for (int j = jlo-1; j < jhi+1; j++) {
				int idx = i*qy+j;
				// these live on cell-centers
				dafm[idx] = a[idx] - a_int[idx];
				dafp[idx] = a_int[(i+1)*qy+j] - a[idx];

				// these live on cell-centers
				d2af[idx] = 6.0*(a_int[idx] - 2.0*a[idx] + a_int[(i+1)*qy+j]);
			}
		}

		for (int i = ilo-3; i < ihi+3; i++) {
			for (int j = jlo-1; j < jhi+1; j++) {
				int idx = i*qy+j;
				d2ac[idx] = a[(i-1)*qy+j] - 2.0*a[idx] + a[(i+1)*qy+j];
			}
		}

		for (int i = ilo-2; i < ihi+3; i++) {
			for (int j = jlo-1; j < jhi+1; j++) {
				int idx = i*qy+j;
				// this lives on the interface
				d3a[idx] = d2ac[idx] - d2ac[(i-1)*qy+j];
			}
		}

		// this is a look over cell centers, affecting
		// i-1/2,R and i+1/2,L
		for (int i = ilo-1; i < ihi+1; i++) {
			for (int j = jlo-1; j < jhi+1; j++) {
				int idx = i*qy+j;

				// limit? MC Eq. 24 and 25
				if (dafm[idx] * dafp[idx] <= 0.0 ||
				    (a[idx] - a[(i-2)*qy+j])*(a[(i+2)*qy+j] - a[idx]) <= 0.0)  {

					// we are at an extrema

					s = copysign(1.0, d2ac[idx]);
					if (s == copysign(1.0, d2ac[(i-1)*qy+j]) &&
					    s == copysign(1.0, d2ac[(i+1)*qy+j]) &&
					    s == copysign(1.0, d2af[idx]))  {
						// MC Eq. 26
						d2a_lim = s*fmin(fmin(fabs(d2af[idx]),
						                      C2*fabs(d2ac[(i-1)*qy+j])),
						                 fmin(C2*fabs(d2ac[idx]),
						                      C2*fabs(d2ac[(i+1)*qy+j])));
					} else {
						d2a_lim = 0.0;
					}

					if (fabs(d2af[idx]) <= 1.e-12 *
					    fmax(fmax(fabs(a[(i-2)*qy+j]),
					              fabs(a[(i-1)*qy+j])),
					         fmax(fmax(fabs(a[idx]), fabs(a[(i+1)*qy+j])),
					              fabs(a[(i+2)*qy+j]))))  {
						rho = 0.0;
					} else {
						// MC Eq. 27
						rho = d2a_lim/d2af[idx];
					}

					if (rho < 1.0 - 1.e-12)  {
						// we may need to limit -- these quantities are at cell-centers
						d3a_fmin = fmin(fmin(d3a[(i-1)*qy+j],
						                     d3a[idx]),
						                fmin(d3a[(i+1)*qy+j], d3a[(i+2)*qy+j]));
						d3a_fmax = fmax(fmax(d3a[(i-1)*qy+j], d3a[idx]),
						                fmax(d3a[(i+1)*qy+j], d3a[(i+2)*qy+j]));

						if (C3*fmax(fabs(d3a_fmin),
						            fabs(d3a_fmax)) <= (d3a_fmax - d3a_fmin))  {
							// limit
							if (dafm[idx]*dafp[idx] < 0.0)  {
								// Eqs. 29, 30
								ar[idx] = a[idx] - rho*dafm[idx]; // note: typo in Eq 29
								al[(i+1)*qy+j] = a[idx] + rho*dafp[idx];
							} else if (fabs(dafm[idx]) >= 2.0*fabs(dafp[idx]))  {
								// Eq. 31
								ar[idx] = a[idx] -
								          2.0*(1.0 - rho)*dafp[idx] - rho*dafm[idx];
							} else if (fabs(dafp[idx]) >= 2.0*fabs(dafm[idx]))  {
								// Eq. 32
								al[(i+1)*qy+j] = a[idx] +
								                 2.0*(1.0 - rho)*dafm[idx] +
								                 rho*dafp[idx];
							}

						}
					}

				} else {
					// if Eqs. 24 or 25 didn't hold we still may need to limit
					if (fabs(dafm[idx]) >= 2.0*fabs(dafp[idx]))  {
						ar[idx] = a[idx] - 2.0*dafp[idx];
					}
					if (fabs(dafp[idx]) >= 2.0*fabs(dafm[idx]))  {
						al[(i+1)*qy+j] = a[idx] + 2.0*dafm[idx];
					}
				}

			}
		}

	} else if (idir == 2)  {

		for (int i = ilo-1; i < ihi+1; i++) {
			for (int j = jlo-2; j < jhi+3; j++) {
				int idx = i*qy+j;

				// interpolate to the edges
				a_int[idx] = (7.0/12.0)*(a[idx-1] + a[idx]) -
				             (1.0/12.0)*(a[idx-2] + a[idx+1]);

				al[idx] = a_int[idx];
				ar[idx] = a_int[idx];

			}
		}

		for (int i = ilo-1; i < ihi+1; i++) {
			for (int j = jlo-2; j < jhi+3; j++) {
				int idx = i*qy+j;
				// these live on cell-centers
				dafm[idx] = a[idx] - a_int[idx];
				dafp[idx] = a_int[idx+1] - a[idx];

				// these live on cell-centers
				d2af[idx] = 6.0*(a_int[idx] - 2.0*a[idx] + a_int[idx+1]);
			}
		}

		for (int i = ilo-1; i < ihi+1; i++) {
			for (int j = jlo-3; j < jhi+3; j++) {
				int idx = i*qy+j;
				d2ac[idx] = a[idx-1] - 2.0*a[idx] + a[idx+1];
			}
		}

		for (int j = jlo-2; j < jhi+3; j++) {
			for (int i = ilo-1; i < ihi+1; i++) {
				int idx = i*qy+j;
				// this lives on the interface
				d3a[idx] = d2ac[idx] - d2ac[idx-1];
			}
		}

		// this is a look over cell centers, affecting
		// j-1/2,R and j+1/2,L
		for (int i = ilo-1; i < ihi+1; i++) {
			for (int j = jlo-1; j < jhi+1; j++) {
				int idx = i*qy+j;

				// limit? MC Eq. 24 and 25
				if (dafm[idx] * dafp[idx] <= 0.0 ||
				    (a[idx] - a[idx-2])*(a[idx+2] - a[idx]) <= 0.0)  {

					// we are at an extrema

					s = copysign(1.0, d2ac[idx]);
					if (s == copysign(1.0, d2ac[idx-1]) &&
					    s == copysign(1.0, d2ac[idx+1]) &&
					    s == copysign(1.0, d2af[idx]))  {
						// MC Eq. 26
						d2a_lim = s*fmin(fmin(fabs(d2af[idx]),
						                      C2*fabs(d2ac[idx-1])),
						                 fmin(C2*fabs(d2ac[idx]),
						                      C2*fabs(d2ac[idx+1])));
					} else {
						d2a_lim = 0.0;
					}

					if (fabs(d2af[idx]) <=
					    1.e-12*fmax(fmax(fabs(a[idx-2]),
					                     fabs(a[idx-1])),
					                fmax(fmax(fabs(a[idx]),
					                          fabs(a[idx+1])),
					                     fabs(a[idx+2]))))  {
						rho = 0.0;
					} else {
						// MC Eq. 27
						rho = d2a_lim/d2af[idx];
					}

					if (rho < 1.0 - 1.e-12)  {
						// we may need to limit -- these quantities are at cell-centers
						d3a_fmin = fmin(fmin(d3a[idx-1], d3a[idx]),
						                fmin(d3a[idx+1], d3a[idx+2]));
						d3a_fmax = fmax(fmax(d3a[idx-1], d3a[idx]),
						                fmax(d3a[idx+1], d3a[idx+2]));

						if (C3*fmax(fabs(d3a_fmin),
						            fabs(d3a_fmax)) <= (d3a_fmax - d3a_fmin))  {
							// limit
							if (dafm[idx]*dafp[idx] < 0.0)  {
								// Eqs. 29, 30
								ar[idx] = a[idx] - rho*dafm[idx]; // note: typo in Eq 29
								al[idx+1] = a[idx] + rho*dafp[idx];
							} else if (fabs(dafm[idx]) >= 2.0*fabs(dafp[idx]))  {
								// Eq. 31
								ar[idx] = a[idx] - 2.0*(1.0 - rho)*dafp[idx] -
								          rho*dafm[idx];
							} else if (fabs(dafp[idx]) >= 2.0*fabs(dafm[idx]))  {
								// Eq. 32
								al[idx+1] = a[idx] + 2.0*(1.0 - rho)*dafm[idx] +
								            rho*dafp[idx];
							}

						}
					}

				} else {
					// if Eqs. 24 or 25 didn't hold we still may need to limit
					if (fabs(dafm[idx]) >= 2.0*fabs(dafp[idx]))  {
						ar[idx] = a[idx] - 2.0*dafp[idx];
					}
					if (fabs(dafp[idx]) >= 2.0*fabs(dafm[idx]))  {
						al[idx+1] = a[idx] + 2.0*dafm[idx];
					}
				}

			}
		}

	}

}


void states_nolimit_c(double *a, int qx, int qy,
                      int ng, int idir,
                      double *al, double *ar) {

	double a_int[qx*qy];;


	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx;
	int jlo = ng;
	int jhi = ng+ny;

	// our convention here is that:
	//     al[idx]   will be al_{i-1/2,j),
	//     al[(i+1)*qy+j] will be al_{i+1/2,j)

	// we need interface values on all faces of the domain
	if (idir == 1)  {

		for (int i = ilo-2; i < ihi+3; i++) {
			for (int j = jlo-1; j < jhi+1; j++) {
				int idx = i*qy+j;

				// interpolate to the edges
				a_int[idx] = (7.0/12.0)*
				             (a[(i-1)*qy+j] + a[idx]) -
				             (1.0/12.0)*(a[(i-2)*qy+j] + a[(i+1)*qy+j]);

				al[idx] = a_int[idx];
				ar[idx] = a_int[idx];

			}
		}

	} else if (idir == 2)  {

		for (int i = ilo-1; i < ihi+1; i++) {
			for (int j = jlo-2; j < jhi+3; j++) {
				int idx = i*qy+j;

				// interpolate to the edges
				a_int[idx] = (7.0/12.0)*(a[idx-1] + a[idx]) -
				             (1.0/12.0)*(a[idx-2] + a[idx+1]);

				al[idx] = a_int[idx];
				ar[idx] = a_int[idx];

			}
		}
	}

}
