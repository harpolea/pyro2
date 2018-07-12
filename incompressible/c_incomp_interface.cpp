#include "incomp_interface_h.h"

void mac_vels_c(int qx, int qy, int ng, double dx,
                double dy, double dt,
                double *u, double *v,
                double *ldelta_ux, double *ldelta_vx,
                double *ldelta_uy, double *ldelta_vy,
                double *gradp_x, double *gradp_y,
                double *u_MAC, double *v_MAC) {

	double u_xl[qx*qy];
	double u_xr[qx*qy];
	double u_yl[qx*qy];
	double u_yr[qx*qy];
	double v_xl[qx*qy];
	double v_xr[qx*qy];
	double v_yl[qx*qy];
	double v_yr[qx*qy];

	for (int i = 0; i < qx*qy; i++) {
		u_xl[i] = 0;
		u_xr[i] = 0;
		u_yl[i] = 0;
		u_yr[i] = 0;
		v_xl[i] = 0;
		v_xr[i] = 0;
		v_yl[i] = 0;
		v_yr[i] = 0;
	}

	// get the full u and v left and right states (including transverse
	// terms) on both the x- and y-interfaces
	get_interface_states(qx, qy, ng, dx, dy, dt,
	                     u, v,
	                     ldelta_ux, ldelta_vx,
	                     ldelta_uy, ldelta_vy,
	                     gradp_x, gradp_y,
	                     u_xl, u_xr, u_yl, u_yr,
	                     v_xl, v_xr, v_yl, v_yr);

	// Riemann problem -- this follows Burger's equation.  We don't use
	// any input velocity for the upwinding.  Also, we only care about
	// the normal states here (u on x and v on y)
	riemann_and_upwind(qx, qy, ng, u_xl, u_xr, u_MAC);
	riemann_and_upwind(qx, qy, ng, v_yl, v_yr, v_MAC);

}


//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
void states_c(int qx, int qy, int ng, double dx,
              double dy, double dt,
              double *u, double *v,
              double *ldelta_ux, double *ldelta_vx,
              double *ldelta_uy, double *ldelta_vy,
              double *gradp_x, double *gradp_y,
              double *u_MAC, double *v_MAC,
              double *u_xint, double *v_xint,
              double *u_yint, double *v_yint) {

	// this is similar to mac_vels, but it predicts the interface states
	// of both u and v on both interfaces, using the MAC velocities to
	// do the upwinding.

	double u_xl[qx*qy];
	double u_xr[qx*qy];
	double u_yl[qx*qy];
	double u_yr[qx*qy];
	double v_xl[qx*qy];
	double v_xr[qx*qy];
	double v_yl[qx*qy];
	double v_yr[qx*qy];

	for (int i = 0; i < qx*qy; i++) {
		u_xl[i] = 0;
		u_xr[i] = 0;
		u_yl[i] = 0;
		u_yr[i] = 0;
		v_xl[i] = 0;
		v_xr[i] = 0;
		v_yl[i] = 0;
		v_yr[i] = 0;
	}

	// get the full u and v left and right states (including transverse
	// terms) on both the x- and y-interfaces
	get_interface_states(qx, qy, ng, dx, dy, dt,
	                     u, v,
	                     ldelta_ux, ldelta_vx,
	                     ldelta_uy, ldelta_vy,
	                     gradp_x, gradp_y,
	                     u_xl, u_xr, u_yl, u_yr,
	                     v_xl, v_xr, v_yl, v_yr);


	// upwind using the MAC velocity to determine which state exists on
	// the interface
	upwind(qx, qy, ng, u_xl, u_xr, u_MAC, u_xint);
	upwind(qx, qy, ng, v_xl, v_xr, u_MAC, v_xint);
	upwind(qx, qy, ng, u_yl, u_yr, v_MAC, u_yint);
	upwind(qx, qy, ng, v_yl, v_yr, v_MAC, v_yint);

}


//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
void get_interface_states(int qx, int qy, int ng,
                          double dx, double dy, double dt,
                          double *u, double *v,
                          double *ldelta_ux, double *ldelta_vx,
                          double *ldelta_uy, double *ldelta_vy,
                          double *gradp_x, double *gradp_y,
                          double *u_xl, double *u_xr,
                          double *u_yl, double *u_yr,
                          double *v_xl, double *v_xr,
                          double *v_yl, double *v_yr) {

	// Compute the unsplit predictions of u and v on both the x- and
	// y-interfaces.  This includes the transverse terms.

	double uhat_adv[qx*qy];
	double vhat_adv[qx*qy];

	double u_xint[qx*qy];
	double u_yint[qx*qy];
	double v_xint[qx*qy];
	double v_yint[qx*qy];

	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx;
	int jlo = ng;
	int jhi = ng+ny;

	// first predict u and v to both interfaces, considering only the normal
	// part of the predictor.  These are the 'hat' states.

	for (int i = 0; i < qx*qy; i++) {
		uhat_adv[i] = 0;
		vhat_adv[i] = 0;
		u_xint[i] = 0;
		u_yint[i] = 0;
		v_xint[i] = 0;
		v_yint[i] = 0;
	}

	double dtdx = dt/dx;
	double dtdy = dt/dy;

	for (int i = ilo-2; i < ihi+2; i++) {
		for (int j = jlo-2; j < jhi+2; j++) {
			int idx = i*qy+j;
			int idx_ip = (i+1)*qy+j;

			// u on x-edges
			u_xl[idx_ip] = u[idx] +
			               0.5*(1.0 - dtdx*u[idx]) * ldelta_ux[idx];
			u_xr[idx] = u[idx] -
			            0.5*(1.0 + dtdx*u[idx])*ldelta_ux[idx];

			// v on x-edges
			v_xl[idx_ip] = v[idx] +
			               0.5*(1.0 - dtdx*u[idx]) * ldelta_vx[idx];
			v_xr[idx] = v[idx] -
			            0.5*(1.0 + dtdx*u[idx])*ldelta_vx[idx];

			// u on y-edges
			u_yl[idx+1] = u[idx] +
			              0.5*(1.0 - dtdy*v[idx])*ldelta_uy[idx];
			u_yr[idx] = u[idx] -
			            0.5*(1.0 + dtdy*v[idx])*ldelta_uy[idx];

			// v on y-edges
			v_yl[idx+1] = v[idx] +
			              0.5*(1.0 - dtdy*v[idx])*ldelta_vy[idx];
			v_yr[idx] = v[idx] -
			            0.5*(1.0 + dtdy*v[idx])*ldelta_vy[idx];

		}
	}


	// now get the normal advective velocities on the interfaces by solving
	// the Riemann problem.
	riemann(qx, qy, ng, u_xl, u_xr, uhat_adv);
	riemann(qx, qy, ng, v_yl, v_yr, vhat_adv);

	// now that we have the advective velocities, upwind the left and right
	// states using the appropriate advective velocity.

	// on the x-interfaces, we upwind based on uhat_adv
	upwind(qx, qy, ng, u_xl, u_xr, uhat_adv, u_xint);
	upwind(qx, qy, ng, v_xl, v_xr, uhat_adv, v_xint);

	// on the y-interfaces, we upwind based on vhat_adv
	upwind(qx, qy, ng, u_yl, u_yr, vhat_adv, u_yint);
	upwind(qx, qy, ng, v_yl, v_yr, vhat_adv, v_yint);

	// at this point, these states are the `hat' states -- they only
	// considered the normal to the interface portion of the predictor.


	// add the transverse flux differences to the preliminary interface states
	for (int i = ilo-2; i < ihi+2; i++) {
		for (int j = jlo-2; j < jhi+2; j++) {
			int idx = i*qy+j;
			int idx_ip = (i+1)*qy+j;

			double ubar = 0.5*(uhat_adv[idx] + uhat_adv[idx_ip]);
			double vbar = 0.5*(vhat_adv[idx] + vhat_adv[idx+1]);

			// v du/dy is the transerse term for the u states on x-interfaces
			double vu_y = vbar*(u_yint[idx+1] - u_yint[idx]);

			u_xl[idx_ip] = u_xl[idx_ip] -
			               0.5*dtdy*vu_y - 0.5*dt*gradp_x[idx];
			u_xr[idx] = u_xr[idx] - 0.5*dtdy*vu_y -
			            0.5*dt*gradp_x[idx];

			// v dv/dy is the transverse term for the v states on x-interfaces
			double vv_y = vbar*(v_yint[idx+1] - v_yint[idx]);

			v_xl[idx_ip] = v_xl[idx_ip] -
			               0.5*dtdy*vv_y - 0.5*dt*gradp_y[idx];
			v_xr[idx] = v_xr[idx] - 0.5*dtdy*vv_y -
			            0.5*dt*gradp_y[idx];

			// u dv/dx is the transverse term for the v states on y-interfaces
			double uv_x = ubar*(v_xint[(i+1)*qy+j] - v_xint[idx]);

			v_yl[idx+1] = v_yl[idx+1] - 0.5*dtdx*uv_x -
			              0.5*dt*gradp_y[idx];
			v_yr[idx] = v_yr[idx] - 0.5*dtdx*uv_x - 0.5*dt*gradp_y[idx];

			// u du/dx is the transverse term for the u states on y-interfaces
			double uu_x = ubar*(u_xint[(i+1)*qy+j] - u_xint[idx]);

			u_yl[idx+1] = u_yl[idx+1] - 0.5*dtdx*uu_x -
			              0.5*dt*gradp_x[idx];
			u_yr[idx] = u_yr[idx] - 0.5*dtdx*uu_x - 0.5*dt*gradp_x[idx];

		}
	}

}


//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
void upwind(int qx, int qy, int ng, double *q_l, double *q_r,
            double *s, double *q_int) {

	// upwind the left and right states based on the specified input
	// velocity, s.  The resulting interface state is q_int

	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx;
	int jlo = ng;
	int jhi = ng+ny;

	for (int i = ilo-1; i < ihi+1; i++) {
		for (int j = jlo-1; j < jhi+1; j++) {
			int idx = i*qy+j;

			if (s[idx] > 0.0) {
				q_int[idx] = q_l[idx];
			} else if (s[idx] == 0.0) {
				q_int[idx] = 0.5*(q_l[idx] + q_r[idx]);
			} else {
				q_int[idx] = q_r[idx];
			}
		}
	}
}


//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
void riemann(int qx, int qy, int ng, double *q_l, double *q_r, double *s) {

	// Solve the Burger's Riemann problem given the input left and right
	// states and return the state on the interface.
	//
	// This uses the expressions from Almgren, Bell, and Szymczak 1996.

	int nx = qx - 2*ng;
	int ny = qy - 2*ng;
	int ilo = ng;
	int ihi = ng+nx;
	int jlo = ng;
	int jhi = ng+ny;

	for (int i = ilo-1; i < ihi+1; i++) {
		for (int j = jlo-1; j < jhi+1; j++) {
			int idx = i*qy+j;

			if ((q_l[idx] > 0.0) && ((q_l[idx] + q_r[idx]) > 0.0)) {
				s[idx] = q_l[idx];
			} else if ((q_l[idx] <= 0.0) && (q_r[idx] >= 0.0)) {
				s[idx] = 0.0;
			} else {
				s[idx] = q_r[idx];
			}

		}
	}

}

//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
void riemann_and_upwind(int qx, int qy, int ng,
                        double *q_l, double *q_r, double *q_int) {

	// First solve the Riemann problem given q_l and q_r to give the
	// velocity on the interface and { use this velocity to upwind to
	// determine the state (q_l, q_r, or a mix) on the interface).
	//
	// This differs from upwind, above, in that we don't take in a
	// velocity to upwind with).

	double s[qx*qy];

	for (int i = 0; i < qx*qy; i++)
		s[i] = 0;

	riemann(qx, qy, ng, q_l, q_r, s);
	upwind(qx, qy, ng, q_l, q_r, s, q_int);

}
