void mac_vels_c(int qx, int qy, int ng, double dx,
                double dy, double dt,
                double *u, double *v,
                double *ldelta_ux, double *ldelta_vx,
                double *ldelta_uy, double *ldelta_vy,
                double *gradp_x, double *gradp_y,
                double *u_MAC, double *v_MAC);

void states_c(int qx, int qy, int ng, double dx,
              double dy, double dt,
              double *u, double *v,
              double *ldelta_ux, double *ldelta_vx,
              double *ldelta_uy, double *ldelta_vy,
              double *gradp_x, double *gradp_y,
              double *u_MAC, double *v_MAC,
              double *u_xint, double *v_xint,
              double *u_yint, double *v_yint);


void get_interface_states(int qx, int qy, int ng,
                          double dx, double dy, double dt,
                          double *u, double *v,
                          double *ldelta_ux, double *ldelta_vx,
                          double *ldelta_uy, double *ldelta_vy,
                          double *gradp_x, double *gradp_y,
                          double *u_xl, double *u_xr,
                          double *u_yl, double *u_yr,
                          double *v_xl, double *v_xr,
                          double *v_yl, double *v_yr);

void upwind(int qx, int qy, int ng, double *q_l, double *q_r,
            double *s, double *q_int);


void riemann(int qx, int qy, int ng, double *q_l, double *q_r,
             double *s);

void riemann_and_upwind(int qx, int qy, int ng, double *q_l,
                        double *q_r, double *q_int);
