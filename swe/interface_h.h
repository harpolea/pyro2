void states_c(int idir, int qx, int qy, int ng,
            double dx, double dt,
            int ih, int iu, int iv, int ix, int nvar, int nspec,
            double g,
            double *qv, double *dqv,
            double *q_l, double *q_r);

void riemann_Roe_c(int idir, int qx, int qy, int ng,
                 int nvar, int ih, int ixmom, int iymom,
                 int ihX, int nspec,
                 int lower_solid, int upper_solid,
                 double g, double *U_l, double *U_r, double *F);

void riemann_HLLC_c(int idir, int qx, int qy, int ng,
                  int nvar, int ih, int ixmom, int iymom,
                  int ihX, int nspec,
                  int lower_solid, int upper_solid,
                  double g, double *U_l, double *U_r,
                  double *F);

void consFlux(int idir, double g, int ih, int ixmom,
              int iymom, int ihX, int nvar, int nspec,
              double *U_state, double *F);
