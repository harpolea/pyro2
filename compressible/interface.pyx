import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float_t DTYPE_t

def states(int idir, int qx, int qy, int ng, float dx,
           float dt, int irho, int iu, int iv, int ip, int ix, int nvar, int nspec, float gamma,
           np.ndarray[DTYPE_t, ndim=3] qv, np.ndarray[DTYPE_t, ndim=3] dqv):
    cdef np.ndarray q_l = np.zeros([qx, qy, nvar], dtype=DTYPE)
    cdef np.ndarray q_r = np.zeros([qx, qy, nvar], dtype=DTYPE)
    cdef int ilo, ihi, jlo, jhi
    cdef int nx, ny
    cdef int i, j, n, m

    cdef np.ndarray dq = np.zeros(nvar, dtype=DTYPE)
    cdef np.ndarray q = np.zeros(nvar, dtype=DTYPE)
    cdef np.ndarray lvec = np.zeros([nvar, nvar], dtype=DTYPE)
    cdef np.ndarray rvec = np.zeros([nvar, nvar], dtype=DTYPE)
    cdef np.ndarray eval = np.zeros(nvar, dtype=DTYPE)
    cdef np.ndarray betal = np.zeros(nvar, dtype=DTYPE)
    cdef np.ndarray betar = np.zeros(nvar, dtype=DTYPE)

    cdef float dtdx, dtdx4, cs
    cdef float sum, sum_l, sum_r, factor
    cdef int ns

    nx = qx - 2*ng
    ny = qy - 2*ng
    ilo = ng
    ihi = ng+nx
    jlo = ng
    jhi = ng+ny

    ns = nvar - nspec

    dtdx = dt/dx
    dtdx4 = 0.25*dtdx

    # this is the loop over zones.  For zone i, we see q_l[i+1] and q_r[i]
    for j in range(jlo-2, jhi+2):
       for i in range(ilo-2, ihi+2):

            dq[:] = dqv[i,j,:]
            q[:] = qv[i,j,:]

            cs = np.sqrt(gamma*q[ip]/q[irho])

            lvec[:,:] = 0.0
            rvec[:,:] = 0.0
            eval[:] = 0.0

            # compute the eigenvalues and eigenvectors
            if idir == 1:
                 eval[:] = [q[iu] - cs, q[iu], q[iu], q[iu] + cs]

                 lvec[0,:ns] = [ 0.0, -0.5*q[irho]/cs, 0.0, 0.5/(cs*cs)  ]
                 lvec[1,:ns] = [ 1.0, 0.0,             0.0, -1.0/(cs*cs) ]
                 lvec[2,:ns] = [ 0.0, 0.0,             1.0, 0.0          ]
                 lvec[3,:ns] = [ 0.0, 0.5*q[irho]/cs,  0.0, 0.5/(cs*cs)  ]

                 rvec[0,:ns] = [1.0, -cs/q[irho], 0.0, cs*cs ]
                 rvec[1,:ns] = [1.0, 0.0,       0.0, 0.0 ]
                 rvec[2,:ns] = [0.0, 0.0,       1.0, 0.0 ]
                 rvec[3,:ns] = [1.0, cs/q[irho],  0.0, cs*cs ]

                 # now the species -- they only have a 1 in their corresponding slot
                 eval[ns:] = q[iu]
                 for n in range(ix, ix+nspec):
                    lvec[n,n] = 1.0
                    rvec[n,n] = 1.0

            else:
                eval[:] = [q[iv] - cs, q[iv], q[iv], q[iv] + cs]

                lvec[0,:ns] = [ 0.0, 0.0, -0.5*q[irho]/cs, 0.5/(cs*cs)  ]
                lvec[1,:ns] = [ 1.0, 0.0, 0.0,             -1.0/(cs*cs) ]
                lvec[2,:ns] = [ 0.0, 1.0, 0.0,             0.0          ]
                lvec[3,:ns] = [ 0.0, 0.0, 0.5*q[irho]/cs,  0.5/(cs*cs)  ]

                rvec[0,:ns] = [1.0, 0.0, -cs/q[irho], cs*cs ]
                rvec[1,:ns] = [1.0, 0.0, 0.0,       0.0 ]
                rvec[2,:ns] = [0.0, 1.0, 0.0,       0.0 ]
                rvec[3,:ns] = [1.0, 0.0, cs/q[irho],  cs*cs ]

                # now the species -- they only have a 1 in their corresponding slot
                eval[ns:] = q[iv]
                for n in range(ix, ix+nspec):
                    lvec[n,n] = 1.0
                    rvec[n,n] = 1.0

            # define the reference states
            if idir == 1:
                # this is one the right face of the current zone,
                # so the fastest moving eigenvalue is eval[3] = u + c
                factor = 0.5*(1.0 - dtdx*max(eval[3], 0.0))
                q_l[i+1,j,:] = q[:] + factor*dq[:]

                # left face of the current zone, so the fastest moving
                # eigenvalue is eval[3] = u - c
                factor = 0.5*(1.0 + dtdx*min(eval[0], 0.0))
                q_r[i,  j,:] = q[:] - factor*dq[:]

            else:

                factor = 0.5*(1.0 - dtdx*max(eval[3], 0.0))
                q_l[i,j+1,:] = q[:] + factor*dq[:]

                factor = 0.5*(1.0 + dtdx*min(eval[0], 0.0))
                q_r[i,j,  :] = q[:] - factor*dq[:]


            # compute the Vhat functions
            for m in range(nvar):
                sum = np.dot(lvec[m,:],dq[:])

                betal[m] = dtdx4*(eval[3] - eval[m])*(np.sign(eval[m]) + 1.0)*sum
                betar[m] = dtdx4*(eval[0] - eval[m])*(1.0 - np.sign(eval[m]))*sum

            # construct the states
            for m in range(nvar):
                sum_l = np.dot(betal[:],rvec[:,m])
                sum_r = np.dot(betar[:],rvec[:,m])

                if idir == 1:
                    q_l[i+1,j,m] = q_l[i+1,j,m] + sum_l
                    q_r[i,  j,m] = q_r[i,  j,m] + sum_r
                else:
                    q_l[i,j+1,m] = q_l[i,j+1,m] + sum_l
                    q_r[i,j,  m] = q_r[i,j,  m] + sum_r

    return q_l, q_r


def riemann_cgf(int idir, int qx, int qy, int ng,
                int nvar, int idens, int ixmom, int iymom,
                int iener, int irhoX, int nspec,
                int lower_solid, int upper_solid,
                float gamma, np.ndarray[DTYPE_t, ndim=3] U_l, np.ndarray[DTYPE_t, ndim=3] U_r):
    """
    # Solve riemann shock tube problem for a general equation of
    # state using the method of Colella, Glaz, and Ferguson.  See
    # Almgren et al. 2010 (the CASTRO paper) for details.
    #
    # The Riemann problem for the Euler's equation produces 4 regions,
    # separated by the three characteristics (u - cs, u, u + cs):
    #
    #
    #        u - cs    t    u      u + cs
    #          \       ^   .       /
    #           \  *L  |   . *R   /
    #            \     |  .     /
    #             \    |  .    /
    #         L    \   | .   /    R
    #               \  | .  /
    #                \ |. /
    #                 \|./
    #        ----------+----------------> x
    #
    # We care about the solution on the axis.  The basic idea is to use
    # estimates of the wave speeds to figure out which region we are in,
    # and: use jump conditions to evaluate the state there.
    #
    # Only density jumps across the u characteristic.  All primitive
    # variables jump across the other two.  Special attention is needed
    # if a rarefaction spans the axis.
    """

    cdef np.ndarray F = np.zeros([qx, qy, nvar], dtype=DTYPE)

    cdef int ilo, ihi, jlo, jhi
    cdef int nx, ny
    cdef int i, j

    cdef float smallc = 1.e-10
    cdef float smallrho = 1.e-10
    cdef float smallp = 1.e-10

    cdef float rho_l, un_l, ut_l, p_l, rhoe_l
    cdef float rho_r, un_r, ut_r, p_r, rhoe_r
    cdef np.ndarray xn = np.zeros(nspec, dtype=DTYPE)
    cdef float rhostar_l, rhostar_r, rhoestar_l, rhoestar_r
    cdef float ustar, pstar, cstar_l, cstar_r
    cdef float lambda_l, lambdastar_l, lambda_r, lambdastar_r
    cdef float W_l, W_r, c_l, c_r, sigma, alpha

    cdef float rho_state, un_state, ut_state, p_state, rhoe_state


    nx = qx - 2*ng
    ny = qy - 2*ng
    ilo = ng
    ihi = ng+nx
    jlo = ng
    jhi = ng+ny

    for j in range(jlo-1, jhi+1):
        for i in range(ilo-1, ihi+1):

            # primitive variable states
            rho_l  = U_l[i, j, idens]

            # un = normal velocity; ut = transverse velocity
            if idir == 1:
               un_l    = U_l[i, j, ixmom]/rho_l
               ut_l    = U_l[i, j, iymom]/rho_l
            else:
               un_l    = U_l[i, j, iymom]/rho_l
               ut_l    = U_l[i, j, ixmom]/rho_l

            rhoe_l = U_l[i, j, iener] - 0.5*rho_l*(un_l**2 + ut_l**2)

            p_l   = rhoe_l*(gamma - 1.0)
            p_l = max(p_l, smallp)

            rho_r  = U_r[i, j, idens]

            if idir == 1:
               un_r    = U_r[i, j, ixmom]/rho_r
               ut_r    = U_r[i, j, iymom]/rho_r
            else:
               un_r    = U_r[i, j, iymom]/rho_r
               ut_r    = U_r[i, j, ixmom]/rho_r

            rhoe_r = U_r[i, j, iener] - 0.5*rho_r*(un_r**2 + ut_r**2)

            p_r   = rhoe_r*(gamma - 1.0)
            p_r = max(p_r, smallp)

            # define the Lagrangian sound speed
            W_l = max(smallrho*smallc, np.sqrt(gamma*p_l*rho_l))
            W_r = max(smallrho*smallc, np.sqrt(gamma*p_r*rho_r))

            # and the regular sound speeds
            c_l = max(smallc, np.sqrt(gamma*p_l/rho_l))
            c_r = max(smallc, np.sqrt(gamma*p_r/rho_r))

            # define the star states
            pstar = (W_l*p_r + W_r*p_l + W_l*W_r*(un_l - un_r))/(W_l + W_r)
            pstar = max(pstar, smallp)
            ustar = (W_l*un_l + W_r*un_r + (p_l - p_r))/(W_l + W_r)

            # now compute the remaining state to the left and right
            # of the contact (in the star region)
            rhostar_l = rho_l + (pstar - p_l)/c_l**2
            rhostar_r = rho_r + (pstar - p_r)/c_r**2

            rhoestar_l = rhoe_l + \
                 (pstar - p_l)*(rhoe_l/rho_l + p_l/rho_l)/c_l**2
            rhoestar_r = rhoe_r + \
                 (pstar - p_r)*(rhoe_r/rho_r + p_r/rho_r)/c_r**2

            cstar_l = max(smallc, np.sqrt(gamma*pstar/rhostar_l))
            cstar_r = max(smallc, np.sqrt(gamma*pstar/rhostar_r))

            # figure out which state we are in, based on the location of
            # the waves
            if (ustar > 0.0):

                # contact is moving to the right, we need to understand
                # the L and *L states

                # Note: transverse velocity only jumps across contact
                ut_state = ut_l

                # define eigenvalues
                lambda_l = un_l - c_l
                lambdastar_l = ustar - cstar_l

                if (pstar > p_l):
                    # the wave is a shock -- find the shock speed
                    sigma = (lambda_l + lambdastar_l)/2.0

                    if (sigma > 0.0):
                        # shock is moving to the right -- solution is L state
                        rho_state = rho_l
                        un_state = un_l
                        p_state = p_l
                        rhoe_state = rhoe_l

                    else:
                        # solution is *L state
                        rho_state = rhostar_l
                        un_state = ustar
                        p_state = pstar
                        rhoe_state = rhoestar_l

                else:
                    # the wave is a rarefaction
                    if (lambda_l < 0.0 and lambdastar_l < 0.0):
                        # rarefaction fan is moving to the left -- solution is
                        # *L state
                        rho_state = rhostar_l
                        un_state = ustar
                        p_state = pstar
                        rhoe_state = rhoestar_l

                    elif (lambda_l > 0.0 and lambdastar_l > 0.0):
                        # rarefaction fan is moving to the right -- solution is
                        # L state
                        rho_state = rho_l
                        un_state = un_l
                        p_state = p_l
                        rhoe_state = rhoe_l

                    else:
                        # rarefaction spans x/t = 0 -- interpolate
                        alpha = lambda_l/(lambda_l - lambdastar_l)

                        rho_state  = alpha*rhostar_l  + (1.0 - alpha)*rho_l
                        un_state   = alpha*ustar      + (1.0 - alpha)*un_l
                        p_state    = alpha*pstar      + (1.0 - alpha)*p_l
                        rhoe_state = alpha*rhoestar_l + (1.0 - alpha)*rhoe_l

            elif (ustar < 0):
                # contact moving left, we need to understand the R and *R
                # states

                # Note: transverse velocity only jumps across contact
                ut_state = ut_r

                # define eigenvalues
                lambda_r = un_r + c_r
                lambdastar_r = ustar + cstar_r

                if (pstar > p_r):
                    # the wave if a shock -- find the shock speed
                    sigma = (lambda_r + lambdastar_r)/2.0

                    if (sigma > 0.0):
                        # shock is moving to the right -- solution is *R state
                        rho_state = rhostar_r
                        un_state = ustar
                        p_state = pstar
                        rhoe_state = rhoestar_r

                    else:
                        # solution is R state
                        rho_state = rho_r
                        un_state = un_r
                        p_state = p_r
                        rhoe_state = rhoe_r

                else:
                    # the wave is a rarefaction
                    if (lambda_r < 0.0 and lambdastar_r < 0.0):
                        # rarefaction fan is moving to the left -- solution is
                        # R state
                        rho_state = rho_r
                        un_state = un_r
                        p_state = p_r
                        rhoe_state = rhoe_r

                    elif (lambda_r > 0.0 and lambdastar_r > 0.0):
                        # rarefaction fan is moving to the right -- solution is
                        # *R state
                        rho_state = rhostar_r
                        un_state = ustar
                        p_state = pstar
                        rhoe_state = rhoestar_r

                    else:
                        # rarefaction spans x/t = 0 -- interpolate
                        alpha = lambda_r/(lambda_r - lambdastar_r)

                        rho_state  = alpha*rhostar_r  + (1.0 - alpha)*rho_r
                        un_state   = alpha*ustar      + (1.0 - alpha)*un_r
                        p_state    = alpha*pstar      + (1.0 - alpha)*p_r
                        rhoe_state = alpha*rhoestar_r + (1.0 - alpha)*rhoe_r

            else:  # ustar == 0

                rho_state = 0.5*(rhostar_l + rhostar_r)
                un_state = ustar
                ut_state = 0.5*(ut_l + ut_r)
                p_state = pstar
                rhoe_state = 0.5*(rhoestar_l + rhoestar_r)

            # species now
            if nspec > 0:
                if ustar > 0.0:
                    xn[:] = U_l[i, j, irhoX:irhoX-1+nspec]/U_l[i, j, idens]

                elif ustar < 0.0:
                    xn[:] = U_r[i, j, irhoX:irhoX-1+nspec]/U_r[i, j, idens]
                else:
                    xn[:] = 0.5*(U_l[i, j, irhoX:irhoX-1+nspec]/U_l[i, j, idens] + \
                                 U_r[i, j, irhoX:irhoX-1+nspec]/U_r[i, j, idens])

            # are we on a solid boundary?
            if idir == 1 :
               if (i == ilo) and (lower_solid == 1):
                  un_state = 0.0

               if (i == ihi+1) and (upper_solid == 1):
                  un_state = 0.0

            elif idir == 2:
               if (j == jlo) and (lower_solid == 1):
                  un_state = 0.0

               if (j == jhi+1) and (upper_solid == 1):
                  un_state = 0.0

            # compute the fluxes
            F[i, j, idens] = rho_state*un_state

            if idir == 1:
               F[i, j, ixmom] = rho_state*un_state**2 + p_state
               F[i, j, iymom] = rho_state*ut_state*un_state
            else:
               F[i, j, ixmom] = rho_state*ut_state*un_state
               F[i, j, iymom] = rho_state*un_state**2 + p_state


            F[i, j, iener] = rhoe_state*un_state + \
                 0.5*rho_state*(un_state**2 + ut_state**2)*un_state + \
                 p_state*un_state

            if (nspec > 0):
               F[i, j, irhoX:irhoX-1+nspec] = xn[:]*rho_state*un_state

    return F


def riemann_prim(int idir, int qx, int qy, int ng,
                 int nvar, int irho, int iu, int iv, int ip,
                 int iX, int nspec,
                 int lower_solid, int upper_solid,
                 float gamma,
                 np.ndarray[DTYPE_t, ndim=3] q_l,
                 np.ndarray[DTYPE_t, ndim=3] q_r):
    """
    # this is like riemann_cgf, except that it works on a primitive
    # variable input state and returns the primitive variable interface
    # state

    # Solve riemann shock tube problem for a general equation of
    # state using the method of Colella, Glaz, and Ferguson.  See
    # Almgren et al. 2010 (the CASTRO paper) for details.
    #
    # The Riemann problem for the Euler's equation produces 4 regions,
    # separated by the three characteristics (u - cs, u, u + cs):
    #
    #
    #        u - cs    t    u      u + cs
    #          \       ^   .       /
    #           \  *L  |   . *R   /
    #            \     |  .     /
    #             \    |  .    /
    #         L    \   | .   /    R
    #               \  | .  /
    #                \ |. /
    #                 \|./
    #        ----------+----------------> x
    #
    # We care about the solution on the axis.  The basic idea is to use
    # estimates of the wave speeds to figure out which region we are in,
    # and: use jump conditions to evaluate the state there.
    #
    # Only density jumps across the u characteristic.  All primitive
    # variables jump across the other two.  Special attention is needed
    # if a rarefaction spans the axis.

    """

    cdef np.ndarray q_int = np.zeros([qx, qy, nvar], dtype=DTYPE)

    cdef int ilo, ihi, jlo, jhi
    cdef int nx, ny
    cdef int i, j

    cdef float smallc = 1.e-10
    cdef float smallrho = 1.e-10
    cdef float smallp = 1.e-10

    cdef float rho_l, un_l, ut_l, p_l
    cdef float rho_r, un_r, ut_r, p_r
    cdef np.ndarray xn = np.zeros(nspec, dtype=DTYPE)
    cdef float rhostar_l, rhostar_r
    cdef float ustar, pstar, cstar_l, cstar_r
    cdef float lambda_l, lambdastar_l, lambda_r, lambdastar_r
    cdef float W_l, W_r, c_l, c_r, sigma
    cdef float alpha

    cdef float rho_state, un_state, ut_state, p_state

    nx = qx - 2*ng
    ny = qy - 2*ng
    ilo = ng
    ihi = ng+nx
    jlo = ng
    jhi = ng+ny

    for j in range(jlo-1, jhi+1):
        for i in range(ilo-1, ihi+1):

            # primitive variable states
            rho_l  = q_l[i, j, irho]

            # un = normal velocity; ut = transverse velocity
            if idir == 1:
               un_l    = q_l[i, j, iu]
               ut_l    = q_l[i, j, iv]
            else:
               un_l    = q_l[i, j, iv]
               ut_l    = q_l[i, j, iu]


            p_l   = q_l[i, j, ip]
            p_l = max(p_l, smallp)

            rho_r  = q_r[i, j, irho]

            if idir == 1:
               un_r    = q_r[i, j, iu]
               ut_r    = q_r[i, j, iv]
            else:
               un_r    = q_r[i, j, iv]
               ut_r    = q_r[i, j, iu]

            p_r   = q_r[i, j, ip]
            p_r = max(p_r, smallp)

            # define the Lagrangian sound speed
            W_l = max(smallrho*smallc, np.sqrt(gamma*p_l*rho_l))
            W_r = max(smallrho*smallc, np.sqrt(gamma*p_r*rho_r))

            # and the regular sound speeds
            c_l = max(smallc, np.sqrt(gamma*p_l/rho_l))
            c_r = max(smallc, np.sqrt(gamma*p_r/rho_r))

            # define the star states
            pstar = (W_l*p_r + W_r*p_l + W_l*W_r*(un_l - un_r))/(W_l + W_r)
            pstar = max(pstar, smallp)
            ustar = (W_l*un_l + W_r*un_r + (p_l - p_r))/(W_l + W_r)

            # now compute the remaining state to the left and right
            # of the contact (in the star region)
            rhostar_l = rho_l + (pstar - p_l)/c_l**2
            rhostar_r = rho_r + (pstar - p_r)/c_r**2

            cstar_l = max(smallc, np.sqrt(gamma*pstar/rhostar_l))
            cstar_r = max(smallc, np.sqrt(gamma*pstar/rhostar_r))

            # figure out which state we are in, based on the location of
            # the waves
            if ustar > 0.0:

                # contact is moving to the right, we need to understand
                # the L and *L states

                # Note: transverse velocity only jumps across contact
                ut_state = ut_l

                # define eigenvalues
                lambda_l = un_l - c_l
                lambdastar_l = ustar - cstar_l

                if pstar > p_l:
                    # the wave is a shock -- find the shock speed
                    sigma = (lambda_l + lambdastar_l)/2.0

                    if sigma > 0.0:
                        # shock is moving to the right -- solution is L state
                        rho_state = rho_l
                        un_state = un_l
                        p_state = p_l

                    else:
                        # solution is *L state
                        rho_state = rhostar_l
                        un_state = ustar
                        p_state = pstar

                else:
                      # the wave is a rarefaction
                    if (lambda_l < 0.0) and (lambdastar_l < 0.0):
                        # rarefaction fan is moving to the left -- solution is
                        # *L state
                        rho_state = rhostar_l
                        un_state = ustar
                        p_state = pstar

                    elif (lambda_l > 0.0) and (lambdastar_l > 0.0):
                        # rarefaction fan is moving to the right -- solution is
                        # L state
                        rho_state = rho_l
                        un_state = un_l
                        p_state = p_l

                    else:
                        # rarefaction spans x/t = 0 -- interpolate
                        alpha = lambda_l/(lambda_l - lambdastar_l)

                        rho_state  = alpha*rhostar_l  + (1.0 - alpha)*rho_l
                        un_state   = alpha*ustar      + (1.0 - alpha)*un_l
                        p_state    = alpha*pstar      + (1.0 - alpha)*p_l

            elif ustar < 0:

                # contact moving left, we need to understand the R and *R
                # states

                # Note: transverse velocity only jumps across contact
                ut_state = ut_r

                # define eigenvalues
                lambda_r = un_r + c_r
                lambdastar_r = ustar + cstar_r

                if pstar > p_r:
                    # the wave if a shock -- find the shock speed
                    sigma = (lambda_r + lambdastar_r)/2.0

                    if sigma > 0.0:
                        # shock is moving to the right -- solution is *R state
                        rho_state = rhostar_r
                        un_state = ustar
                        p_state = pstar

                    else:
                        # solution is R state
                        rho_state = rho_r
                        un_state = un_r
                        p_state = p_r

                else:
                    # the wave is a rarefaction
                    if (lambda_r < 0.0) and (lambdastar_r < 0.0):
                        # rarefaction fan is moving to the left -- solution is
                        # R state
                        rho_state = rho_r
                        un_state = un_r
                        p_state = p_r

                    elif (lambda_r > 0.0) and (lambdastar_r > 0.0):
                        # rarefaction fan is moving to the right -- solution is
                        # *R state
                        rho_state = rhostar_r
                        un_state = ustar
                        p_state = pstar

                    else:
                        # rarefaction spans x/t = 0 -- interpolate
                        alpha = lambda_r/(lambda_r - lambdastar_r)

                        rho_state  = alpha*rhostar_r  + (1.0 - alpha)*rho_r
                        un_state   = alpha*ustar      + (1.0 - alpha)*un_r
                        p_state    = alpha*pstar      + (1.0 - alpha)*p_r

            else:  # ustar == 0

                rho_state = 0.5*(rhostar_l + rhostar_r)
                un_state = ustar
                ut_state = 0.5*(ut_l + ut_r)
                p_state = pstar

            # species now
            if nspec > 0:
               if (ustar > 0.0):
                  xn[:] = q_l[i, j, iX:iX-1+nspec]

               elif ustar < 0.0:
                  xn[:] = q_r[i, j, iX:iX-1+nspec]
               else:
                  xn[:] = 0.5*(q_l[i, j, iX:iX-1+nspec] + q_r[i, j, iX:iX-1+nspec])

            # are we on a solid boundary?
            if idir == 1:
               if (i == ilo) and (lower_solid == 1):
                  un_state = 0.0

               if (i == ihi+1 and upper_solid == 1):
                  un_state = 0.0

            elif (idir == 2):
               if (j == jlo and lower_solid == 1):
                  un_state = 0.0

               if (j == jhi+1 and upper_solid == 1):
                  un_state = 0.0

            q_int[i, j, irho] = rho_state

            if idir == 1:
               q_int[i, j, iu] = un_state
               q_int[i, j, iv] = ut_state
            else:
               q_int[i, j, iu] = ut_state
               q_int[i, j, iv] = un_state

            q_int[i, j, ip] = p_state

            if nspec > 0:
               q_int[i, j, iX:iX-1+nspec] = xn[:]

    return q_int


def riemann_hllc(idir, qx, qy, ng,
                nvar, idens, ixmom, iymom, iener, irhoX, nspec,
                lower_solid, upper_solid,
                gamma, U_l, U_r):
    """
    # this is the HLLC Riemann solver.  The implementation follows
    # directly out of Toro's book.  Note: this fores not handle the
    # transonic rarefaction.
    """

    cdef np.ndarray F = np.zeros([qx, qy, nvar], dtype=DTYPE)

    cdef int ilo, ihi, jlo, jhi
    cdef int nx, ny
    cdef int i, j

    cdef float smallc = 1.e-10
    cdef float smallrho = 1.e-10
    cdef float smallp = 1.e-10

    cdef float rho_l, un_l, ut_l, rhoe_l, p_l
    cdef float rho_r, un_r, ut_r, rhoe_r, p_r
    cdef np.ndarray xn = np.zeros(nspec, dtype=DTYPE)

    cdef float rhostar_l, rhostar_r, rho_avg
    cdef float ustar, pstar
    cdef float Q, p_min, p_max, p_lr, p_guess
    cdef float factor, factor2
    cdef float g_l, g_r, A_l, B_l, A_r, B_r, z
    cdef float S_l, S_r, S_c
    cdef float c_l, c_r, c_avg

    cdef np.ndarray U_state = np.zeros(nvar, dtype=DTYPE)
    cdef float HLLCfactor

    nx = qx - 2*ng
    ny = qy - 2*ng
    ilo = ng
    ihi = ng+nx-1
    jlo = ng
    jhi = ng+ny-1

    for j in range(jlo-1, jhi+1):
        for i in range(ilo-1, ihi+1):

            # primitive variable states
            rho_l  = U_l[i, j, idens]

            # un = normal velocity; ut = transverse velocity
            if idir == 1:
               un_l    = U_l[i, j, ixmom]/rho_l
               ut_l    = U_l[i, j, iymom]/rho_l
            else:
               un_l    = U_l[i, j, iymom]/rho_l
               ut_l    = U_l[i, j, ixmom]/rho_l

            rhoe_l = U_l[i, j, iener] - 0.5*rho_l*(un_l**2 + ut_l**2)

            p_l   = rhoe_l*(gamma - 1.0)
            p_l = max(p_l, smallp)

            rho_r  = U_r[i, j, idens]

            if idir == 1:
               un_r    = U_r[i, j, ixmom]/rho_r
               ut_r    = U_r[i, j, iymom]/rho_r
            else:
               un_r    = U_r[i, j, iymom]/rho_r
               ut_r    = U_r[i, j, ixmom]/rho_r

            rhoe_r = U_r[i, j, iener] - 0.5*rho_r*(un_r**2 + ut_r**2)

            p_r   = rhoe_r*(gamma - 1.0)
            p_r = max(p_r, smallp)

            # compute the sound speeds
            c_l = max(smallc, np.sqrt(gamma*p_l/rho_l))
            c_r = max(smallc, np.sqrt(gamma*p_r/rho_r))

            # Estimate the star quantities -- use one of three methods to
            # for this -- the primitive variable Riemann solver, the two
            # shock approximation, or the two rarefaction approximation.
            # Pick the method based on the pressure states at the
            # interface.

            p_max = max(p_l, p_r)
            p_min = min(p_l, p_r)

            Q = p_max/p_min

            rho_avg = 0.5*(rho_l + rho_r)
            c_avg = 0.5*(c_l + c_r)

            # primitive variable Riemann solver (Toro, 9.3)
            factor = rho_avg*c_avg
            factor2 = rho_avg/c_avg

            pstar = 0.5*(p_l + p_r) + 0.5*(un_l - un_r)*factor
            ustar = 0.5*(un_l + un_r) + 0.5*(p_l - p_r)/factor

            rhostar_l = rho_l + (un_l - ustar)*factor2
            rhostar_r = rho_r + (ustar - un_r)*factor2

            if (Q > 2) and ((pstar < p_min) or (pstar > p_max)):

               # use a more accurate Riemann solver for the estimate here

               if pstar < p_min:

                  # 2-rarefaction Riemann solver
                  z = (gamma - 1.0)/(2.0*gamma)
                  p_lr = (p_l/p_r)**z

                  ustar = (p_lr*un_l/c_l + un_r/c_r + \
                            2.0*(p_lr - 1.0)/(gamma - 1.0)) / \
                          (p_lr/c_l + 1.0/c_r)

                  pstar = 0.5*(p_l*(1.0 + (gamma - 1.0)*(un_l - ustar)/ \
                                       (2.0*c_l) )**(1.0/z) + \
                                 p_r*(1.0 + (gamma - 1.0)*(ustar - un_r)/ \
                                       (2.0*c_r) )**(1.0/z) )

                  rhostar_l = rho_l*(pstar/p_l)**(1.0/gamma)
                  rhostar_r = rho_r*(pstar/p_r)**(1.0/gamma)

               else:

                  # 2-shock Riemann solver
                  A_r = 2.0/((gamma + 1.0)*rho_r)
                  B_r = p_r*(gamma - 1.0)/(gamma + 1.0)

                  A_l = 2.0/((gamma + 1.0)*rho_l)
                  B_l = p_l*(gamma - 1.0)/(gamma + 1.0)

                  # guess of the pressure
                  p_guess = max(0.0, pstar)

                  g_l = np.sqrt(A_l / (p_guess + B_l))
                  g_r = np.sqrt(A_r / (p_guess + B_r))

                  pstar = (g_l*p_l + g_r*p_r - (un_r - un_l))/(g_l + g_r)

                  ustar = 0.5*(un_l + un_r) + \
                       0.5*( (pstar - p_r)*g_r - (pstar - p_l)*g_l)

                  rhostar_l = rho_l*(pstar/p_l + (gamma-1.0)/(gamma+1.0))/ \
                       ( (gamma-1.0)/(gamma+1.0)*(pstar/p_l) + 1.0)

                  rhostar_r = rho_r*(pstar/p_r + (gamma-1.0)/(gamma+1.0))/ \
                       ( (gamma-1.0)/(gamma+1.0)*(pstar/p_r) + 1.0)

            # estimate the nonlinear wave speeds

            if pstar <= p_l:
               # rarefaction
               S_l = un_l - c_l
            else:
               # shock
               S_l = un_l - c_l*np.sqrt(1.0 +
                    ((gamma+1.0)/(2.0*gamma))* (pstar/p_l - 1.0))


            if (pstar <= p_r):
               # rarefaction
               S_r = un_r + c_r
            else:
               # shock
               S_r = un_r + c_r*np.sqrt(1.0 +
                     ((gamma+1.0)/(2.0/gamma))* (pstar/p_r - 1.0))


            #  We could just take S_c = u_star as the estimate for the
            #  contact speed, but we can actually for this more accurately
            #  by using the Rankine-Hugonoit jump conditions across each
            #  of the waves (see Toro 10.58, Batten et al. SIAM
            #  J. Sci. and Stat. Comp., 18:1553 (1997)
            S_c = (p_r - p_l + rho_l*un_l*(S_l - un_l) -
                   rho_r*un_r*(S_r - un_r))/ \
                  (rho_l*(S_l - un_l) - rho_r*(S_r - un_r))


            # figure out which region we are in and compute the state and
            # the interface fluxes using the HLLC Riemann solver
            if (S_r <= 0.0):
               # R region
               U_state[:] = U_r[i, j, :]

               F[i, j, :] = consFlux(idir, gamma, idens, ixmom, iymom,
                                     iener, irhoX, nvar, nspec, U_state)

            elif (S_r > 0.0) and (S_c <= 0):
               # R* region
               HLLCfactor = rho_r*(S_r - un_r)/(S_r - S_c)

               U_state[idens] = HLLCfactor

               if idir == 1:
                  U_state[ixmom] = HLLCfactor*S_c
                  U_state[iymom] = HLLCfactor*ut_r
               else:
                  U_state[ixmom] = HLLCfactor*ut_r
                  U_state[iymom] = HLLCfactor*S_c


               U_state[iener] = HLLCfactor*(U_r[i, j, iener]/rho_r +
                    (S_c - un_r)*(S_c + p_r/(rho_r*(S_r - un_r))))

               # species
               if nspec > 0:
                  U_state[irhoX:irhoX-1+nspec] = HLLCfactor * \
                    U_r[i, j, irhoX:irhoX-1+nspec]/rho_r


               # find the flux on the right interface
               F[i, j, :] = consFlux(idir, gamma, idens, ixmom, iymom,
                                     iener, irhoX, nvar, nspec,
                                     U_r[i, j, :])

               # correct the flux
               F[i, j, :] = F[i, j, :] + S_r*(U_state[:] - U_r[i, j, :])

            elif (S_c > 0.0 and S_l < 0.0):
               # L* region
               HLLCfactor = rho_l*(S_l - un_l)/(S_l - S_c)

               U_state[idens] = HLLCfactor

               if idir == 1:
                  U_state[ixmom] = HLLCfactor*S_c
                  U_state[iymom] = HLLCfactor*ut_l
               else:
                  U_state[ixmom] = HLLCfactor*ut_l
                  U_state[iymom] = HLLCfactor*S_c


               U_state[iener] = HLLCfactor*(U_l[i, j, iener]/rho_l +
                    (S_c - un_l)*(S_c + p_l/(rho_l*(S_l - un_l))))

               # species
               if (nspec > 0):
                  U_state[irhoX:irhoX-1+nspec] = HLLCfactor * \
                    U_l[i, j, irhoX:irhoX-1+nspec]/rho_l


               # find the flux on the left interface
               F[i, j, :] = consFlux(idir, gamma, idens, ixmom, iymom,
                                     iener, irhoX, nvar, nspec,
                                     U_l[i, j, :])

               # correct the flux
               F[i, j, :] = F[i, j, :] + S_l*(U_state[:] - U_l[i, j, :])

            else:
               # L region
               U_state[:] = U_l[i, j, :]

               F[i, j, :] = consFlux(idir, gamma, idens, ixmom, iymom,
                                     iener, irhoX, nvar, nspec,
                                     U_state)

            # we should deal with solid boundaries somehow here
    return F

def consFlux(int idir, float gamma, int idens, int ixmom, int iymom,
             int iener, int irhoX, int nvar, int nspec, np.ndarray[DTYPE_t, ndim=1] U_state):

    cdef np.ndarray F = np.zeros(nvar, dtype=DTYPE)

    cdef float p, u, v

    u = U_state[ixmom]/U_state[idens]
    v = U_state[iymom]/U_state[idens]

    p = (U_state[iener] - 0.5*U_state[idens]*(u*u + v*v))*(gamma - 1.0)

    if idir == 1:
        F[idens] = U_state[idens]*u
        F[ixmom] = U_state[ixmom]*u + p
        F[iymom] = U_state[iymom]*u
        F[iener] = (U_state[iener] + p)*u
        if nspec > 0:
            F[irhoX:irhoX-1+nspec] = U_state[irhoX:irhoX-1+nspec]*u

    else:
        F[idens] = U_state[idens]*v
        F[ixmom] = U_state[ixmom]*v
        F[iymom] = U_state[iymom]*v + p
        F[iener] = (U_state[iener] + p)*v
        if nspec > 0:
            F[irhoX:irhoX-1+nspec] = U_state[irhoX:irhoX-1+nspec]*v

    return F


def artificial_viscosity(int qx, int qy, int ng, float dx, float dy,
                         float cvisc,
                         np.ndarray[DTYPE_t, ndim=2] u,
                         np.ndarray[DTYPE_t, ndim=2] v):
    """
    # compute the artifical viscosity.  Here, we compute edge-centered
    # approximations to the divergence of the velocity.  This follows
    # directly Colella \ Woodward (1984) Eq. 4.5
    #
    # data locations:
    #
    #   j+3/2--+---------+---------+---------+
    #          |         |         |         |
    #     j+1  +         |         |         |
    #          |         |         |         |
    #   j+1/2--+---------+---------+---------+
    #          |         |         |         |
    #        j +         X         |         |
    #          |         |         |         |
    #   j-1/2--+---------+----Y----+---------+
    #          |         |         |         |
    #      j-1 +         |         |         |
    #          |         |         |         |
    #   j-3/2--+---------+---------+---------+
    #          |    |    |    |    |    |    |
    #              i-1        i        i+1
    #        i-3/2     i-1/2     i+1/2     i+3/2
    #
    # X is the location of avisco_x[i, j]
    # Y is the location of avisco_y[i, j]
    """


    cdef np.ndarray avisco_x = np.zeros([qx, qy], dtype=DTYPE)
    cdef np.ndarray avisco_y = np.zeros([qx, qy], dtype=DTYPE)
    cdef int ilo, ihi, jlo, jhi
    cdef int nx, ny
    cdef int i, j
    cdef float divU_x, divU_y

    nx = qx - 2*ng
    ny = qy - 2*ng
    ilo = ng
    ihi = ng+nx
    jlo = ng
    jhi = ng+ny

    for j in range(jlo-1, jhi+1):
        for i in range(ilo-1, ihi+1):

            # start by computing the divergence on the x-interface.  The
            # x-difference is simply the difference of the cell-centered
            # x-velocities on either side of the x-interface.  For the
            # y-difference, first average the four cells to the node on
            # each end of the edge, and: difference these to find the
            # edge centered y difference.
            divU_x = (u[i, j] - u[i-1, j])/dx + \
                 0.25*(v[i,j+1] + v[i-1,j+1] - v[i,j-1] - v[i-1,j-1])/dy

            avisco_x[i, j] = cvisc*max(-divU_x*dx, 0.0)

            # now the y-interface value
            divU_y = 0.25*(u[i+1,j] + u[i+1,j-1] - u[i-1, j] - u[i-1,j-1])/dx + \
                 (v[i, j] - v[i,j-1])/dy

            avisco_y[i, j] = cvisc*max(-divU_y*dy, 0.0)

    return avisco_x, avisco_y
