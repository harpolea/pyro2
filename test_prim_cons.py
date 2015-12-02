from compressible_gr.unsplitFluxes import *
import numpy as np
import matplotlib.pyplot as plt
import random

def test_prim_con():
    """
    Tests the cons_to_prim and prim_to_cons functions by generating random initial data and checking that it can be recovered.
    """

    N = 10000 # number of sets of initial data to try


    Qs = np.zeros((5,N)) # matrix of data
    errs = np.zeros((5, N)) # matrix of relative errors in each of the variables after converting forwards and back

    for i in range(N):
        c = 1. # speed of light
        gamma = 5./3.
        K = 1.e5 # polytropic coefficient

        # generate random primitive data
        rho = random.random() * 1.e-5
        s = c * random.random() # speed - must be less than c
        u = s * random.random() # this randomly makes u some fraction of the total speed
        v = np.sqrt(s**2 - u**2)
        p = K * rho**gamma
        h = 1. + (p * gamma) / (rho * (gamma - 1.))

        Q = (rho, u, v, h, p)
        Qs[:, i] = [rho, u, v, h, p]

        Qc = prim_to_cons(Q, c, gamma)
        Qp, _ = cons_to_prim(Qc, c, gamma)

        errs[:, i] =np.log(np.fabs((np.array(list(Q)) - np.array(list(Qp)))/np.fabs(list(Q))))

    # now do a box plot of the relative errors
    #errs[:,:] += 35.
    labels = [r"$\rho$", r"$u$", r"$v$", r"$p$", r"$h$"]
    colours = ['pink', 'lightblue', 'lightgreen', 'lemonchiffon', 'thistle']
    plt.figure(figsize=(10,7))
    plt.rc("font", size=15)

    bplot = plt.boxplot(np.transpose(errs), labels=labels, patch_artist=True)

    # add some colour
    for patch, colour in zip(bplot['boxes'], colours):
        patch.set_facecolor(colour)

    plt.ylabel(r'$\ln\|$relative error$\|$')
    plt.savefig('compressible_gr/prim_cons_errs.png')
    plt.show()

if __name__ == "__main__":
    test_prim_con()
