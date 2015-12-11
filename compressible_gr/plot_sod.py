import numpy as np
import matplotlib.pyplot as plt

def plot_sod():
    exact = np.loadtxt("sod-exact.out")
    x_exact   = exact[:,0]
    rho_exact = exact[:,1]
    u_exact   = exact[:,2]
    p_exact   = exact[:,3]
    e_exact   = exact[:,4]

    fig, axes = plt.subplots(nrows=4, ncols=1, num=1)

    plt.rc("font", size=10)

    ax = axes.flat[0]

    ax.plot(x_exact, rho_exact, 'b')

    ax.set_ylabel(r"$\rho$")
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.1)

    ax = axes.flat[1]

    ax.plot(x_exact, u_exact, 'r')

    ax.set_ylabel(r"$u$")
    ax.set_xlim(0,1.0)

    ax = axes.flat[2]

    ax.plot(x_exact, p_exact, 'k')

    ax.set_ylabel(r"$p$")
    ax.set_xlim(0,1.0)

    ax = axes.flat[3]

    ax.plot(x_exact, e_exact, 'g')

    ax.set_ylabel(r"$e$")
    ax.set_xlim(0,1.0)

    plt.subplots_adjust(hspace=0.25)

    fig.set_size_inches(4.5,9.0)

    plt.savefig("sod_compare.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    plot_sod()
