#!/usr/bin/env python
from __future__ import division
from ipopt import minimize_ipopt
import os as os
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
# from numba import jit, float64, uint8, prange


def get_options(parser):
    """ Define command line options."""
    parser.add_option(
        "-s",
        "--save",
        action="store_true", dest="save", default=False,
        help="Whether to save the output in a file. Default: False.")
    parser.add_option(
        "-C",
        "--coeff",
        dest="C", default=1.0,
        help="Coefficient in the exponent. Default: 1.0.")
    parser.add_option(
        "-d",
        "--dim",
        dest="dim",
        default=3,
        help="Dimension of the ambient space. Default: 3.")
    parser.add_option(
        "-N",
        "--numpts",
        dest="N",
        default=800,
        help="Number of particles. Default: 800.")
    parser.add_option(
        "-i",
        "--initpts",
        dest="initpts",
        default=None,
        help="A starting configuration. Must be a string with the file name\
        relative to the script's directory. Default: None.")
    parser.add_option(
        "-r",
        "--repeat",
        dest="iterations",
        default=1,
        help="A number of iterations to perform. Always 1 if a configuration\
        is given. Default: 1.")

    options, args = parser.parse_args()
    return bool(options.save), float(options.C), int(options.dim),\
        int(options.N), options.initpts, int(options.iterations)


def pplot(x, dim):
    X3 = x.reshape((-1, dim))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], marker='o', color='red')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

# @jit(float64(float64[:], float64, uint8), nogil=True, cache=True)


def gaussian_scp(X, C, dim):
    #   SciPy is somewhat different in terms of function/gradient calls
    en_all = 0
    for l in range(dim):
        en_all = en_all - C*(X.reshape((-1, dim))[:, l][None, :]
                             - X.reshape((-1, dim))[:, l][:, None])**2.
    en_all = np.exp(en_all)
    return en_all.sum()


def gaussian_scp_grad(X, C, dim):
    grad = np.zeros_like(X)
    en_all = 0
    for l in range(dim):
        en_all = en_all - C*(X.reshape((-1, dim))[:, l][None, :]
                             - X.reshape((-1, dim))[:, l][:, None])**2.
    en_all = np.exp(en_all)
    for l in range(dim):
        grad.reshape((-1, dim))[:, l] = C * 2. * np.sum(
            en_all*(-2*(X.reshape((-1, dim))[:, l][None, :]
                        - X.reshape((-1, dim))[:, l][:, None])),
            1)
    return -grad


if __name__ == "__main__":
    parser = OptionParser()
    save, C, dim, N, initpts, iterations = get_options(parser)
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    f0 = np.inf
    for I in range(iterations):
        if initpts is not None:
            u = np.loadtxt(initpts, delimiter='\t').flatten()
        else:
            u = np.random.random((N, dim)).flatten()
        #
        # Bounds
        lb = np.zeros_like(u)
        ub = np.ones_like(u)
        bnds = tuple([(lb[i], ub[i]) for i in range(N*dim)])
        #
        res = minimize_ipopt(lambda X: gaussian_scp(X, C, dim), u,
                             jac=lambda X: gaussian_scp_grad(X, C, dim),
                             bounds=bnds, options={'maxiter': 1000})
        print("Status: %s\nEnergy: %10.4f\n" % (res.success, res.fun))
        if res.fun < f0:
            f0 = res.fun
            x0 = res.x
            # prompt = input("Save the config (y/[n])?\n")
            if save:
                fname = ('out/G_' + str(C) + '_dim_' + str(dim)+'_N_'
                         + str(N)+'.out')
                if not os.path.isdir("out"):
                    os.mkdir("out")
                else:
                    np.savetxt(fname, x0.reshape((-1, dim)), fmt='%.18f',
                               delimiter='\t')
            else:
                pplot(x0, dim)
