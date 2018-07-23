#!/usr/bin/env python
from __future__ import division
import os as os
import numpy as np
from math import pi, cos, sin
from optparse import OptionParser
# from scipy.optimize import check_grad, approx_fprime
# from scipy.linalg import norm
#
from ipopt import minimize_ipopt  # NOQA
# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
# Pycuda
import pycuda.autoinit  # NOQA
from pycuda.compiler import SourceModule
from pycuda.gpuarray import to_gpu


def get_options(parser):
    """ Define command line options."""
    parser.add_option(
        "-s",
        "--save",
        action="store_true", dest="save", default=False,
        help="Whether to save the output in a file. Default: False.")
    parser.add_option(
        "-p",
        "--power",
        dest="S", default=2.0,
        help="(Absolute value of) the Riesz power. Default: 2.")
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
    # save, C, dim, N, initpts, iterations = get_options(parser)
    return bool(options.save), float(options.S), int(options.dim),\
        int(options.N), options.initpts, int(options.iterations)


def sph2cart(phitheta):
    return np.array([cos(phitheta[0])*sin(phitheta[1]),
                     sin(phitheta[0])*sin(phitheta[1]),
                     cos(phitheta[1])])


def pplot(x):
    dim = 3
    X3 = np.empty((len(x)//dim, dim))
    for i in range(len(X3)):
        X3[i, :] = sph2cart(x[i*2:(i+1)*2]).reshape((-1,))
    r = 1
    coeff = .94
    phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    # Set colors and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(coeff*x, coeff*y, coeff*z,  rstride=4, cstride=4,
                      color="blue", alpha=0.3, linewidth=1)
    ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], marker='o', color='red')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


modE = SourceModule("""
__global__
void energy(double *pt, double2 *p, double* s, int n) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
     if (i < n) {
        double S = 0.0;
        for (int j = 0; j < n; j++) {
            if (j==i)
                continue;
            double cosphi = cos(p[j].x - p[i].x);
            double sintheta = sin(p[j].y) * sin(p[i].y);
            double costheta = cos(p[j].y) * cos(p[i].y);
            double distSqr = 2.0 - 2.0*(sintheta*cosphi+costheta);
            S += 1/(distSqr*distSqr); // s = 4
            // S += pow(distSqr, -s/2.0);
        }
        pt[i] = S;
    }
}
""")

modG3 = SourceModule("""

__global__
void gradient3(double3* grad3, double2 *p, double* s, int n) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
    double Sx = 0.0;
    double Sy = 0.0;
    double Sz = 0.0;

    for (int j = 0; j < n; j++) {
        if (j==i)
            continue;
        double dx = cos(p[j].x) * sin(p[j].y) - cos(p[i].x) * sin(p[i].y);
        double dy = sin(p[j].x) * sin(p[j].y) - sin(p[i].x) * sin(p[i].y);
        double dz = cos(p[j].y)               - cos(p[i].y);
        double distSqr = dx*dx + dy*dy + dz*dz;
        double S = distSqr*distSqr*distSqr; // s = 4

        // s = 4
        Sx += 2.0*4.0* dx/S;
        Sy += 2.0*4.0* dy/S;
        Sz += 2.0*4.0* dz/S;
    }

    grad3[i].x = Sx;
    grad3[i].y = Sy;
    grad3[i].z = Sz;
  }
}
""")

modG = SourceModule("""
#define DIM 3

__global__
void gradient(double2* grad, double3* grad3, double2 *p, int n) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
    double M0 = -sin(p[i].x) * sin(p[i].y);
    double M1 =  cos(p[i].x) * sin(p[i].y);
    double M2 =  0.0;
    double M3 = cos(p[i].x) * cos(p[i].y);
    double M4 = sin(p[i].x) * cos(p[i].y);
    double M5 = -sin(p[i].y);

    grad[i].x = grad3[i].x*M0 + grad3[i].y*M1 + grad3[i].z*M2;
    grad[i].y = grad3[i].x*M3 + grad3[i].y*M4 + grad3[i].z*M5;
  }
}
""")
energy = modE.get_function("energy")
gradient3 = modG3.get_function("gradient3")
gradient = modG.get_function("gradient")


def riesz(cnf, pt_dev, cnf_dev, s_dev, pt, n):
    blocksize = 256
    numblocks = int(((n + blocksize - 1) // blocksize)[0])
    cnf_dev.set(cnf)
    energy(
        pt_dev, cnf_dev, s_dev, n,
        block=(blocksize, 1, 1), grid=(numblocks, 1)
    )
    pt_dev.get(pt)
    return pt.sum()


def riesz_grad(cnf, grad_dev, grad3_dev, cnf_dev, s_dev, n):
    blocksize = 256
    numblocks = int(((n + blocksize - 1) // blocksize)[0])
    cnf_dev.set(cnf)
    gradient3(
        grad3_dev, cnf_dev, s_dev, n,
        block=(blocksize, 1, 1), grid=(numblocks, 1)
    )
    gradient(
        grad_dev, grad3_dev, cnf_dev, n,
        block=(blocksize, 1, 1), grid=(numblocks, 1)
    )
    return grad_dev.get()


if __name__ == "__main__":
    parser = OptionParser()
    save, S, dim, N, initpts, iterations = get_options(parser)
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Initialize
    if initpts is not None:
        cnf = np.loadtxt(initpts, delimiter='\t').flatten()
        N = cnf.shape[0]
        dim = cnf.shape[1]
    else:
        cnf = 2*pi*np.random.random((dim-1)*N)
        # np.array([0,0, 0,pi/2, 0, pi])#
    s = S * np.ones(1, dtype='float64')
    n = N*np.ones(1, dtype='uint32')
    pt = np.zeros(N, dtype='float64')
    grad3 = np.zeros(dim*N, dtype='float64')
    grad = np.zeros((dim-1)*N, dtype='float64')

    s_dev = to_gpu(s)
    n_dev = to_gpu(n)
    cnf_dev = to_gpu(cnf)
    grad3_dev = to_gpu(grad3)
    grad_dev = to_gpu(grad)
    pt_dev = to_gpu(pt)
    #
    f0 = np.inf
    # Bounds
    lb = np.zeros_like(cnf)
    ub = 2*pi*np.ones_like(cnf)
    bnds = tuple([(lb[i], ub[i]) for i in range(len(cnf))])
    # print(check_grad(lambda X: riesz(X, pt_dev, cnf_dev, s_dev, pt, n),
    #                  lambda X: riesz_grad(X, grad_dev, grad3_dev, cnf_dev,
    #                                       s_dev, n), cnf  ))
    # print(grad_dev.get())
    # print(" Approximate grad:")
    # print(approx_fprime(cnf,
    #                     lambda X: riesz(X, pt_dev, cnf_dev, s_dev, pt, n),
    #                     1e-8
    #                     ))
    # print(norm(grad_dev.get() - approx_fprime(cnf,
    #                     lambda X: riesz(X, pt_dev, cnf_dev, s_dev, pt, n),
    #                     1e-8
    #                     )))

    # print(riesz(cnf, pt_dev, cnf_dev, s_dev, pt, n))
    # print(riesz_grad(cnf, grad_dev, grad3_dev, cnf_dev, s_dev, n)[-1])
    for I in range(iterations):
        res = minimize_ipopt(
            lambda X: riesz(X, pt_dev, cnf_dev, s_dev, pt, n), cnf,
            jac=lambda X: riesz_grad(X, grad_dev, grad3_dev,
                                     cnf_dev, s_dev, n),
            bounds=bnds, options={'maxiter': 1000}
        )
        print("Status: %s\nEnergy: %10.4f\n" % (res.success, res.fun))
        if res.fun < f0:
            f0 = res.fun
            x0 = res.x
    if save:
        if not os.path.isdir("out"):
            os.mkdir("out")
        fname = ('out/G_' + str(s) + '_dim_'
                 + str(dim)+'_N_' + str(N)+'.out')
        np.savetxt(fname, x0.reshape((-1, dim)), fmt='%.18f', delimiter='\t')
    else:
        pplot(x0)
