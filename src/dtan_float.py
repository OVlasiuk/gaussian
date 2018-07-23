#!/usr/bin/env python
from __future__ import division
import os as os
import numpy as np
from optparse import OptionParser
#
from ipopt import minimize_ipopt
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
        "-C",
        "--coeff",
        dest="C", default=10.0,
        help="Coefficient in the exponent. Default: 10.")
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


modE = SourceModule("""
#define DIM 3
#define BLOCK_SIZE 256
//typedef struct { float x, y, z, e; } Body;

__global__
void energy(float *pt, float3 *p, float* c, int n) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
     if (i < n) {
        float S = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz;

            S += exp(- *c *distSqr);
        }
        pt[i] = S;
    }
}
""")

modG = SourceModule("""
#define DIM 3

//typedef struct { float x, y, z, e;  } Body;

__global__
void gradient(float3* grad, float3 *p, float* c, int n) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
    float Sx = 0.0;
    float Sy = 0.0;
    float Sz = 0.0;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float S = exp(- *c *(dx*dx + dy*dy + dz*dz));

      Sx += 4.0f* *c *dx*S;
      Sy += 4.0f* *c *dy*S;
      Sz += 4.0f* *c *dz*S;
    }

    grad[i].x = Sx;
    grad[i].y = Sy;
    grad[i].z = Sz;
  }
}
""")
energy = modE.get_function("energy")
gradient = modG.get_function("gradient")


def gaussian(cnf, pt_dev, cnf_dev, c_dev, pt, n):
    blocksize = 256
    numblocks = int(((n + blocksize - 1) // blocksize)[0])
    cnf_dev.set(cnf.astype('float32'))
    c_dev.set(c.astype('float32'))
    energy(pt_dev, cnf_dev, c_dev, n,
           block=(blocksize, 1, 1), grid=(numblocks, 1))
    pt[:] = pt_dev.get().astype('float64')
    return pt.sum().astype('float64')


def gaussian_grad(cnf, grad_dev, cnf_dev, c_dev, n):
    blocksize = 256
    numblocks = int(((n + blocksize - 1) // blocksize)[0])
    cnf_dev.set(cnf.astype('float32'))
    c_dev.set(c.astype('float32'))
    gradient(grad_dev, cnf_dev, c_dev, n,
             block=(blocksize, 1, 1), grid=(numblocks, 1))
#     pt_dev.get(pt)
    return grad_dev.get().astype('float64')


if __name__ == "__main__":
    parser = OptionParser()
    save, C, dim, N, initpts, iterations = get_options(parser)
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Initialize
    if initpts is not None:
        cnf = np.loadtxt(initpts, delimiter='\t').flatten()
        N = cnf.shape[0]
        dim = cnf.shape[1]
    else:
        cnf = np.random.random(dim*N)
    c = C * np.ones(1, dtype='float64')
    n = N*np.ones(1, dtype='uint32')
    pt = np.zeros(N, dtype='float64')
    grad = np.zeros(dim*N, dtype='float64')

    c_dev = to_gpu(c.astype('float32'))
    n_dev = to_gpu(n)
    cnf_dev = to_gpu(cnf.astype('float32'))
    grad_dev = to_gpu(grad.astype('float32'))
    pt_dev = to_gpu(pt.astype('float32'))

    # Bounds
    lb = np.zeros_like(cnf)
    ub = np.ones_like(cnf)
    bnds = tuple([(lb[i], ub[i]) for i in range(N*dim)])

    f0 = np.inf
    # for I in range(iterations):

    res = minimize_ipopt(
        lambda X: gaussian(X, pt_dev, cnf_dev, c_dev, pt, n),
        cnf, jac=lambda X: gaussian_grad(X, grad_dev, cnf_dev, c_dev, n),
        bounds=bnds, options={'maxiter': 1000}
    )
    print("Status: %s\nEnergy: %10.4f\n" % (res.success, res.fun))
    if res.fun < f0:
        f0 = res.fun
        x0 = res.x
    if save:
        if not os.path.isdir("../out"):
            os.mkdir("../out")
        fname = ('../out/G_' + str(C) + '_dim_'
                 + str(dim)+'_N_' + str(N)+'.out')
        np.savetxt(fname, x0.reshape((-1, dim)), fmt='%.18f', delimiter='\t')
    else:
        pplot(x0, dim)
