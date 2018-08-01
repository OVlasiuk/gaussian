#!/usr/bin/env python
from __future__ import division
import os as os
from re import sub
import argparse
#
import numpy as np
# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA


def pplot(fname, save):
    X3 = np.loadtxt(fname, delimiter='\t')
    r = 1
    coeff = .94
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    # Set colors and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(coeff*x, coeff*y, coeff*z,  rstride=3, cstride=3,
                      color="teal", alpha=0.4, linewidth=1)
    # ax.plot_surface(coeff*x, coeff*y, coeff*z, rstride=1, cstride=1,
    # linewidth=0, antialiased=False)
    print(str(X3.shape[0]) + " points")
    ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], marker='o', color='red')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")
    plt.tight_layout()
    savename = sub(r'\....\Z', '.png', fname)
    if save:
        print(savename)
        plt.savefig(savename)
    plt.show()


if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    print(os.getcwd())
    parser = argparse.ArgumentParser(description='Load a configuration from\
                                     file and plot it')
    parser.add_argument(
        'fname', metavar='filename', type=str, nargs=1,
        help='A textfile with the configuration. Must be a string\
                        with the file name relative to the script\'s\
                        directory. Default: None.')
    parser.add_argument('-s', dest='save', action='store_true',
                        default=False,
                        help='Whether to save the output. Default: false.')

    args = parser.parse_args()
    pplot(args.fname[0], args.save)
