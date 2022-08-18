#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import math
import dolfin

################################################################################

def compute_energies_normalization(
        problem,
        k=None,
        x0=None):

    dim = problem.mesh_dimension

    if (k is None):
        k = [0.1 * math.pi / problem.mesh.hmin()]*dim

    if (x0 is None):
        x0 = [0]*dim

    if (dim == 2):
        U_expr = dolfin.Expression(
            ("sin(kx * (x[0] - x0))", "sin(ky * (x[1] - y0))"),
            kx=k[0], ky=k[1],
            x0=x0[0], y0=x0[1],
            element=problem.U_fe)
    elif (dim == 3):
        U_expr = dolfin.Expression(
            ("sin(kx * (x[0] - x0))", "sin(ky * (x[1] - y0))", "sin(kz * (x[2] - z0))"),
            kx=k[0], ky=k[1], kz=k[2],
            x0=x0[0], y0=x0[1], z0=x0[2],
            element=problem.U_fe)

    problem.U.interpolate(U_expr)
    problem.U_norm = problem.U.vector().norm("l2")
    problem.U.vector()[:] /= problem.U_norm
    problem.U_norm = problem.U.vector().norm("l2")

    for energy in problem.energies:
        energy.ener0 = energy.assemble_ener(w_weight=0)
        # print(energy.name)
        # print(energy.ener0)

    problem.U.vector().zero()
    problem.U_norm = 0
