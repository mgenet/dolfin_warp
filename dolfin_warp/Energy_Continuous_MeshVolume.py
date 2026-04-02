#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy
import typing

import dolfin_warp as dwarp

from .Energy_Continuous import ContinuousEnergy
from .Problem           import Problem

################################################################################

class MeshVolumeContinuousEnergy(ContinuousEnergy):



    def __init__(self,
            problem: Problem,
            quadrature_degree: typing.Optional[int] = None,
            name: str = "vol",
            w: float = 1.):

        self.problem           = problem
        self.printer           = self.problem.printer
        self.quadrature_degree = quadrature_degree
        self.name              = name
        self.w                 = w

        self.printer.print_str("Defining mesh volume energy…")
        self.printer.inc()

        # dV
        self.form_compiler_parameters = {
            "quadrature_degree":self.quadrature_degree}
        self.dV = dolfin.Measure(
            "dx",
            domain=self.problem.mesh,
            metadata=self.form_compiler_parameters)

        # forms
        self.mesh_V0 = dolfin.assemble(dolfin.Constant(1.) * self.dV)
        self.Psi = 1./self.problem.J/self.mesh_V0
        # self.Psi = -self.problem.J/self.mesh_V0 # MG20260331: negative energies are kind of a mess…
        if   (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
            self.DPsi  = dolfin.derivative(self.Psi , self.problem.U, self.problem.dU_test )
            self.DDPsi = dolfin.derivative(self.DPsi, self.problem.U, self.problem.dU_trial)
        elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
            self.DPsi   = dolfin.dot(dolfin.diff(self.Psi, self.problem.U), self.problem.dU_test)
            self.DDPsi  = dolfin.dot(dolfin.dot(dolfin.diff(dolfin.diff(self.Psi, self.problem.U), self.problem.U), self.problem.dU_test), self.problem.dU_trial)
            # self.DDPsi += dolfin.dot(dolfin.diff(self.Psi, self.problem.U), self.problem.ddU_test_trial)

        self.ener_form = self.Psi   * self.dV
        self.res_form  = self.DPsi  * self.dV
        self.jac_form  = self.DDPsi * self.dV

        self.printer.dec()
