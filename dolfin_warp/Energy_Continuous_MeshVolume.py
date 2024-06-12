#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2024                                       ###
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
            type: str = "mesh-discrete",
            model: str = "ogdenciarletgeymonatneohookean",
            b_fin: typing.Optional["list[float]"] = None,
            poisson: float = 0.,
            volume_subdomain_data = None,
            volume_subdomain_id = None,
            surface_subdomain_data = None,
            surface_subdomain_id = None,
            w: float = 1.):
    

        self.problem           = problem
        self.printer           = self.problem.printer
        self.quadrature_degree = quadrature_degree
        self.name              = name
        self.w                 = w

        self.printer.print_str("Defining mesh volume energy…")
        self.printer.inc()

        # J
        self.I = dolfin.Identity(self.problem.mesh_dimension)
        self.F = self.I + dolfin.grad(self.problem.U)
        self.J = dolfin.det(self.F)

        # dV
        self.form_compiler_parameters = {
            "quadrature_degree":self.quadrature_degree}
        self.dV = dolfin.Measure(
            "dx",
            domain=self.problem.mesh,
            metadata=self.form_compiler_parameters)

        # forms
        self.mesh_V0 = dolfin.assemble(dolfin.Constant(1.) * self.dV)
        self.Psi = -self.J/self.mesh_V0
        # self.Psi = 1/self.J/self.mesh_V0/self.mesh_V0
        self.ener_form = self.Psi * self.dV

        self.DPsi = dolfin.derivative(self.Psi, self.problem.U, self.problem.dU_test)
        self.res_form  = self.DPsi  * self.dV

        self.DDPsi = dolfin.derivative(self.DPsi, self.problem.U, self.problem.dU_trial)
        self.jac_form  = self.DDPsi * self.dV

        self.printer.dec()
        self.printer.dec()
